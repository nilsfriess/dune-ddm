#pragma once

#include <spdlog/spdlog.h>

#include <dune/istl/bcrsmatrix.hh>

#include <mpi.h>
#include <memory>
#include <utility>
#include <vector>

struct TripleWithRank {
  int rank;
  std::size_t row;
  std::size_t col;
  double val;
};

enum class Attribute : std::uint8_t { owner, copy };

class AttributeSet {
public:
  std::set<Attribute> subset;
  using Type = Attribute;

  AttributeSet() = default;
  explicit AttributeSet(const std::set<Attribute> &s) : subset(s) {}
  AttributeSet(const std::initializer_list<Attribute> &list) : subset(list) {}

  bool contains(const Type &attribute) const { return subset.find(attribute) != subset.end(); }
};

inline std::ostream &operator<<(std::ostream &out, Attribute attribute)
{
  out << (attribute == Attribute::owner ? "owner" : "copy");
  return out;
}

template <class RemoteIndices>
using RemoteParallelIndices = std::pair<std::shared_ptr<RemoteIndices>, std::shared_ptr<typename RemoteIndices::ParallelIndexSet>>;

template <class RemoteIndices>
RemoteParallelIndices<RemoteIndices> makeRemoteParallelIndices(std::shared_ptr<RemoteIndices> ri)
{
  auto paridxs = std::make_shared<typename RemoteIndices::ParallelIndexSet>(ri->sourceIndexSet());
  std::vector<int> nbs(ri->getNeighbours().begin(), ri->getNeighbours().end());
  ri->setIndexSets(*paridxs, *paridxs, ri->communicator(), nbs);
  ri->template rebuild<false>();
  return std::make_pair(ri, paridxs);
}

/** @brief Builds a matrix on rank zero from rows owned by the MPI ranks.

    The \p rows parameter contains the rows that the current rank wants to insert into the matrix.
    The number of rows each MPI rank contributes is allowed to differ. The size of the individual rows
    must be the same on all ranks.

    This function must be called collectively on the communicator \p comm. On rank 0 the returned
    matrix will contain all the rows; on all other ranks the returned matrix is empty.

    Only values larger than (in absolute value) \p clip_tolerance will end up in the matrix.
    Pass a negative value to include all values.
*/
template <class Vec>
Dune::BCRSMatrix<double> gatherMatrixFromRows(const std::vector<Vec> &rows, MPI_Comm comm, int verbose = 0, double clip_tolerance = 0)
{
  int rank = 0;
  int size = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  if (rows.size() == 0) {
    DUNE_THROW(Dune::Exception, "No rows to build matrix from");
  }

  std::size_t columns = rows[0].N();
  for (const auto &row : rows) {
    if (row.N() != columns) {
      DUNE_THROW(Dune::Exception, "Rows have different sizes");
    }
  }

  // Check that all rows on all ranks have the same size
  int min_columns = 0;
  int max_columns = 0;
  MPI_Allreduce(&columns, &min_columns, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(&columns, &max_columns, 1, MPI_INT, MPI_MAX, comm);
  if (min_columns != max_columns) {
    DUNE_THROW(Dune::Exception, "Rows have different sizes");
  }

  constexpr int nitems = 4;
  std::array<int, nitems> blocklengths = {1, 1, 1, 1};
  std::array<MPI_Datatype, nitems> types = {MPI_INT, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_DOUBLE};
  MPI_Datatype triple_type = MPI_DATATYPE_NULL;
  std::array<MPI_Aint, nitems> offsets{0};
  offsets[0] = offsetof(TripleWithRank, rank);
  offsets[1] = offsetof(TripleWithRank, row);
  offsets[2] = offsetof(TripleWithRank, col);
  offsets[3] = offsetof(TripleWithRank, val);
  MPI_Type_create_struct(nitems, blocklengths.data(), offsets.data(), types.data(), &triple_type);
  MPI_Type_commit(&triple_type);

  std::vector<TripleWithRank> my_triples;
  my_triples.reserve(columns * rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto &row = rows[i];
    for (std::size_t col = 0; col < columns; ++col) {
      if (std::abs(row[col]) > clip_tolerance) {
        my_triples.push_back({rank, i, col, row[col]});
      }
    }
  }

  // Now gather the number of local triples from each process
  std::vector<std::size_t> num_triples;
  if (rank == 0) {
    num_triples.resize(size);
  }
  auto num_my_triples = my_triples.size();
  MPI_Gather(&num_my_triples, 1, MPI_UNSIGNED_LONG, rank == 0 ? num_triples.data() : nullptr, 1, MPI_UNSIGNED_LONG, 0, comm);

  std::vector<int> displacements;
  if (rank == 0) {
    std::vector<std::size_t> displacements_sizet(size);
    std::exclusive_scan(num_triples.begin(), num_triples.end(), displacements_sizet.begin(), 0);
    displacements.resize(size);
    std::transform(displacements_sizet.begin(), displacements_sizet.end(), displacements.begin(), [](auto &&v) { return static_cast<int>(v); });

    for (std::size_t i = 0; i < num_triples.size(); ++i) {
      spdlog::trace("In gatherMatrixFromRows: From rank {} got {} triples", i, num_triples[i]);
    }
  }

  std::vector<TripleWithRank> all_triples;
  if (rank == 0) {
    auto sum = std::reduce(num_triples.begin(), num_triples.end());
    all_triples.resize(sum);

    spdlog::debug("Total {} nonzeros in matrix built on rank 0", sum);
  }
  std::vector<int> num_triples_int(num_triples.size());
  std::transform(num_triples.begin(), num_triples.end(), num_triples_int.begin(), [](auto &&v) { return static_cast<int>(v); });
  MPI_Gatherv(my_triples.data(), static_cast<int>(my_triples.size()), triple_type, all_triples.data(), num_triples_int.data(), displacements.data(), triple_type, 0, comm);

  Dune::BCRSMatrix<double> A0;

  if (rank == 0) {
    // Compute offsets of rows for each rank
    std::map<int, std::size_t> rows_per_rank;
    for (const auto &triple : all_triples) {
      rows_per_rank[triple.rank] = std::max(rows_per_rank[triple.rank], triple.row + 1);
    }

    std::vector<std::size_t> row_offsets(size);
    row_offsets[0] = 0;
    for (int i = 1; i < size; ++i) {
      row_offsets[i] = row_offsets[i - 1] + rows_per_rank[i - 1];
    }

    A0.setBuildMode(Dune::BCRSMatrix<double>::implicit);
    A0.setImplicitBuildModeParameters(all_triples.size() / size, 0.2); // TODO: Make this robust
    A0.setSize(row_offsets[size - 1] + rows_per_rank[size - 1], columns);

    for (const auto &triple : all_triples) {
      A0.entry(row_offsets[triple.rank] + triple.row, triple.col) = triple.val;
    }
    A0.compress();
  }

  MPI_Type_free(&triple_type);
  return A0;
}

/** @brief Overload for special case of one vector per rank */
template <class Vec>
Dune::BCRSMatrix<double> gatherMatrixFromRows(const Vec &row, MPI_Comm comm, int verbose = 0, double clip_tolerance = 0)
{
  return gatherMatrixFromRows(std::vector<Vec>{row}, comm, verbose, clip_tolerance);
}
