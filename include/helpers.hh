#pragma once

#include <algorithm>
#include <cstddef>
#include <spdlog/spdlog.h>

#include <dune/common/parallel/mpitraits.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <memory>
#include <mpi.h>
#include <utility>
#include <vector>

#include <iostream>

#define MPI_CHECK(call)                                                                                                                                                                                \
  do {                                                                                                                                                                                                 \
    int err = (call);                                                                                                                                                                                  \
    if (err != MPI_SUCCESS) {                                                                                                                                                                          \
      char err_string[MPI_MAX_ERROR_STRING];                                                                                                                                                           \
      int resultlen;                                                                                                                                                                                   \
      MPI_Error_string(err, err_string, &resultlen);                                                                                                                                                   \
      std::cerr << "MPI error at " << __FILE__ << ":" << __LINE__ << " - " << err_string << std::endl;                                                                                                 \
      MPI_Abort(MPI_COMM_WORLD, err);                                                                                                                                                                  \
    }                                                                                                                                                                                                  \
  } while (0)

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
Dune::BCRSMatrix<double> gatherMatrixFromRows(const std::vector<Vec> &rows, MPI_Comm comm, double clip_tolerance = 0)
{
  int rank = 0;
  int size = 0;
  MPI_CHECK(MPI_Comm_size(comm, &size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));

  if (rows.size() == 0) {
    DUNE_THROW(Dune::Exception, "No rows to build matrix from");
  }

  std::size_t columns = rows[0].N();
  for (const auto &row : rows) {
    if (row.N() != columns) {
      DUNE_THROW(Dune::Exception, "Rows have different sizes");
    }

    if (row.two_norm() == 0) {
      DUNE_THROW(Dune::Exception, "Must not provide rows that are zero");
    }
  }

  // Check that all rows on all ranks have the same size
  int min_columns = 0;
  int max_columns = 0;
  MPI_CHECK(MPI_Allreduce(&columns, &min_columns, 1, MPI_INT, MPI_MIN, comm));
  MPI_CHECK(MPI_Allreduce(&columns, &max_columns, 1, MPI_INT, MPI_MAX, comm));
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
  MPI_CHECK(MPI_Type_create_struct(nitems, blocklengths.data(), offsets.data(), types.data(), &triple_type));
  MPI_CHECK(MPI_Type_commit(&triple_type));

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
  MPI_CHECK(MPI_Gather(&num_my_triples, 1, MPI_UNSIGNED_LONG, rank == 0 ? num_triples.data() : nullptr, 1, MPI_UNSIGNED_LONG, 0, comm));

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
  MPI_CHECK(MPI_Gatherv(my_triples.data(), static_cast<int>(my_triples.size()), triple_type, all_triples.data(), num_triples_int.data(), displacements.data(), triple_type, 0, comm));

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
    A0.setImplicitBuildModeParameters(all_triples.size() / size, 1); // TODO: Make this robust
    A0.setSize(row_offsets[size - 1] + rows_per_rank[size - 1], columns);

    for (const auto &triple : all_triples) {
      A0.entry(row_offsets[triple.rank] + triple.row, triple.col) = triple.val;
    }
    A0.compress();
  }
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  MPI_CHECK(MPI_Type_free(&triple_type));
  return A0;
}

/** @brief Overload for special case of one vector per rank */
template <class Vec>
Dune::BCRSMatrix<double> gatherMatrixFromRows(const Vec &row, MPI_Comm comm, double clip_tolerance = 0)
{
  return gatherMatrixFromRows(std::vector<Vec>{row}, comm, clip_tolerance);
}

/** @brief Variant of gatherMatrixFromRows where the rows are passed in a column major 1d array.

    The parameter \p n_cols is the length of the individual rows, the number of rows is inferred from the \p rows array.
*/
inline Dune::BCRSMatrix<double> gatherMatrixFromRowsFlat(const std::vector<double> &rows, std::size_t n_cols, MPI_Comm comm, double clip_tolerance = 0)
{
  int rank = 0;
  int size = 0;
  MPI_CHECK(MPI_Comm_size(comm, &size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));

  if (rows.size() == 0) {
    DUNE_THROW(Dune::Exception, "No rows to build matrix from");
  }

  if (rows.size() % n_cols != 0) {
    DUNE_THROW(Dune::Exception, "Rows size is not a multiple of the number of columns");
  }
  const auto n_rows = rows.size() / n_cols;

  // Check that all rows on all ranks have the same size
  std::size_t min_columns = 0;
  std::size_t max_columns = 0;
  MPI_Datatype size_t_type = Dune::MPITraits<std::size_t>::getType();
  MPI_CHECK(MPI_Allreduce(&n_cols, &min_columns, 1, size_t_type, MPI_MIN, comm));
  MPI_CHECK(MPI_Allreduce(&n_cols, &max_columns, 1, size_t_type, MPI_MAX, comm));
  if (min_columns != max_columns) {
    DUNE_THROW(Dune::Exception, "Rows have different sizes");
  }

  // Now we convert the values in rows into CSR format that rank 0 can then directly assemble.
  // We create the data structures as if the matrix was sequential, rank 0 is responsible for
  // figuring out the correct row offsets.
  std::vector<std::size_t> row_offsets;
  std::vector<std::size_t> col_indices;
  std::vector<double> values;

  // We start by counting the nonzeros and sending this (along with the number of rows we own) to rank 0
  std::array<std::size_t, 2> nnz_and_n_rows{};
  nnz_and_n_rows[0] = std::count_if(rows.begin(), rows.end(), [clip_tolerance](auto x) { return std::abs(x) > clip_tolerance; });
  nnz_and_n_rows[1] = n_rows;

  std::vector<std::size_t> nnz_and_n_rows_data(rank == 0 ? 2 * size : 0);
  MPI_Request req = MPI_REQUEST_NULL;
  MPI_CHECK(MPI_Igather(nnz_and_n_rows.data(), 2, size_t_type, nnz_and_n_rows_data.data(), 2, size_t_type, 0, comm, &req));

  // During the send, we create the CSR data structures. Note that the rows array is in column-major order,
  row_offsets.reserve(n_rows + 1);
  col_indices.reserve(nnz_and_n_rows[0]);
  values.reserve(nnz_and_n_rows[0]);
  row_offsets.push_back(0);
  std::size_t nnz_count = 0;
  for (std::size_t row = 0; row < n_rows; ++row) {
    for (std::size_t col = 0; col < n_cols; ++col) {
      auto value = rows[row + col * n_rows];
      if (std::abs(value) > clip_tolerance) {
        col_indices.push_back(col);
        values.push_back(value);
        ++nnz_count;
      }
    }
    row_offsets.push_back(nnz_count);
  }

  // Wait for the Gather to finish
  MPI_CHECK(MPI_Wait(&req, MPI_STATUSES_IGNORE));

  // Now send the CSR data to rank 0
  std::size_t total_nnz = 0;
  std::size_t total_n_rows = 0;
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      total_nnz += nnz_and_n_rows_data[2 * i];
      total_n_rows += nnz_and_n_rows_data[2 * i + 1];
    }
  }
  std::vector<std::size_t> global_row_offsets;
  std::vector<std::size_t> global_col_indices;
  std::vector<double> global_values;
  if (rank == 0) {
    global_row_offsets.resize(total_n_rows + 1); // +1 because we need the last offset for the end of the last row
    global_col_indices.resize(total_nnz);
    global_values.resize(total_nnz);
  }

  std::vector<int> col_values_displacements(size);
  std::vector<int> row_offsets_displacements(size);
  std::vector<int> col_values_counts(size);
  std::vector<int> row_offsets_counts(size);
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      col_values_counts[i] = static_cast<int>(nnz_and_n_rows_data[2 * i]);
      row_offsets_counts[i] = static_cast<int>(nnz_and_n_rows_data[2 * i + 1]);
    }

    std::exclusive_scan(col_values_counts.begin(), col_values_counts.end(), col_values_displacements.begin(), 0);
    std::exclusive_scan(row_offsets_counts.begin(), row_offsets_counts.end(), row_offsets_displacements.begin(), 0);
  }
  MPI_CHECK(
      MPI_Gatherv(col_indices.data(), static_cast<int>(col_indices.size()), size_t_type, global_col_indices.data(), col_values_counts.data(), col_values_displacements.data(), size_t_type, 0, comm));
  MPI_CHECK(MPI_Gatherv(row_offsets.data(), static_cast<int>(row_offsets.size() - 1), size_t_type, global_row_offsets.data(), row_offsets_counts.data(), row_offsets_displacements.data(), size_t_type,
                        0, comm));
  MPI_CHECK(MPI_Gatherv(values.data(), static_cast<int>(values.size()), MPI_DOUBLE, global_values.data(), col_values_counts.data(), col_values_displacements.data(), MPI_DOUBLE, 0, comm));

  if (rank == 0) {
    // Set the last row offset to the total number of nonzeros
    global_row_offsets[total_n_rows] = total_nnz;

    // Accumulate all nonzero numbers over all ranks. This is necessary because the row offsets are relative to the local rows,
    // so we have to adjust them to be relative to the global rows.
    std::vector<std::size_t> row_offsets_accumulated(size);
    row_offsets_accumulated[0] = 0;
    for (int i = 1; i < size; ++i) {
      row_offsets_accumulated[i] = row_offsets_accumulated[i - 1] + col_values_counts[i - 1];
    }

    // Now we have to adjust the row offsets to be relative to the global rows by adding the accumulated offsets.
    for (int s = 0; s < size; ++s) {
      for (std::size_t i = 0; i < row_offsets_counts[s]; ++i) {
        global_row_offsets[row_offsets_displacements[s] + i] += row_offsets_accumulated[s];
      }
    }

    // Now we can build the matrix
    Dune::BCRSMatrix<double> A0;
    A0.setBuildMode(Dune::BCRSMatrix<double>::implicit);
    A0.setImplicitBuildModeParameters(total_nnz / size, 1); // TODO: Make this robust
    A0.setSize(total_n_rows, n_cols);

    for (std::size_t i = 0; i < total_n_rows; ++i) {
      for (std::size_t j = global_row_offsets[i]; j < global_row_offsets[i + 1]; ++j) {
        A0.entry(i, global_col_indices[j]) = global_values[j];
      }
    }

    A0.compress();

    return A0;
  }

  return Dune::BCRSMatrix<double>{};
}
