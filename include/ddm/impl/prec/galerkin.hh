#pragma once

#include "ddm/check.hh"
#include "ddm/coarse_space.hh"
#include "ddm/communication.hh"
#include "ddm/debug/vec_sink.hh"
#include "ddm/index.hh"
#include "ddm/logger.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/prec/prec.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/host_view.hh"

#include <algorithm>
#include <dune/common/parallel/mpitraits.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/solvercategory.hh>
#include <format>
#include <map>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ddm {

namespace detail {
// The rows of the coarse matrix that belong to the basis vectors of one rank. They come in row blocks, one per block of
// the coarse space and in the same order: row block b has one row per vector of block b.
template <class T>
struct CoarseRows {
  struct RowBlock {
    std::vector<Index> cols;             ///< global number of each column of values, increasing
    std::shared_ptr<MultiVec<T>> values; ///< the entries, values->cols() == cols.size()
  };

  Index first_row;              ///< global number of the first row, the row blocks follow each other from there
  std::vector<RowBlock> blocks; ///< one per block of the coarse space
};

// Tags for the messages of neighbour_basis(), distinct from the ones the exchanges of a Communication use
inline constexpr int neighbour_info_tag = 110;
inline constexpr int neighbour_data_tag = 111;

// The basis of a neighbouring rank, on the indices it shares with us
template <class T>
struct NeighbourRows {
  Index m;                           ///< number of basis vectors of the neighbour
  Index global_offset;               ///< global number of the neighbour's first basis vector
  std::vector<Index> indices;        ///< the shared indices, in our numbering
  std::unique_ptr<MultiVec<T>> rows; ///< row k holds the neighbour's basis vectors at indices[k]
};

template <class T>
struct NeighbourBasis {
  Index global_offset;                   ///< global number of our first basis vector
  std::map<int, NeighbourRows<T>> peers; ///< by rank
};

// Finds the indices idx (of the overlapping index set, in any order) that lie in the block. For each of them, adds its
// row in the block to block_rows and its position in idx to positions.
template <class T>
void find_in_block(const typename CoarseSpace<T>::Block& block, std::span<const Index> idx, std::vector<Index>& block_rows, std::vector<Index>& positions)
{
  block_rows.clear();
  positions.clear();
  for (Index p = 0; p < std::ssize(idx); ++p) {
    if (block.covers_all()) {
      block_rows.push_back(idx[p]);
      positions.push_back(p);
      continue;
    }
    const auto it = std::ranges::lower_bound(block.indices, idx[p]);
    if (it != block.indices.end() && *it == idx[p]) {
      block_rows.push_back(static_cast<Index>(it - block.indices.begin()));
      positions.push_back(p);
    }
  }
}

// Finds the indices that two blocks have in common: their rows in block a and in block b
template <class T>
void find_common(const typename CoarseSpace<T>::Block& a, const typename CoarseSpace<T>::Block& b, std::vector<Index>& rows_a, std::vector<Index>& rows_b)
{
  rows_a.clear();
  rows_b.clear();
  if (a.covers_all() || b.covers_all()) { // then a and b are the same block, the only one
    rows_a.resize(a.vectors->rows());
    std::iota(rows_a.begin(), rows_a.end(), Index{0});
    rows_b = rows_a;
    return;
  }
  // Both index lists are ascending
  for (std::size_t i = 0, k = 0; i < a.indices.size() && k < b.indices.size();) {
    if (a.indices[i] < b.indices[k]) ++i;
    else if (b.indices[k] < a.indices[i]) ++k;
    else {
      rows_a.push_back(static_cast<Index>(i++));
      rows_b.push_back(static_cast<Index>(k++));
    }
  }
}

// dst(dst_rows[k], first_col + c) = src(src_rows[k], c) for all k and all columns c of src
template <class T>
void copy_rows(const MultiVec<T>& src, std::span<const Index> src_rows, MultiVec<T>& dst, std::span<const Index> dst_rows, Index first_col)
{
  if (src_rows.empty() || src.cols() == 0) return;
  auto tmp = src.create_multivector(std::ssize(src_rows), src.cols());
  src.pack_rows(src_rows, *tmp);
  dst.unpack_rows(*tmp, dst_rows, first_col);
}

// The rows idx (of the overlapping index set, in any order) of the basis of the rank, i.e. of all blocks extended by
// zero and side by side
template <class T>
std::unique_ptr<MultiVec<T>> basis_rows(const CoarseSpace<T>& cs, std::span<const Index> idx)
{
  auto result = cs.blocks[0].vectors->create_multivector(std::ssize(idx), cs.size());
  std::vector<Index> block_rows;
  std::vector<Index> positions;
  Index first_col = 0;
  for (const auto& block : cs.blocks) {
    find_in_block<T>(block, idx, block_rows, positions);
    copy_rows(*block.vectors, block_rows, *result, positions, first_col);
    first_col += block.vectors->cols();
  }
  return result;
}

/** Returns the bases of our neighbours on the indices they share with us, i.e. R_ij V_j for every neighbour j. Also
 *  determines where the basis of each rank starts in the global numbering of the coarse basis. Collective.
 */
template <class T>
NeighbourBasis<T> neighbour_basis(const CoarseSpace<T>& cs)
{
  const auto& comm = *cs.overlap->comm;
  const auto& pattern = comm.communication_pattern();
  const auto& shared = pattern.reduction_indices();
  MPI_Comm mpi_comm = pattern.communicator();
  const int rank = comm.rank();

  // The number of basis vectors and the global number of the first one, for us and every neighbour
  struct Info {
    Index m;
    Index global_offset;
  };
  Info mine{cs.size(), 0};
  MPI_Exscan(&mine.m, &mine.global_offset, 1, Dune::MPITraits<Index>::getType(), MPI_SUM, mpi_comm);
  if (rank == 0) mine.global_offset = 0; // MPI_Exscan leaves the result on rank 0 undefined

  std::map<int, Info> infos;
  std::vector<MPI_Request> reqs;
  for (const auto& [peer, _] : shared) {
    MPI_Irecv(&infos[peer], sizeof(Info), MPI_BYTE, peer, neighbour_info_tag, mpi_comm, &reqs.emplace_back());
    MPI_Isend(&mine, sizeof(Info), MPI_BYTE, peer, neighbour_info_tag, mpi_comm, &reqs.emplace_back());
  }
  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  reqs.clear();

  // Send the rows of our basis each neighbour shares with us, receive the rows of their bases we share with them. The
  // buffers live on the backend of the basis, the messages go through host views of them.
  // TODO(20261126-093234): Exchange the neighbour basis without going through host views
  NeighbourBasis<T> result;
  result.global_offset = mine.global_offset;
  std::map<int, std::unique_ptr<MultiVec<T>>> send;
  for (const auto& [peer, indices] : shared) {
    const std::vector<Index> send_idx(indices.send_idx.begin(), indices.send_idx.end());
    send[peer] = basis_rows(cs, send_idx);

    auto& nb = result.peers[peer];
    nb.m = infos.at(peer).m;
    nb.global_offset = infos.at(peer).global_offset;
    nb.indices.assign(indices.recv_idx.begin(), indices.recv_idx.end());
    nb.rows = cs.blocks[0].vectors->create_multivector(std::ssize(nb.indices), nb.m);
  }

  {
    const auto mpi_type = Dune::MPITraits<T>::getType();
    std::vector<MultiVecHostView<const T>> send_views;
    std::vector<MultiVecHostView<T>> recv_views;
    for (auto& [peer, nb] : result.peers) {
      auto& recv = recv_views.emplace_back(nb.rows->host_view(write));
      MPI_Irecv(recv.data(), static_cast<int>(recv.size()), mpi_type, peer, neighbour_data_tag, mpi_comm, &reqs.emplace_back());
      const auto& out = send_views.emplace_back(std::as_const(*send.at(peer)).host_view(read));
      MPI_Isend(out.data(), static_cast<int>(out.size()), mpi_type, peer, neighbour_data_tag, mpi_comm, &reqs.emplace_back());
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  } // the views are released here, so the received data is part of the buffers
  return result;
}

/** Returns our rows of the coarse matrix A_0 = R_0 A R_0^T, where R_0^T contains the basis vectors of all ranks,
 *  extended by zero. Collective.
 *
 *  Our basis comes in blocks (see CoarseSpace), and so do our rows. Block j holds the vectors W_j on its indices, and
 *  A_blocks[j] is the matrix on these indices (the restriction of A, e.g. the matrix of the local subdomain). The rows
 *  of block j are
 *
 *    W_j^H A_j [C_k ...],
 *
 *  where each C_k is a group of basis vectors (a block of ours or the basis of a neighbouring rank) restricted to the
 *  indices of block j. Only groups that share indices with block j are used: the others give zero. This only needs the
 *  matrix on the indices of block j, because W_j vanishes on the outermost layer of its indices: only the rows of A_j
 *  on that layer miss columns of A, and W_j^H removes exactly these rows. The result is only correct for such a basis.
 *
 *  The coarse basis vectors are numbered globally rank by rank, and within a rank block by block. Row block j holds
 *  the rows of the vectors of block j; values is dense, and cols[c] is the global number of its column c.
 */
template <class T>
CoarseRows<T> coarse_matrix_rows(std::span<const LocalMat<T>* const> A_blocks, const CoarseSpace<T>& cs)
{
  DDM_CHECK(A_blocks.size() == cs.blocks.size(), "coarse_matrix_rows: {} matrices for {} blocks", A_blocks.size(), cs.blocks.size());
  const auto neighbours = neighbour_basis(cs);

  // The global number of the first vector of each of our blocks
  std::vector<Index> block_offset(cs.blocks.size());
  Index offset = neighbours.global_offset;
  for (std::size_t k = 0; k < cs.blocks.size(); ++k) {
    block_offset[k] = offset;
    offset += cs.blocks[k].vectors->cols();
  }

  // A group of basis vectors that can couple with block j: src holds the vectors, rows src_rows of src are the rows
  // dst_rows of block j
  struct Group {
    Index global_offset;
    const MultiVec<T>* src;
    std::vector<Index> src_rows;
    std::vector<Index> dst_rows;
  };

  CoarseRows<T> rows;
  rows.first_row = neighbours.global_offset;
  for (std::size_t j = 0; j < cs.blocks.size(); ++j) {
    const auto& W = *cs.blocks[j].vectors;
    const auto& A_j = *A_blocks[j];
    DDM_CHECK(A_j.rows() == W.rows() && A_j.cols() == W.rows(), "coarse_matrix_rows: matrix {} is {}x{}, but block {} has {} rows", j, A_j.rows(), A_j.cols(), j, W.rows());

    auto& row_block = rows.blocks.emplace_back();
    if (W.cols() == 0) {
      row_block.values = W.create_multivector(0, 0);
      continue;
    }

    std::vector<Group> groups;
    for (const auto& [peer, nb] : neighbours.peers) {
      Group g{nb.global_offset, nb.rows.get(), {}, {}};
      find_in_block<T>(cs.blocks[j], nb.indices, g.dst_rows, g.src_rows);
      if (!g.dst_rows.empty() && nb.m > 0) groups.push_back(std::move(g));
    }
    for (std::size_t k = 0; k < cs.blocks.size(); ++k) {
      Group g{block_offset[k], cs.blocks[k].vectors.get(), {}, {}};
      find_common<T>(cs.blocks[j], cs.blocks[k], g.dst_rows, g.src_rows);
      if (!g.dst_rows.empty() && g.src->cols() > 0) groups.push_back(std::move(g));
    }
    std::ranges::sort(groups, {}, &Group::global_offset);

    // C = [C_k ...], the groups side by side on the rows of block j
    Index cols = 0;
    for (const auto& g : groups) cols += g.src->cols();
    auto C = W.create_multivector(W.rows(), cols);
    row_block.cols.reserve(cols);
    for (const auto& g : groups) {
      copy_rows(*g.src, g.src_rows, *C, g.dst_rows, std::ssize(row_block.cols));
      for (Index c = 0; c < g.src->cols(); ++c) row_block.cols.push_back(g.global_offset + c);
    }

    // values = W_j^H A_j C, with a single pass over the matrix for all groups
    auto AC = A_j.create_range_multivector(cols);
    A_j.spmm(*C, *AC);
    row_block.values = W.create_multivector(W.cols(), cols);
    W.dot(*AC, *row_block.values);
  }
  return rows;
}
} // namespace detail

/** Coarse correction z = R_0^T A_0^{-1} R_0 r, where R_0^T contains the coarse basis vectors of all subdomains,
 *  extended by zero, and A_0 = R_0 A R_0^T is the coarse matrix (see detail::coarse_matrix_rows()).
 *
 *  The coarse matrix is gathered on rank 0 and the coarse problem is solved there with a sequential solver. Not
 *  registered: the coarse space has to be built by someone else (e.g. a two-level preconditioner).
 *
 *  Config:
 *  - mat:    config of the coarse matrix (see create_local_mat_like(), default: the type of A's local matrix)
 *  - solver: config of the solver for the coarse problem (see create_subproblem_solver(), default: exact solve)
 *  - prec:   config of the preconditioner of that solver (see create_subproblem_solver())
 */
template <class T>
class CoarseCorrection final : public Prec<T> {
public:
  // A_blocks[b] is the restriction of A to the indices of block b of the coarse space (with a single block, to the
  // whole overlapping index set). They are only used in the constructor. Collective
  CoarseCorrection(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, std::span<const LocalMat<T>* const> A_blocks, std::shared_ptr<const CoarseSpace<T>> cs)
      : Prec<T>(std::move(A))
      , cs_(std::move(cs))
      , apply_event_(Logger::get().registerOrGetEvent("Coarse correction", "apply"))
      , coarse_solve_event_(Logger::get().registerOrGetEvent("Coarse correction", "coarse solve"))
      , factorize_event_(Logger::get().registerOrGetEvent("Coarse correction", "factorize"))
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Coarse correction", "setup")};
    DDM_CHECK(cs_->blocks.size() == 1 && cs_->blocks[0].covers_all(), "coarse correction: only a single block that covers all indices is supported so far");
    const auto& V = *cs_->blocks[0].vectors;
    DDM_CHECK(cs_->overlap->n_original == this->mat()->rows(), "coarse correction: the coarse space was built for {} indices, but the matrix has {} rows", cs_->overlap->n_original,
              this->mat()->rows());

    d_.emplace(create_vector(V, V.rows()));
    x_.emplace(create_vector(V, V.rows()));
    c_.emplace(create_vector(V, V.cols()));

    setup_coarse_problem(config, detail::coarse_matrix_rows(A_blocks, *cs_));

    // Visualise the basis vectors if the user has installed a vector sink
    auto save_vectors = config.get("save_vectors", -1);
    if (save_vectors > 0 and debug::has_vec_sink<T>()) {
      auto v = create_vector(V, V.rows());
      auto v_novlp = this->mat()->create_domain_vector();
      for (Index i = 0; i < std::min<Index>(cs_->size(), save_vectors); ++i) {
        v.zero();
        if (debug::vec_sink_debug_rank<T>() == this->mat()->communication()->rank()) v = column(*cs_->blocks[0].vectors, i);
        cs_->overlap->comm->reduce(v);
        v_novlp.copy_n_from(v, v_novlp.size());
        debug::add_vec_to_sink(std::format("basis{:04d}", i), v_novlp);
      }
    }
  }

  Dune::SolverCategory::Category category() const override
  {
    return this->mat()->category();
  }

private:
  // Collective
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    Logger::ScopedLog sl{apply_event_};
    const auto& V = *cs_->blocks[0].vectors;
    const auto& comm = *cs_->overlap->comm;
    const auto n = cs_->overlap->n_original;

    // r is consistent, so the owners of the overlap indices have the right values
    d_->copy_n_from(r, n);
    comm.broadcast(*d_);

    // Restriction: our part of R_0 r is V_i^H d
    V.dot(d_->multivec(), c_->multivec());

    // Solve the coarse problem on rank 0 and hand everyone its part of the solution
    gather_coarse_defect();
    if (rank_ == 0) {
      Logger::ScopedLog sl_solve{coarse_solve_event_};
      *x0_ = T{0};
      solver_->solve(*d0_, *x0_);
    }
    scatter_coarse_solution();

    // Prolongation: R_0^T x_0 is the sum of V_i x_i over all subdomains, which the reduction makes consistent
    V.mult(c_->multivec(), x_->multivec());
    comm.reduce(*x_);
    z.copy_n_from(*x_, n);
  }

  // The coarse space would have to be rebuilt too, so the owner of the coarse space creates a new coarse correction
  void do_update() override
  {
    TODO("CoarseCorrection::do_update");
  }

  void do_info() const override
  {
    logger::info("Coarse correction, solved on rank 0");
    if (!solver_) return; // the coarse problem only exists on rank 0

    logger::increase_indent();
    logger::info("Coarse problem size: {}", d0_->size());
    logger::info("Coarse solver info");
    logger::increase_indent();
    solver_->info();
    logger::decrease_indent();
    logger::decrease_indent();
  }

  // Gathers the coarse matrix on rank 0 and sets up the coarse solver there. Collective
  void setup_coarse_problem(const Dune::ParameterTree& config, const detail::CoarseRows<T>& rows)
  {
    MPI_Comm mpi_comm = cs_->overlap->comm->communication_pattern().communicator();
    MPI_Comm_rank(mpi_comm, &rank_);
    int size{};
    MPI_Comm_size(mpi_comm, &size);

    // Our rows come in row blocks (see detail::coarse_matrix_rows()). Every rank sends the number of its rows, the shape
    // (rows and columns) of every row block, the columns of all row blocks, and their values (column by column, one row
    // block after the other). The rows are numbered rank by rank, so the row counts also give the position of every
    // rank's part of the coarse vectors.
    // TODO(20261126-093234): Avoid the host views
    std::vector<Index> shapes;
    std::vector<Index> cols;
    std::vector<T> values;
    Index n_rows = 0;
    for (const auto& row_block : rows.blocks) {
      shapes.push_back(row_block.values->rows());
      shapes.push_back(row_block.values->cols());
      n_rows += row_block.values->rows();
      cols.insert(cols.end(), row_block.cols.begin(), row_block.cols.end());
      const auto view = std::as_const(*row_block.values).host_view(read);
      values.insert(values.end(), view.data(), view.data() + view.size());
    }

    const Index sizes[4] = {n_rows, static_cast<Index>(shapes.size()), static_cast<Index>(cols.size()), static_cast<Index>(values.size())};
    std::vector<Index> all_sizes(rank_ == 0 ? 4 * size : 0);
    MPI_Gather(sizes, 4, Dune::MPITraits<Index>::getType(), all_sizes.data(), 4, Dune::MPITraits<Index>::getType(), 0, mpi_comm);

    // counts[k][p] and displs[k][p] for the shapes (k = 0), columns (1) and values (2) of rank p
    std::vector<int> counts[3];
    std::vector<int> displs[3];
    if (rank_ == 0) {
      row_counts_.assign(size, 0);
      row_displs_.assign(size, 0);
      for (int k = 0; k < 3; ++k) {
        counts[k].assign(size, 0);
        displs[k].assign(size, 0);
      }
      for (int p = 0; p < size; ++p) {
        row_counts_[p] = static_cast<int>(all_sizes[4 * p]);
        for (int k = 0; k < 3; ++k) counts[k][p] = static_cast<int>(all_sizes[4 * p + 1 + k]);
        if (p > 0) {
          row_displs_[p] = row_displs_[p - 1] + row_counts_[p - 1];
          for (int k = 0; k < 3; ++k) displs[k][p] = displs[k][p - 1] + counts[k][p - 1];
        }
      }
    }

    const auto gather = [&]<class U>(const std::vector<U>& local, int k) {
      std::vector<U> all(rank_ == 0 ? displs[k].back() + counts[k].back() : 0);
      MPI_Gatherv(local.data(), static_cast<int>(local.size()), Dune::MPITraits<U>::getType(), all.data(), counts[k].data(), displs[k].data(), Dune::MPITraits<U>::getType(), 0, mpi_comm);
      return all;
    };
    const auto all_shapes = gather(shapes, 0);
    const auto all_cols = gather(cols, 1);
    const auto all_values = gather(values, 2);

    if (rank_ != 0) return;

    // Calls f(rows, cols, values) for every row block of every rank, with the global numbers of its rows and columns
    // and its values column by column
    const auto for_each_row_block = [&](auto&& f) {
      std::vector<Index> block_rows;
      for (int p = 0; p < size; ++p) {
        Index row = row_displs_[p];
        const Index* col = all_cols.data() + displs[1][p];
        const T* value = all_values.data() + displs[2][p];
        for (int b = 0; b < counts[0][p] / 2; ++b) {
          const Index r = all_shapes[displs[0][p] + 2 * b];
          const Index c = all_shapes[displs[0][p] + 2 * b + 1];
          block_rows.resize(r);
          std::iota(block_rows.begin(), block_rows.end(), row);
          f(std::span<const Index>(block_rows), std::span<const Index>(col, c), std::span<const T>(value, r * c));
          row += r;
          col += c;
          value += r * c;
        }
      }
    };

    const Index n0 = row_displs_.back() + row_counts_.back();
    Pattern pattern(n0, n0);
    for_each_row_block([&](std::span<const Index> block_rows, std::span<const Index> block_cols, std::span<const T>) {
      for (auto r : block_rows)
        for (auto c : block_cols) pattern.add(r, c);
    });
    pattern.finalize();

    // The values of a row block are stored column by column like add_values() expects it
    auto A0_local = create_local_mat_like<T>(config.sub("mat"), pattern, this->mat()->local());
    for_each_row_block([&](std::span<const Index> block_rows, std::span<const Index> block_cols, std::span<const T> block_values) {
      if (!block_rows.empty() && !block_cols.empty()) A0_local->add_values(block_rows, block_cols, block_values);
    });
    A0_local->assemble();

    // The coarse matrix only lives on rank 0. It gets a communication on MPI_COMM_SELF that owns all coarse dofs, so
    // that it can also be solved with a preconditioner that needs one, e.g. a (two-level) Schwarz preconditioner with
    // several local subdomains, which gives a three-level method
    std::vector<CommunicationNodes> roots(n0);
    for (Index i = 0; i < n0; ++i) roots[i] = {0, static_cast<std::int64_t>(i)};
    auto A0 = std::make_shared<const Mat<T>>(config.sub("mat"), A0_local, std::make_shared<Communication>(MPI_COMM_SELF, roots));
    Logger::get().set_stat("Coarse correction/coarse dofs", static_cast<double>(n0));
    Logger::get().set_stat("Coarse correction/coarse nnz", static_cast<double>(pattern.nnz()));
    Logger::ScopedLog sl{factorize_event_};
    solver_ = create_subproblem_solver<T>(config.sub("solver"), config.sub("prec"), A0);
    d0_.emplace(A0->create_range_vector());
    x0_.emplace(A0->create_domain_vector());
  }

  // Collects every rank's part of R_0 r (in c_) in d0_ on rank 0. Collective
  // TODO(20261126-093234): Avoid the host views, this is done in every apply()
  void gather_coarse_defect()
  {
    MPI_Comm mpi_comm = cs_->overlap->comm->communication_pattern().communicator();
    const auto c = std::as_const(*c_).host_view(read);
    if (rank_ == 0) {
      auto d0 = d0_->host_view(write);
      MPI_Gatherv(c.data(), static_cast<int>(c.size()), Dune::MPITraits<T>::getType(), d0.data(), row_counts_.data(), row_displs_.data(), Dune::MPITraits<T>::getType(), 0, mpi_comm);
    }
    else {
      MPI_Gatherv(c.data(), static_cast<int>(c.size()), Dune::MPITraits<T>::getType(), nullptr, nullptr, nullptr, Dune::MPITraits<T>::getType(), 0, mpi_comm);
    }
  }

  // Hands every rank its part of the coarse solution x0_ on rank 0, in c_. Collective
  // TODO(20261126-093234): Avoid the host views, this is done in every apply()
  void scatter_coarse_solution()
  {
    MPI_Comm mpi_comm = cs_->overlap->comm->communication_pattern().communicator();
    auto c = c_->host_view(write);
    if (rank_ == 0) {
      const auto x0 = std::as_const(*x0_).host_view(read);
      MPI_Scatterv(x0.data(), row_counts_.data(), row_displs_.data(), Dune::MPITraits<T>::getType(), c.data(), static_cast<int>(c.size()), Dune::MPITraits<T>::getType(), 0, mpi_comm);
    }
    else {
      MPI_Scatterv(nullptr, nullptr, nullptr, Dune::MPITraits<T>::getType(), c.data(), static_cast<int>(c.size()), Dune::MPITraits<T>::getType(), 0, mpi_comm);
    }
  }

  std::shared_ptr<const CoarseSpace<T>> cs_;
  int rank_{};

  std::optional<Vec<T>> d_; ///< defect on the overlapping index set
  std::optional<Vec<T>> x_; ///< correction on the overlapping index set
  std::optional<Vec<T>> c_; ///< our part of the coarse defect and of the coarse solution

  // Only on rank 0
  std::vector<int> row_counts_;       ///< number of coarse rows (= basis vectors) of every rank
  std::vector<int> row_displs_;       ///< global number of the first coarse row of every rank
  std::shared_ptr<Solver<T>> solver_; ///< solver for the coarse problem
  std::optional<Vec<T>> d0_;          ///< coarse defect
  std::optional<Vec<T>> x0_;          ///< coarse solution

  Logger::Event* apply_event_;
  Logger::Event* coarse_solve_event_; ///< only used on rank 0
  Logger::Event* factorize_event_;    ///< only used on rank 0
};
} // namespace ddm
