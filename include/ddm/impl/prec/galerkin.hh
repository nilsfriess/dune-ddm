#pragma once

#include "ddm/check.hh"
#include "ddm/coarse_space.hh"
#include "ddm/communication.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/prec/prec.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/host_view.hh"
#include "dune/ddm/logger.hh"

#include <dune/common/parallel/mpitraits.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/solvercategory.hh>
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
// The rows of the coarse matrix that belong to the basis vectors of one rank
template <class T>
struct CoarseRows {
  Index first_row;                     ///< global number of the first row, the rows are first_row, ..., first_row + values->rows() - 1
  std::vector<Index> cols;             ///< global number of each column of values
  std::shared_ptr<MultiVec<T>> values; ///< the entries, values->cols() == cols.size()
};

// Tags for the messages of neighbour_basis(), distinct from the ones the exchanges of a Communication use
inline constexpr int neighbour_info_tag = 110;
inline constexpr int neighbour_data_tag = 111;

// The columns of NeighbourBasis::columns that belong to the basis of one rank
struct NeighbourBasisBlock {
  Index first;         ///< first column of the block
  Index end;           ///< one past the last column of the block
  Index global_offset; ///< global number of the first basis vector of the block's rank
};

template <class T>
struct NeighbourBasis {
  std::shared_ptr<MultiVec<T>> columns;      ///< the blocks R_ij V_j side by side, one row per index of the overlapping index set
  std::map<int, NeighbourBasisBlock> blocks; ///< by rank, including our own block (R_ii V_i = V_i)
};

/** Returns the basis vectors of all subdomains that intersect ours, restricted to our subdomain: block j is R_ij V_j,
 *  i.e. the basis of rank j on the indices both hold, zero elsewhere. Our own basis V is one of the blocks. Also
 *  determines where the basis of each rank starts in the global numbering of the coarse basis. Collective.
 *
 *  comm is the communication of the overlapping index set the basis is defined on.
 */
template <class T>
NeighbourBasis<T> neighbour_basis(const Communication& comm, const MultiVec<T>& V)
{
  DDM_CHECK(V.rows() == static_cast<Index>(comm.roots().size()), "neighbour_basis: basis has {} rows, but the index set has size {}", V.rows(), comm.roots().size());

  const auto& pattern = comm.communication_pattern();
  const auto& shared = pattern.reduction_indices();
  MPI_Comm mpi_comm = pattern.communicator();
  const int rank = comm.rank();

  // The number of basis vectors and the global number of the first one, for us and every neighbour
  struct Info {
    Index m;
    Index global_offset;
  };
  std::map<int, Info> infos;

  auto& mine = infos[rank];
  mine.m = V.cols();
  MPI_Exscan(&mine.m, &mine.global_offset, 1, Dune::MPITraits<Index>::getType(), MPI_SUM, mpi_comm);
  if (rank == 0) mine.global_offset = 0; // MPI_Exscan leaves the result on rank 0 undefined

  std::vector<MPI_Request> reqs;
  for (const auto& [peer, _] : shared) {
    MPI_Irecv(&infos[peer], sizeof(Info), MPI_BYTE, peer, neighbour_info_tag, mpi_comm, &reqs.emplace_back());
    MPI_Isend(&mine, sizeof(Info), MPI_BYTE, peer, neighbour_info_tag, mpi_comm, &reqs.emplace_back());
  }
  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  reqs.clear();

  // The blocks, side by side in the order of the ranks
  NeighbourBasis<T> result;
  Index cols = 0;
  for (const auto& [block_rank, info] : infos) {
    result.blocks[block_rank] = {cols, cols + info.m, info.global_offset};
    cols += info.m;
  }
  result.columns = V.create_multivector(V.rows(), cols);

  // Our own block is V itself
  std::vector<Index> all(V.rows());
  std::iota(all.begin(), all.end(), Index{0});
  result.columns->unpack_rows(V, all, result.blocks.at(rank).first);

  // Send the rows of V each neighbour shares with us, receive the rows of their bases we share with them. The buffers
  // live on the backend of V, the messages go through host views of them.
  // TODO(20261007-103129): Exchange the neighbour basis without going through host views
  struct Buffers {
    std::vector<Index> send_idx;
    std::vector<Index> recv_idx;
    std::unique_ptr<MultiVec<T>> send;
    std::unique_ptr<MultiVec<T>> recv;
  };
  std::map<int, Buffers> buffers;
  for (const auto& [peer, indices] : shared) {
    auto& buffer = buffers[peer];
    buffer.send_idx.assign(indices.send_idx.begin(), indices.send_idx.end());
    buffer.recv_idx.assign(indices.recv_idx.begin(), indices.recv_idx.end());
    buffer.send = V.create_multivector(static_cast<Index>(buffer.send_idx.size()), V.cols());
    buffer.recv = V.create_multivector(static_cast<Index>(buffer.recv_idx.size()), infos.at(peer).m);
    V.pack_rows(buffer.send_idx, *buffer.send);
  }

  {
    const auto mpi_type = Dune::MPITraits<T>::getType();
    std::vector<MultiVecHostView<const T>> send_views;
    std::vector<MultiVecHostView<T>> recv_views;
    for (auto& [peer, buffer] : buffers) {
      auto& recv = recv_views.emplace_back(buffer.recv->host_view(write));
      MPI_Irecv(recv.data(), static_cast<int>(recv.size()), mpi_type, peer, neighbour_data_tag, mpi_comm, &reqs.emplace_back());
      const auto& send = send_views.emplace_back(std::as_const(*buffer.send).host_view(read));
      MPI_Isend(send.data(), static_cast<int>(send.size()), mpi_type, peer, neighbour_data_tag, mpi_comm, &reqs.emplace_back());
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  } // the views are released here, so the received data is part of the buffers

  // Each neighbour's rows go to the rows we share with it, in the columns of its block
  for (const auto& [peer, buffer] : buffers) result.columns->unpack_rows(*buffer.recv, buffer.recv_idx, result.blocks.at(peer).first);
  return result;
}

/** Returns our block row of the coarse matrix A_0 = R_0 A R_0^T, where R_0^T = [R_1^T V_1, ..., R_N^T V_N] contains the
 *  basis vectors of all subdomains, extended by zero. A_ovlp is the restriction A_i = R_i A R_i^T of the global matrix to
 *  the overlapping index set of the coarse space. Collective.
 *
 *  The block of our row that belongs to subdomain j is
 *
 *    (A_0)_ij = V_i^H R_i A R_j^T V_j = V_i^H A_i R_ij V_j,  with R_ij = R_i R_j^T,
 *
 *  i.e. it only needs our overlapping matrix and the basis of subdomain j restricted to our subdomain. The second
 *  equality holds because the basis V_i vanishes on the outermost layer of the overlap: only the rows of A_i on that
 *  layer miss columns of A (the ones outside our subdomain), and V_i^H removes exactly these rows. The result is only
 *  correct for such a basis.
 *
 *  The coarse basis vectors are numbered globally rank by rank, so ours are first_row, ..., first_row + m_i - 1 (m_i =
 *  number of our basis vectors). The result holds the rows of A_0 with these numbers, restricted to the columns of the
 *  subdomains that intersect ours (all other entries of these rows are zero): values is a dense m_i x M multivector,
 *  where M is the number of basis vectors of these subdomains (including ours), and cols[c] is the global number of
 *  column c of values. The columns are sorted by rank, and within a rank by basis vector, so cols is increasing.
 */
template <class T>
CoarseRows<T> coarse_matrix_rows(const LocalMat<T>& A_ovlp, const CoarseSpace<T>& cs)
{
  const auto& V = *cs.basis;
  const auto& ovlp = *cs.overlap;
  DDM_CHECK(A_ovlp.rows() == V.rows() && A_ovlp.cols() == V.rows(), "coarse_matrix_rows: matrix is {}x{}, but the basis has {} rows", A_ovlp.rows(), A_ovlp.cols(), V.rows());

  // The bases of all subdomains that intersect ours, restricted to our subdomain: one block R_ij V_j per subdomain j,
  // side by side. Our own basis V_i is one of the blocks, so the products below compute the whole block row at once.
  const auto neighbours = neighbour_basis(*ovlp.comm, V);

  // A_i_basis = A_i [R_ij V_j ...], with a single pass over the matrix for all blocks
  auto A_i_basis = A_ovlp.create_range_multivector(neighbours.columns->cols());
  A_ovlp.spmm(*neighbours.columns, *A_i_basis);

  // values = V_i^H A_i [R_ij V_j ...]: one row per basis vector of ours, one column per basis vector of a subdomain
  // that intersects ours
  CoarseRows<T> rows;
  rows.first_row = neighbours.blocks.at(ovlp.comm->rank()).global_offset;
  rows.values = V.create_multivector(V.cols(), A_i_basis->cols());
  V.dot(*A_i_basis, *rows.values);

  // Column `column` of the result is basis vector (column - block.first) of the block's subdomain, whose global
  // number starts at block.global_offset
  rows.cols.reserve(A_i_basis->cols());
  for (const auto& [block_rank, block] : neighbours.blocks)
    for (Index column = block.first; column < block.end; ++column) rows.cols.push_back(block.global_offset + (column - block.first));
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
 *  - mat:    config of the coarse matrix (see create_local_mat())
 *  - solver: config of the solver for the coarse problem (see create_solver())
 *  - prec:   config of the preconditioner of that solver (see create_prec())
 */
template <class T>
class CoarseCorrection final : public Prec<T> {
public:
  // A_ovlp is the restriction of A to the overlapping index set of the coarse space. Collective
  CoarseCorrection(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, std::shared_ptr<const LocalMat<T>> A_ovlp, std::shared_ptr<const CoarseSpace<T>> cs)
      : Prec<T>(std::move(A))
      , cs_(std::move(cs))
  {
    const auto& V = *cs_->basis;
    DDM_CHECK(cs_->overlap->n_original == this->mat()->rows(), "coarse correction: the coarse space was built for {} indices, but the matrix has {} rows", cs_->overlap->n_original,
              this->mat()->rows());

    d_.emplace(V.create_vector(V.rows()));
    x_.emplace(V.create_vector(V.rows()));
    c_.emplace(V.create_vector(V.cols()));

    setup_coarse_problem(config, detail::coarse_matrix_rows(*A_ovlp, *cs_));
  }

  Dune::SolverCategory::Category category() const override { return this->mat()->category(); }

private:
  // Collective
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    const auto& V = *cs_->basis;
    const auto& comm = *cs_->overlap->comm;
    const auto n = cs_->overlap->n_original;

    // r is consistent, so the owners of the overlap indices have the right values
    d_->copy_n_from(r, n);
    comm.broadcast(*d_);

    // Restriction: our part of R_0 r is V_i^H d
    V.dot(*d_, *c_);

    // Solve the coarse problem on rank 0 and hand everyone its part of the solution
    gather_coarse_defect();
    if (rank_ == 0) {
      *x0_ = T{0};
      solver_->solve(*d0_, *x0_);
    }
    scatter_coarse_solution();

    // Prolongation: R_0^T x_0 is the sum of V_i x_i over all subdomains, which the reduction makes consistent
    V.mv(*c_, *x_);
    comm.reduce(*x_);
    z.copy_n_from(*x_, n);
  }

  // The coarse space would have to be rebuilt too, so the owner of the coarse space creates a new coarse correction
  void do_update() override { TODO("CoarseCorrection::do_update"); }

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

    // How many rows (= basis vectors) and columns every rank has. The rows are numbered rank by rank, so the row
    // counts also give the position of every rank's part of the coarse vectors.
    const Index sizes[2] = {rows.values->rows(), rows.values->cols()};
    std::vector<Index> all_sizes(rank_ == 0 ? 2 * size : 0);
    MPI_Gather(sizes, 2, Dune::MPITraits<Index>::getType(), all_sizes.data(), 2, Dune::MPITraits<Index>::getType(), 0, mpi_comm);

    std::vector<int> col_counts(size, 0);
    std::vector<int> col_displs(size, 0);
    std::vector<int> value_counts(size, 0);
    std::vector<int> value_displs(size, 0);
    if (rank_ == 0) {
      row_counts_.assign(size, 0);
      row_displs_.assign(size, 0);
      for (int p = 0; p < size; ++p) {
        row_counts_[p] = static_cast<int>(all_sizes[2 * p]);
        col_counts[p] = static_cast<int>(all_sizes[2 * p + 1]);
        value_counts[p] = row_counts_[p] * col_counts[p];
        if (p > 0) {
          row_displs_[p] = row_displs_[p - 1] + row_counts_[p - 1];
          col_displs[p] = col_displs[p - 1] + col_counts[p - 1];
          value_displs[p] = value_displs[p - 1] + value_counts[p - 1];
        }
      }
    }

    // The column numbers and the values (column by column) of every rank's rows
    std::vector<Index> all_cols(rank_ == 0 ? col_displs.back() + col_counts.back() : 0);
    MPI_Gatherv(rows.cols.data(), static_cast<int>(rows.cols.size()), Dune::MPITraits<Index>::getType(), all_cols.data(), col_counts.data(), col_displs.data(), Dune::MPITraits<Index>::getType(), 0,
                mpi_comm);

    std::vector<T> all_values(rank_ == 0 ? value_displs.back() + value_counts.back() : 0);
    // TODO(20261007-103129): Avoid the host view
    {
      const auto values = rows.values->host_view(read);
      MPI_Gatherv(values.data(), static_cast<int>(values.size()), Dune::MPITraits<T>::getType(), all_values.data(), value_counts.data(), value_displs.data(), Dune::MPITraits<T>::getType(), 0,
                  mpi_comm);
    }

    if (rank_ != 0) return;

    // Row first_row + a of rank p has the entries all_values[a + c * rows] in the columns all_cols[c]
    const Index n0 = row_displs_.back() + row_counts_.back();
    Pattern pattern(n0, n0);
    for (int p = 0; p < size; ++p)
      for (Index a = 0; a < row_counts_[p]; ++a)
        for (Index c = 0; c < col_counts[p]; ++c) pattern.add(row_displs_[p] + a, all_cols[col_displs[p] + c]);
    pattern.finalize();

    // Every rank's values are one dense block, but stored column by column, while add_values() expects it row by row.
    // So transpose each block once and add it with a single call.
    auto A0_local = create_local_mat<T>(config.sub("mat"), pattern);
    std::vector<Index> block_rows;
    std::vector<T> block_values;
    for (int p = 0; p < size; ++p) {
      const Index m = row_counts_[p];
      const Index cols = col_counts[p];
      block_rows.resize(m);
      std::iota(block_rows.begin(), block_rows.end(), row_displs_[p]);
      block_values.resize(m * cols);
      for (Index a = 0; a < m; ++a)
        for (Index c = 0; c < cols; ++c) block_values[a * cols + c] = all_values[value_displs[p] + a + c * m];
      A0_local->add_values(block_rows, std::span<const Index>(all_cols.data() + col_displs[p], cols), block_values);
    }
    A0_local->assemble();

    auto A0 = std::make_shared<const Mat<T>>(A0_local, nullptr);
    solver_ = create_solver<T>(config.sub("solver"), A0, create_prec<T>(config.sub("prec"), A0));
    d0_.emplace(A0->create_range_vector());
    x0_.emplace(A0->create_domain_vector());
  }

  // Collects every rank's part of R_0 r (in c_) in d0_ on rank 0. Collective
  // TODO(20261007-103129): Avoid the host views, this is done in every apply()
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
  // TODO(20261007-103129): Avoid the host views, this is done in every apply()
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
};
} // namespace ddm
