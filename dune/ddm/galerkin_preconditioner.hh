#pragma once

#include "backend/backend.hh"
#include "communication.hh"
#include "factory.hh"
#include "helpers.hh"
#include "logger.hh"
#include "multivector.hh"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>
#include <map>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ddm {

/** @brief Computes R A R^T
 *
 *  The matrix A and the multivector R must live on the same backend.
 *  The resulting matrix is assembled on the host but only on rank
 *  zero. On all other ranks the returned matrix has size 0x0.
 *
 *  Only the columns of R that belong to this rank or to one of its neighbours take part: a rank
 *  that shares no index with us has template vectors that are zero on all of our indices, so its
 *  columns contribute nothing to our rows of the coarse matrix. Each rank assembles the rows of
 *  the coarse matrix belonging to its own template vectors, which is exact without any summation
 *  across ranks because A is the overlapping matrix, i.e. it agrees with the global matrix
 *  wherever a local template vector is supported.
 */
template <class Scalar, class Matrix, class MultiVectorIndex>
Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>> galerkin_product(const Matrix& A, const MultiVector<Scalar, backend::backend_of_t<Matrix>, MultiVectorIndex>& R, Communication& comm)
{
  using Backend = backend::backend_of_t<Matrix>;
  using MultiVector = MultiVector<Scalar, Backend, MultiVectorIndex>;
  using CoarseMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;

  auto ctx = Backend::context(A);
  const MultiVectorIndex n = A.N();
  const int num_t = static_cast<int>(R.cols());
  if (R.rows() != n) DUNE_THROW(Dune::Exception, "The multivector has " << R.rows() << " rows, but the matrix has " << n);

  // 1. Find out how many template vectors each rank provides and where they sit in the coarse
  //    numbering (rank order, so rank k's block starts at the sum of the previous counts).
  const auto& pattern = comm.communication_pattern();
  MPI_Comm mpicomm = pattern.communicator();
  const auto c = comm.communicator();
  const int size = c.size();

  std::vector<int> num_t_per_rank(size);
  MPI_Allgather(&num_t, 1, MPI_INT, num_t_per_rank.data(), 1, MPI_INT, mpicomm);
  const int total_num_t = std::accumulate(num_t_per_rank.begin(), num_t_per_rank.end(), 0);
  if (total_num_t == 0) DUNE_THROW(Dune::Exception, "No rank provided a template vector, the coarse space would be empty");

  std::vector<int> offset_per_rank(size);
  std::exclusive_scan(num_t_per_rank.begin(), num_t_per_rank.end(), offset_per_rank.begin(), 0);

  // 2. Get the zero-extended template vectors of our neighbours. Their columns are concatenated
  //    into one multivector, one block per neighbour in the (sorted) order of
  //    pattern.neighbours(); every column outside a neighbour's block is zero.
  const auto& neighbours = pattern.neighbours();
  std::map<int, MultiVectorIndex> block_offsets;
  MultiVectorIndex recv_cols = 0;
  for (const auto p : neighbours) {
    block_offsets.emplace(p, recv_cols);
    recv_cols += static_cast<MultiVectorIndex>(num_t_per_rank[p]);
  }

  MultiVector received(ctx, n, recv_cols);
  received.zero();

  comm.exchange_all_holders(R, received, block_offsets);

  // 3. y_g = A R_g for every column this rank can see. One batched SpMM per multivector: a single
  //    pass over A's sparsity for all of its columns.
  MultiVector y_own(ctx, n, num_t);
  Backend::spmm(A, R, y_own);

  MultiVector y_recv(ctx, n, recv_cols);
  if (recv_cols > 0) Backend::spmm(A, received, y_recv);

  // 4. This rank's rows of the coarse matrix, in column major order: entry (k, g) is
  //    R_k . y_g for g running over all template vectors.
  std::vector<Scalar> my_rows_flat(static_cast<std::size_t>(num_t) * total_num_t, 0.);
  const auto flat = [&](int g, int k) { return static_cast<std::size_t>(g) * num_t + k; };

  std::vector<Scalar> out_own(std::max(num_t, 1));
  std::vector<Scalar> out_recv(std::max<std::size_t>(recv_cols, 1));

  // Diagonal block: products against this rank's own columns
  const int own_offset = offset_per_rank[c.rank()];
  for (int k = 0; k < num_t; ++k) {
    Backend::batched_dot(ctx, y_own, R.col(k), out_own.data());
    for (int j = 0; j < num_t; ++j) my_rows_flat[flat(own_offset + j, k)] = out_own[j];
  }

  // Off-diagonal blocks: products against the columns that just arrived. A neighbour with no
  // template vectors contributes an all-zero block, which cannot add anything.
  for (int k = 0; k < num_t; ++k) {
    if (recv_cols == 0) break;
    Backend::batched_dot(ctx, y_recv, R.col(k), out_recv.data());
    for (const auto p : neighbours) {
      const int b = static_cast<int>(block_offsets.at(p));
      for (int j = 0; j < num_t_per_rank[p]; ++j) my_rows_flat[flat(offset_per_rank[p] + j, k)] = out_recv[b + j];
    }
  }

  // 5. Assemble the coarse matrix on rank 0 (empty everywhere else)
  return gatherMatrixFromRowsFlat(my_rows_flat, total_num_t, mpicomm);
}

/** @brief How the coarse problem A_c x_c = R d is solved.
 *
 *  Both modes compute the same thing, up to round-off, and differ only in where the work happens.
 *  Selected with the key 'solve_mode' in the subtree that configures the coarse solver.
 */
enum class CoarseSolveMode : std::uint8_t {
  /** @brief Assemble and factorize the coarse matrix on rank 0 alone.
   *
   *  Every apply() gathers the coarse defect there, solves, and scatters the solution back: two
   *  collectives, with the other ranks idle in between. Only rank 0 holds the coarse matrix and its
   *  factorization, which is what makes this the mode for a coarse space too large to replicate.
   */
  RankZero,

  /** @brief Broadcast the coarse matrix and factorize it on every rank.
   *
   *  Every apply() then needs a single MPI_Allgatherv of the coarse defect, after which each rank
   *  solves the whole coarse problem itself and reads off its own part of the solution. That is one
   *  collective instead of two and no serial section, at the price of storing the coarse matrix and
   *  its factorization on every rank.
   */
  Redundant,
};

/** @brief A preconditioner that implements R^T (R A R^T)^-1 R.
 *
 *   The restriction matrix R has dimensions (n1 + n2 + ... + np) x n, where nk is the number
 *   of columns of R that rank k provides (this number can be different for each rank; ranks
 *   may not provide any vectors at all); n is the number of rows/ columns of the matrix A.
 */
template <class Vec, class Communication>
class GalerkinPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  using Scalar = typename Vec::field_type;
  /** @brief The coarse defect and solution. Always host-resident: the coarse problem is gathered,
   *  solved and scattered on the host regardless of where the fine-level vectors live. */
  using CoarseVector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;
  using Solver = Dune::InverseOperator<CoarseVector, CoarseVector>;
  using Backend = backend::backend_of_t<Vec>;
  using Restriction = MultiVector<Scalar, Backend>;

public:
  /** @brief Type of the assembled coarse matrix. */
  using CoarseMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;

  /**
   * @brief Constructor for the Galerkin preconditioner.
   *
   * Sets up a Galerkin-type preconditioner that computes R^T (R A R^T)^-1 R,
   * where R is the restriction matrix built from template vectors.
   *
   * Collective on the communicator of @p comm.
   *
   * @param A     The overlapping matrix (must match the overlapping index set of @p comm). Its
   *              rows must agree with the global matrix wherever the template vectors are
   *              supported; entries at the template vectors' zero positions do not matter.
   * @param R     This rank's template vectors as the columns of a multivector. May be empty: a
   *              rank that contributes nothing to the coarse space still takes part in the setup
   *              and in every apply(). The multivector is moved into the preconditioner.
   *              The caller is responsible for providing "sensible" vectors (e.g. with values at
   *              Dirichlet boundary nodes zeroed out).
   * @param comm  Communication describing the overlapping index set
   * @param ptree Parameters. The subtree @p subtree_name configures the coarse solver and has to
   *              carry at least the key 'type'. The optional key 'solve_mode' picks where the
   *              coarse problem is solved, "rank_zero" (the default) or "redundant", see
   *              CoarseSolveMode.
   * @param subtree_name Name of that subtree, use the emptry string "" to use @p ptree itself.
   *                     This is optional; the default is "galerkin".
   *
   * @throws Dune::Exception if the size of R is not compatible with A
   */
  template <class Mat>
  GalerkinPreconditioner(const Mat& A, MultiVector<Scalar, Backend> R, std::shared_ptr<Communication> comm, const Dune::ParameterTree& ptree = {}, const std::string& subtree_name = "galerkin")
      : comm(std::move(comm))
      , n(A.N())
      , d_ovlp(Backend::context(A), A.N(), 1)
      , x_ovlp(Backend::context(A), A.N(), 1)
      , solver_ptree(ptree)
      , solver_subtree_name(subtree_name)
      , solve_mode(parse_solve_mode(ptree, subtree_name))
  {
    static_assert(std::is_same_v<backend::backend_of_t<Mat>, Backend>,
                  "The matrix passed to the constructor must live on the same backend as the vector type used to define the GalerkinPreconditioner");

    // TODO(20260920-094902): Ensure that R and A live on the same context
    register_log_events();
    build_solver(A, R);
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::overlapping; }

  void pre(Vec&, Vec&) override {}
  void post(Vec&) override {}

  /** @brief The assembled coarse matrix R A R^T, for inspection, debugging and testing.
   *
   *  In CoarseSolveMode::RankZero the matrix is only assembled on rank 0 and the returned matrix is
   *  empty (0x0) everywhere else; in CoarseSolveMode::Redundant every rank holds the same full
   *  matrix. Rows and columns are numbered by concatenating the template vectors of all ranks in
   *  rank order, i.e. the k-th template vector of rank r is coarse index sum(num_t of ranks < r) + k.
   */
  const CoarseMatrix& get_coarse_matrix() const { return *a0; }

  void apply(Vec& x, const Vec& d) override
  {
    Logger::ScopedLog se(apply_event);

    // d_ovlp and x_ovlp cover the overlapping subdomain, so anything longer would run past their
    // end below. Checked before the first collective, so that a violation cannot hang the others.
    if (d.size() > n or x.size() > n) DUNE_THROW(Dune::Exception, "apply() got vectors of size " << x.size() << "/" << d.size() << ", larger than the overlapping subdomain (" << n << ")");

    MPI_Comm mpicomm = comm->communication_pattern().communicator();
    const int rank = comm->communicator().rank();
    const auto mpi_type = Dune::MPITraits<Scalar>::getType();

    // 1. Copy the local defect into the overlapping multivector
    copy_into_column(d, d_ovlp, 0u);

    // 1.5 Fetch the entries of the overlap extension, which this rank has no other way of knowing.
    // When the two index sets have the same size there is no extension to fill, and the defect is
    // already consistent by the invariant in ddm.hh, so the communication would be a no-op.
    if (d.size() < n) comm->broadcast(d_ovlp);

    // 2. Compute the local contribution to the coarse defect in one batched dot product
    if (num_t > 0) Backend::batched_dot(d_ovlp.context(), *restr, d_ovlp.col(0), d_local.data());

    // 3.-5. Assemble the coarse defect, solve the coarse problem, and take this rank's share of the
    // coarse solution back out. See CoarseSolveMode for the difference between the two branches.
    if (solve_mode == CoarseSolveMode::Redundant) {
      MPI_Allgatherv(d_local.data(), num_t, mpi_type, d0.data(), num_t_per_rank.data(), offset_per_rank.data(), mpi_type, mpicomm);

      // TODO(20260920-095526): Don't allocate memory in every apply
      CoarseVector d0v(total_num_t);
      CoarseVector x0v(total_num_t);
      for (int k = 0; k < total_num_t; ++k) d0v[k] = d0[k];

      Dune::InverseOperatorResult res;
      x0v = 0;
      solver->apply(x0v, d0v, res);

      // Every rank solved the whole coarse problem, so its own part is just a slice of x0v
      for (int k = 0; k < num_t; ++k) coarse_solution[k] = x0v[offset_per_rank[rank] + k][0];
    }
    else {
      MPI_Gatherv(d_local.data(), num_t, mpi_type, d0.data(), num_t_per_rank.data(), offset_per_rank.data(), mpi_type, 0, mpicomm);

      if (rank == 0) {
        CoarseVector d0v(total_num_t);
        CoarseVector x0v(total_num_t);
        for (int k = 0; k < total_num_t; ++k) d0v[k] = d0[k];

        Dune::InverseOperatorResult res;
        x0v = 0;
        solver->apply(x0v, d0v, res);

        for (int k = 0; k < total_num_t; ++k) x0[k] = x0v[k][0];
      }

      MPI_Scatterv(x0.data(), num_t_per_rank.data(), offset_per_rank.data(), mpi_type, coarse_solution.data(), num_t, mpi_type, 0, mpicomm);
    }

    // 6. Prolongate this rank's own part of the coarse correction; the reduction below sums it with
    //    the other ranks' parts. The prolongation weights must be readable by the gemv_t kernel, so
    //    they are uploaded to the backend first.
    x_ovlp.zero();
    if (num_t > 0) {
      const auto coarse_solution_dev = Backend::make_buffer_from_host(d_ovlp.context(), coarse_solution);
      Backend::gemv_t(d_ovlp.context(), *restr, coarse_solution_dev.data(), x_ovlp.col(0));
    }

    comm->reduce(x_ovlp, ReductionOperation::Addition);

    // 7. Restrict the solution to the non-overlapping subdomain
    copy_from_column(x_ovlp, 0U, x);
  }

private:
  /** @brief Register logging events for performance monitoring */
  void register_log_events()
  {
    apply_event = Logger::get().registerOrGetEvent("GalerkinPrec", "apply");
    build_solver_event = Logger::get().registerOrGetEvent("GalerkinPrec", "build Matrix");
  }

  /**
   * @brief Build the coarse space solver by assembling R A R^T and factorizing it
   *
   * @param A The overlapping matrix used to compute the Galerkin product
   * @param R The part of R that we own
   *
   */
  // TODO(20260920-095429): build_solver and galerkin_product have some duplicate code
  template <class Mat>
  void build_solver(const Mat& A, MultiVector<Scalar, Backend>& R)
  {
    Logger::ScopedLog se(build_solver_event);
    const MPI_Comm mpicomm = comm->communication_pattern().communicator();
    const int rank = comm->communicator().rank();
    const int size = comm->communicator().size();

    num_t = static_cast<int>(R.cols());
    if (R.rows() != n) DUNE_THROW(Dune::Exception, "The multivector has " << R.rows() << " rows, but the matrix has " << n);

    // Find out how many template vectors each rank has and how large coarse matrix will be
    num_t_per_rank.resize(size);
    MPI_Allgather(&num_t, 1, MPI_INT, num_t_per_rank.data(), 1, MPI_INT, mpicomm);
    total_num_t = std::accumulate(num_t_per_rank.begin(), num_t_per_rank.end(), 0);
    if (total_num_t == 0) DUNE_THROW(Dune::Exception, "No rank provided a template vector, the coarse space would be empty");

    offset_per_rank.resize(size);
    std::exclusive_scan(num_t_per_rank.begin(), num_t_per_rank.end(), offset_per_rank.begin(), 0);

    const int max_num_t = *std::max_element(num_t_per_rank.begin(), num_t_per_rank.end());
    logger::debug("Setting up GalerkinPreconditioner with {} template vector{} ({} in total, at most {} on a single rank)", num_t, (num_t == 1 ? "" : "s"), total_num_t, max_num_t);

    // Buffers that apply() reuses across Krylov iterations. The coarse vectors are only needed
    // where the coarse problem is actually solved; in RankZero mode MPI_Gatherv and MPI_Scatterv
    // ignore them everywhere else. The local buffers keep one spare entry so that a rank without
    // template vectors still hands MPI a valid pointer.
    const bool solve_here = solve_mode == CoarseSolveMode::Redundant or rank == 0;
    d_local.assign(std::max(num_t, 1), 0.);
    coarse_solution.assign(std::max(num_t, 1), 0.);
    d0.resize(solve_here ? total_num_t : 0);
    x0.resize(solve_here ? total_num_t : 0);

    a0 = std::make_shared<CoarseMatrix>(galerkin_product(A, R, *comm));
    if (solve_mode == CoarseSolveMode::Redundant) broadcastMatrix(*a0, mpicomm);

    // The restriction stays around for the prolongation in apply()
    restr = std::make_shared<Restriction>(std::move(R));

    if (solve_here) {
      using Op = Dune::MatrixAdapter<CoarseMatrix, CoarseVector, CoarseVector>;
      Dune::initSolverFactories<Op>();
      auto op = std::make_shared<Op>(a0);

      const auto& subtree = solver_subtree_name.size() == 0 ? solver_ptree : solver_ptree.sub(solver_subtree_name);
      solver = ddm::getDirectSolverFromFactory(op, subtree);
    }
  }

  /** @brief Reads the coarse solve mode from the parameter tree, defaulting to CoarseSolveMode::RankZero. */
  static CoarseSolveMode parse_solve_mode(const Dune::ParameterTree& ptree, const std::string& subtree_name)
  {
    const auto& subtree = subtree_name.size() == 0 ? ptree : ptree.sub(subtree_name);
    const auto& mode_string = subtree.get("solve_mode", "rank_zero");

    if (mode_string == "rank_zero") return CoarseSolveMode::RankZero;
    else if (mode_string == "redundant") return CoarseSolveMode::Redundant;
    else DUNE_THROW(Dune::Exception, "Unknown coarse solve mode: " + mode_string + ", expected 'rank_zero' or 'redundant'");
  }

  std::shared_ptr<Communication> comm;
  std::shared_ptr<CoarseMatrix> a0;                      ///  The coarse matrix R A R^T; only assembled on rank 0, empty elsewhere (Redundant: broadcast to all ranks)
  std::shared_ptr<Restriction> restr;                    ///  This rank's columns of the restriction, used by the prolongation in apply()
  std::shared_ptr<Solver> solver;                        ///  Direct solver for the coarse problem (UMFPack by default)
  std::size_t n;                                         ///  Size of the overlapping index set
  Restriction d_ovlp;                                    ///  Overlapping defect, one column, for temporary storage
  Restriction x_ovlp;                                    ///  Overlapping solution, one column, for temporary storage
  std::vector<Scalar> d0;                                ///  Coarse defect, gathered from all ranks; only sized on rank 0 (Redundant: everywhere)
  std::vector<Scalar> x0;                                ///  Coarse solution, scattered to all ranks; only sized on rank 0
  std::vector<Scalar> d_local;                           ///  This rank's contribution to the coarse defect
  std::vector<Scalar> coarse_solution;                   ///  This rank's share of the coarse solution
  int num_t{};                                           ///  Number of template vectors owned by this rank
  int total_num_t{};                                     ///  Total number of template vectors across all ranks
  std::vector<int> num_t_per_rank;                       ///  Number of template vectors per rank
  std::vector<int> offset_per_rank;                      ///  Offset for each rank's template vectors in global numbering
  Dune::ParameterTree solver_ptree;                      ///  Parameters for the coarse solver
  std::string solver_subtree_name;                       ///  Subtree of solver_ptree holding the coarse solver settings
  CoarseSolveMode solve_mode{CoarseSolveMode::RankZero}; ///  Where the coarse problem is solved
  Logger::Event* apply_event{};                          ///  Logging event for timing the apply method
  Logger::Event* build_solver_event{};                   ///  Logging event for timing the solver building process
};

} // namespace ddm
