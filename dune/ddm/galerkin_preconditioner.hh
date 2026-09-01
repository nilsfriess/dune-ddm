#pragma once

#include "helpers.hh"
#include "logger.hh"

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

/** @brief A Galerkin-type preconditioner that implements R^T (R A R^T)^-1 R.

    This preconditioner constructs a coarse space correction using multiple template vectors
    provided by each MPI rank. The restriction matrix R has dimensions 'total_template_vectors'
    x 'matrix_size', where each column corresponds to one template vector from one rank.

    <b>Algorithm Overview:</b>
    1. Each MPI rank provides zero or more template vectors defined on the overlapping subdomain
    2. The restriction matrix R^T is formed by collecting all template vectors from all ranks
    3. The coarse matrix A_c = R A R^T is assembled by computing all pairwise products
       between template vectors: (R_i, A R_j) for all i,j
    4. The coarse system A_c x_c = R d is solved, either on rank 0 alone or redundantly on every
       rank, see CoarseSolveMode
    5. The correction is prolongated: x += R^T x_c

    <b>Requirements:</b>
    - Template vectors must be defined on the overlapping subdomain matching the index set of the
      communication object passed to the constructor
    - The matrix A must be extended to the same overlapping subdomain
    - Template vectors should be suitable for the problem (e.g., zero on Dirichlet boundaries)
    - A rank may provide no template vector at all, as long as some rank does. All ranks take part
      in the setup and in every apply() either way, so the decision does not have to be collective.

    <b>Typical Use Case:</b>
    This preconditioner is commonly used as the coarse space component in two-level domain
    decomposition methods, combined with a local preconditioner via `CombinedPreconditioner`.

    <b>Scalability:</b>
    The coarse problem size equals the total number of template vectors across all ranks.
    For good scalability, this should grow slowly with the number of MPI processes.
 */
template <class Vec, class Communication>
class GalerkinPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  using Solver = Dune::InverseOperator<Vec, Vec>;

public:
  /** @brief Type of the assembled coarse matrix. This is what gatherMatrixFromRowsFlat() returns. */
  using CoarseMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;

private:
  // apply() hands the raw storage of Vec to MPI as a plain array of doubles, which is only the same
  // thing if a block holds exactly one double.
  static_assert(std::is_same_v<typename Vec::field_type, double>, "GalerkinPreconditioner communicates vector entries as MPI_DOUBLE");
  static_assert(sizeof(typename Vec::block_type) == sizeof(typename Vec::field_type), "GalerkinPreconditioner requires a vector with one scalar per block");

  /** @brief Holds the template vector this rank sends and those it receives, during setup.
   *
   *  The received vectors are zero-extended to the local index set: the communication only writes
   *  the shared indices, and clear() puts the zeros everywhere else.
   */
  struct VecDistributor {
    using value_type = std::pair<int, double>;

    /**
     * @param temp       Vector of the local size, used to size the per-neighbour vectors
     * @param neighbours Ranks we share indices with. Must cover every rank that sends to us
     *                   through the communication interface, see build_solver().
     * @param rank       This rank, sent along with every value so that the receiver can tell the
     *                   senders apart
     * @param size       Number of ranks, i.e. the size of the rank -> vector lookup table
     */
    VecDistributor(const Vec& temp, const std::vector<int>& neighbours, int rank, int size)
        : by_rank(size, nullptr)
        , rank{rank}

    {
      for (const auto& nb : neighbours) {
        auto& vec = others.emplace(nb, temp).first->second;
        vec = 0;
        by_rank[nb] = &vec; // map nodes are stable, so this stays valid
      }
    }

    /** @brief Set all neighbour vectors to zero */
    void clear()
    {
      for (auto& [nb, vec] : others) vec = 0;
    }

    VecDistributor(const VecDistributor&) = delete;
    VecDistributor(VecDistributor&&) = delete;
    VecDistributor& operator=(const VecDistributor&) = delete;
    VecDistributor& operator=(VecDistributor&&) = delete;
    ~VecDistributor() = default;

    Vec* own = nullptr;        /**< The vector this rank sends */
    std::map<int, Vec> others; /**< Vectors received from the neighbours, keyed by rank */
    std::vector<Vec*> by_rank; /**< Points into `others`, null for a rank that is no neighbour */
    int rank;
  };

  /** @brief Gather-scatter helper that tags every value with the rank it came from */
  struct CopyGatherScatterWithRank {
    using DataType = std::pair<int, double>;

    static DataType gather(const VecDistributor& vd, std::size_t i) { return std::make_pair(vd.rank, (*vd.own)[i]); }

    /// Called once per received index, hence the lookup table instead of a search in the map.
    static void scatter(VecDistributor& vd, const DataType& data, std::size_t i)
    {
      const auto& [rank, v] = data;

      // Can only fire if the neighbour list disagrees with the interface the communicator was built
      // from, which would be a bug in build_solver(). Abort instead of throwing: this runs inside a
      // collective, so an exception on one rank would leave the others waiting forever.
      if (rank < 0 or rank >= static_cast<int>(vd.by_rank.size()) or vd.by_rank[rank] == nullptr) {
        logger::error_all("CopyGatherScatterWithRank: received data from rank {}, which is not in the neighbour list", rank);
        MPI_Abort(MPI_COMM_WORLD, 17);
      }
      (*vd.by_rank[rank])[i] = v;
    }
  };

public:
  /**
   * @brief Constructor for the Galerkin preconditioner.
   *
   * Sets up a Galerkin-type preconditioner that computes R^T (R A R^T)^-1 R,
   * where R is the restriction matrix built from template vectors.
   *
   * Collective on the communicator of @p comm.
   *
   * @param A     The overlapping matrix (must match the overlapping index set of @p comm)
   * @param ts    This rank's template vectors, defined on the overlapping subdomain. Each must
   *              have A.N() entries. May be empty: a rank that contributes nothing to the coarse
   *              space still takes part in the setup and in every apply().
   * @param comm  Communication describing the overlapping index set
   * @param ptree Parameters. The subtree @p subtree_name configures the coarse solver and has to
   *              carry at least the key 'type'. The optional key 'solve_mode' picks where the
   *              coarse problem is solved, "rank_zero" (the default) or "redundant", see
   *              CoarseSolveMode.
   * @param subtree_name Name of that subtree, "" to use @p ptree itself
   *
   * @throws Dune::Exception if a template vector does not match the size of the matrix, if no rank
   *         provides a template vector at all, or if the coarse solver is not configured
   */
  template <class Mat>
  GalerkinPreconditioner(const Mat& A, const std::vector<Vec>& ts, std::shared_ptr<Communication> comm, const Dune::ParameterTree& ptree, const std::string& subtree_name = "galerkin")
      : comm(std::move(comm))
      , n(A.N())
      , d_ovlp(n)
      , x_ovlp(n)
      , solver_ptree(ptree)
      , solver_subtree_name(subtree_name)
      , solve_mode(parse_solve_mode(ptree, subtree_name))
  {
    register_log_events();
    update(A, ts);
  }

  /** @brief Reassembles and refactorizes the coarse problem for a new matrix and coarse space.
   *
   *  Collective. Lets a caller that keeps the preconditioner across several solves follow a
   *  changing matrix without rebuilding the object. @p A has to describe the same overlapping
   *  subdomain as the one passed to the constructor, but the number of template vectors may change,
   *  here and on the other ranks.
   */
  template <class Mat>
  void update(const Mat& A, const std::vector<Vec>& ts)
  {
    if (A.N() != n) DUNE_THROW(Dune::Exception, "Matrix size changed from " << n << " to " << A.N() << ", which needs a new GalerkinPreconditioner");

    for (const auto& t : ts)
      if (t.N() != n) DUNE_THROW(Dune::Exception, "Template vectors must match size of matrix");

    num_t = static_cast<int>(ts.size());
    restr_vecs.assign(ts.begin(), ts.end());

    // Zero out template vectors at Dirichlet DOFs (identity rows in the matrix).
    // Without this, the coarse correction injects nonzero values at Dirichlet DOFs,
    // which the outer Krylov solver then has to undo, increasing iteration counts.
    const auto dirichlet = detect_dirichlet_dofs(A);
    for (auto& v : restr_vecs)
      for (std::size_t j = 0; j < n; ++j)
        if (dirichlet[j]) v[j] = 0;

    build_solver(A);
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::overlapping; }

  void pre(Vec&, Vec&) override {}
  void post(Vec&) override {}

  void apply(Vec& x, const Vec& d) override
  {
    Logger::ScopedLog se(apply_event);

    // d_ovlp and x_ovlp cover the overlapping subdomain, so anything longer would run past their
    // end below. Checked before the first collective, so that a violation cannot hang the others.
    if (d.N() > n or x.N() > n) DUNE_THROW(Dune::Exception, "apply() got vectors of size " << x.N() << "/" << d.N() << ", larger than the overlapping subdomain (" << n << ")");

    auto mpicomm = comm->communicator();
    const int rank = comm->communicator().rank();

    // 1. Copy local values from the incoming defect to the overlapping one
    for (std::size_t i = 0; i < d.N(); ++i) d_ovlp[i] = d[i];

    // 1.5 Fetch the entries of the overlap extension, which this rank has no other way of knowing.
    // When the two index sets have the same size there is no extension to fill, and the defect is
    // already consistent by the invariant in ddm.hh, so the communication would be a no-op.
    if (d.N() < n) comm->copyOwnerToAll(d_ovlp, d_ovlp);

    // 2. Compute local contribution of coarse defect
    for (int k = 0; k < num_t; ++k) {
      double dk = 0.;
      for (std::size_t i = 0; i < n; ++i) dk += restr_vecs[k][i] * d_ovlp[i];
      d_local[k] = dk;
    }

    // 3.-5. Assemble the coarse defect, solve the coarse problem, and take this rank's share of the
    // coarse solution back out. See CoarseSolveMode for the difference between the two branches.
    if (solve_mode == CoarseSolveMode::Redundant) {
      MPI_Allgatherv(d_local.data(), num_t, MPI_DOUBLE, d0.data(), num_t_per_rank.data(), offset_per_rank.data(), MPI_DOUBLE, mpicomm);

      x0 = 0;
      Dune::InverseOperatorResult res;
      solver->apply(x0, d0, res);

      // Every rank solved the whole coarse problem, so its own part is just a slice of x0
      for (int k = 0; k < num_t; ++k) coarse_solution[k] = x0[offset_per_rank[rank] + k][0];
    }
    else {
      MPI_Gatherv(d_local.data(), num_t, MPI_DOUBLE, d0.data(), num_t_per_rank.data(), offset_per_rank.data(), MPI_DOUBLE, 0, mpicomm);

      if (rank == 0) {
        x0 = 0;
        Dune::InverseOperatorResult res;
        solver->apply(x0, d0, res);
      }

      MPI_Scatterv(x0.data(), num_t_per_rank.data(), offset_per_rank.data(), MPI_DOUBLE, coarse_solution.data(), num_t, MPI_DOUBLE, 0, mpicomm);
    }

    // 6. Prolongate
    x_ovlp = 0;
    for (int k = 0; k < num_t; ++k)
      for (std::size_t j = 0; j < n; ++j) x_ovlp[j] += coarse_solution[k] * restr_vecs[k][j];

    comm->addOwnerCopyToAll(x_ovlp, x_ovlp);

    // 7. Restrict the solution to the non-overlapping subdomain
    for (std::size_t i = 0; i < x.N(); ++i) x[i] = x_ovlp[i];
  }

  /** @brief The assembled coarse matrix R A R^T, for inspection, debugging and testing.
   *
   *  In CoarseSolveMode::RankZero the matrix is only assembled on rank 0 and the returned matrix is
   *  empty (0x0) everywhere else; in CoarseSolveMode::Redundant every rank holds the same full
   *  matrix. Rows and columns are numbered by concatenating
   *  the template vectors of all ranks in rank order, i.e. the k-th template vector of rank r is
   *  coarse index sum(num_t of ranks < r) + k.
   */
  const CoarseMatrix& get_coarse_matrix() const { return *a0; }

private:
  /** @brief Register logging events for performance monitoring */
  void register_log_events()
  {
    apply_event = Logger::get().registerOrGetEvent("GalerkinPrec", "apply");
    build_solver_event = Logger::get().registerOrGetEvent("GalerkinPrec", "build Matrix");
  }

  /**
   * @brief Build the coarse space solver by assembling R A R^T and factorizing it
   * @param A The overlapping matrix used to compute the Galerkin product
   *
   * This method performs the main computational work:
   * 1. Distributes template vectors across all ranks
   * 2. Computes all pairwise products R_i^T A R_j
   * 3. Assembles the global coarse matrix A0 = R A R^T
   * 4. Factorizes A0 on rank 0 for solving coarse problems
   *
   * Each rank assembles the rows of A0 belonging to its own template vectors. That is exact
   * without any summation across ranks, because A is the overlapping matrix, i.e. it agrees with
   * the global matrix wherever a local template vector is supported.
   */
  template <class Mat>
  void build_solver(const Mat& A)
  {
    Logger::ScopedLog se(build_solver_event);
    auto mpicomm = comm->communicator();
    const int rank = comm->communicator().rank();
    const int size = comm->communicator().size();

    auto* comm_event = Logger::get().registerOrGetEvent("GalerkinPrec", "exchange template vectors");
    auto* local_local_sp_event = Logger::get().registerOrGetEvent("GalerkinPrec", "dot (local<>local)");
    auto* local_remote_sp_event = Logger::get().registerOrGetEvent("GalerkinPrec", "dot (local<>remote)");
    auto* gather_A0 = Logger::get().registerOrGetEvent("GalerkinPrec", "gather A0");
    auto* factor_A0 = Logger::get().registerOrGetEvent("GalerkinPrec", "factor A0");
    auto* prepare_event = Logger::get().registerOrGetEvent("GalerkinPrec", "prepare");

    Logger::get().startEvent(prepare_event);

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

    // This rank's row block of the coarse matrix, in column major order: entry (k, g) is
    // restr_vecs[k]^T A R_g, where g runs over the template vectors of this rank and of its
    // neighbours. All remaining entries are structurally zero and stay that way.
    std::vector<double> my_rows_flat(static_cast<std::size_t>(num_t) * total_num_t, 0.);
    Vec y(n);

    Vec zerovec(n);
    zerovec = 0;

    // The neighbour list has to match the interface built below, because a rank that sends us data
    // through that interface must have a slot in the VecDistributor. RemoteIndices::getNeighbours()
    // does not do that: it only returns whatever was handed to setNeighbours() beforehand as a hint
    // for the rebuild, and stays empty when the remote indices were rebuilt via the default ring
    // exchange. The remote index map, on the other hand, holds exactly the processes we share
    // indices with, which is what Interface::build() iterates over.
    std::vector<int> neighbour_vec;
    for (const auto& [remote_rank, indices] : comm->remoteIndices()) {
      (void)indices;
      if (remote_rank != rank) neighbour_vec.push_back(remote_rank);
    }
    VecDistributor vd(zerovec, neighbour_vec, rank, size);

    using AllSet = Communication::AllSet;
    AllSet all_att;
    Dune::Interface all_all_interface;
    all_all_interface.build(comm->remoteIndices(), all_att, all_att);
    Dune::BufferedCommunicator bcomm;
    bcomm.build<VecDistributor>(all_all_interface);

    Logger::get().endEvent(prepare_event);

    // TODO: Here we previously used a custom implementation of BufferedCommunicator that supports forwardBegin/forwardEnd to overlap computation and communication. This is currently disabled again to
    // make it work with a standard DUNE installation.

    // One exchange per round: in round idx every rank sends its idx-th template vector, or a zero
    // vector once it has run out, and receives the idx-th vector of each neighbour. max_num_t is
    // the same everywhere, so all ranks run the same number of rounds and the exchange stays
    // collective.
    for (int idx = 0; idx < max_num_t; ++idx) {
      vd.own = idx < num_t ? &restr_vecs[idx] : &zerovec;

      Logger::get().startEvent(comm_event);
      vd.clear(); // anything the neighbours do not send stays zero, which zero-extends their vectors
      bcomm.forward<CopyGatherScatterWithRank>(vd);
      Logger::get().endEvent(comm_event);

      if (num_t == 0) continue; // This rank owns no row of the coarse matrix, nothing to compute

      // Scalar products against this rank's own idx-th template vector
      Logger::get().startEvent(local_local_sp_event);
      if (idx < num_t) {
        A.mv(restr_vecs[idx], y);
        for (int k = 0; k < num_t; ++k) my_rows_flat[flat_index(offset_per_rank[rank] + idx, k)] = restr_vecs[k] * y;
      }
      Logger::get().endEvent(local_local_sp_event);

      // ... and against the vectors that just arrived. A neighbour with fewer than idx template
      // vectors sent a zero vector, which cannot contribute anything.
      Logger::get().startEvent(local_remote_sp_event);
      for (const auto& nb : neighbour_vec) {
        if (idx >= num_t_per_rank[nb]) continue;

        A.mv(vd.others.at(nb), y);
        for (int k = 0; k < num_t; ++k) my_rows_flat[flat_index(offset_per_rank[nb] + idx, k)] = restr_vecs[k] * y;
      }
      Logger::get().endEvent(local_remote_sp_event);
    }

    Logger::get().startEvent(gather_A0);
    a0 = std::make_shared<CoarseMatrix>(gatherMatrixFromRowsFlat(my_rows_flat, total_num_t, mpicomm));
    if (solve_mode == CoarseSolveMode::Redundant) broadcastMatrix(*a0, mpicomm);
    Logger::get().endEvent(gather_A0);

    if (rank == 0) logger::debug("Size of coarse space matrix: {}x{}, nonzeros: {}", a0->N(), a0->M(), a0->nonzeroes());

    Logger::get().startEvent(factor_A0);
    if (solve_here) {
      using Op = Dune::MatrixAdapter<CoarseMatrix, Vec, Vec>;
      Dune::initSolverFactories<Op>();
      auto op = std::make_shared<Op>(a0);

      // Since the error message that Dune gives us when there is no 'type' key in the solver subtree
      // is useless, we check ourselves first and tell the user what they need to do.
      const auto& subtree = solver_subtree_name.size() == 0 ? solver_ptree : solver_ptree.sub(solver_subtree_name);
      if (not subtree.hasKey("type"))
        DUNE_THROW(Dune::Exception, "You must specify the solver in the subtree " << get_parameter_tree_prefix(solver_ptree) << solver_subtree_name << " using the key 'type'");
      solver = Dune::getSolverFromFactory(op, subtree);
    }
    Logger::get().endEvent(factor_A0);
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

  /** @brief Position of the coarse matrix entry (row @p k, column @p global_idx) in my_rows_flat.
   *
   *  gatherMatrixFromRowsFlat() expects the rows in column major order. Computed in std::size_t
   *  because num_t * total_num_t overflows an int for a large coarse space.
   */
  std::size_t flat_index(int global_idx, int k) const { return static_cast<std::size_t>(global_idx) * num_t + k; }

  std::shared_ptr<Communication> comm;
  std::shared_ptr<CoarseMatrix> a0;                      ///  The coarse matrix R A R^T; only assembled on rank 0, empty elsewhere
  std::shared_ptr<Solver> solver;                        ///  Direct solver for the coarse problem (UMFPack by default)
  std::vector<Vec> restr_vecs;                           ///  Template vectors used to build the restriction matrix
  std::size_t n;                                         ///  Size of the overlapping index set
  Vec d_ovlp;                                            ///  Overlapping defect vector for temporary storage
  Vec x_ovlp;                                            ///  Overlapping solution vector for temporary storage
  Vec d0;                                                ///  Coarse defect, gathered from all ranks; only sized on rank 0
  Vec x0;                                                ///  Coarse solution, scattered to all ranks; only sized on rank 0
  std::vector<double> d_local;                           ///  This rank's contribution to the coarse defect
  std::vector<double> coarse_solution;                   ///  This rank's share of the coarse solution
  int num_t{};                                           ///  Number of template vectors owned by this rank
  int total_num_t{};                                     ///  Total number of template vectors across all ranks
  std::vector<int> num_t_per_rank;                       ///  Number of template vectors per rank
  std::vector<int> offset_per_rank;                      ///  Offset for each rank's template vectors in global numbering
  Dune::ParameterTree solver_ptree;                      ///  Parameters for the coarse solver, kept for update()
  std::string solver_subtree_name;                       ///  Subtree of solver_ptree holding the coarse solver settings
  CoarseSolveMode solve_mode{CoarseSolveMode::RankZero}; ///  Where the coarse problem is solved
  Logger::Event* apply_event{};                          ///  Logging event for timing the apply method
  Logger::Event* build_solver_event{};                   ///  Logging event for timing the solver building process
};
