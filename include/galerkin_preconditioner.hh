#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <dune/common/parallel/interface.hh>
#include <numeric>

#include <mpi.h>

#include <dune/common/parallel/communicator.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>

#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "helpers.hh"
#include "logger.hh"

/** @brief A Galerkin-type preconditioner that implements R^T (R A R^T)^-1 R.

    This preconditioner constructs a coarse space correction using multiple template vectors
    provided by each MPI rank. The restriction matrix R has dimensions 'total_template_vectors' 
    x 'matrix_size', where each column corresponds to one template vector from one rank.

    <b>Algorithm Overview:</b>
    1. Each MPI rank provides one or more template vectors defined on the overlapping subdomain
    2. The restriction matrix R^T is formed by collecting all template vectors from all ranks
    3. The coarse matrix A_c = R A R^T is assembled by computing all pairwise products 
       between template vectors: (R_i, A R_j) for all i,j
    4. The coarse system A_c x_c = R d is solved centrally on rank 0
    5. The correction is prolongated: x += R^T x_c

    <b>Requirements:</b>
    - Template vectors must be defined on the overlapping subdomain matching the index set in 'ris'
    - The matrix A must be extended to the same overlapping subdomain
    - Template vectors should be suitable for the problem (e.g., zero on Dirichlet boundaries)

    <b>Typical Use Case:</b>
    This preconditioner is commonly used as the coarse space component in two-level domain 
    decomposition methods, combined with a local preconditioner via `CombinedPreconditioner`.

    <b>Scalability:</b>
    The coarse problem size equals the total number of template vectors across all ranks.
    For good scalability, this should grow slowly with the number of MPI processes.
 */
template <class Vec, class Mat, class RemoteIndices, class Solver = Dune::UMFPack<Dune::BCRSMatrix<double>>>
class GalerkinPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  /** @brief Gather-scatter helper for adding values during communication */
  struct AddGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] += v; }
  };

  /** @brief Gather-scatter helper for copying values during communication */
  struct CopyGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] = v; }
  };

  /** @brief Helper class for distributing vectors across MPI ranks during communication */
  struct VecDistributor {
    using value_type = double;

    /**
     * @brief Constructor that initializes vector distributor.
     * @param temp Template vector used for creating neighbor vectors
     * @param neighbours List of neighbor rank IDs
     */
    VecDistributor(const Vec &temp, const std::vector<int> &neighbours) : temp{temp}
    {
      for (const auto &nb : neighbours) {
        others.emplace(nb, temp);
        others[nb] = 0;
      }
    }

    /** @brief Clear all neighbor vectors (set to zero) */
    void clear()
    {
      for (auto &[rank, vec] : others) {
        vec = 0;
      }
    }

    VecDistributor(const VecDistributor &) = delete;
    VecDistributor(const VecDistributor &&) = delete;
    VecDistributor &operator=(const VecDistributor &) = delete;
    VecDistributor &operator=(const VecDistributor &&) = delete;
    ~VecDistributor() = default;

    Vec *own = nullptr;                /**< Pointer to our own vector */
    std::map<int, Vec> others;        /**< Vectors from other ranks */
    Vec temp;                         /**< Template vector for initialization */
  };

  /** @brief Gather-scatter helper with rank information for VecDistributor */
  struct CopyGatherScatterWithRank {
    using DataType = double;

    static DataType gather(const VecDistributor &vd, std::size_t i) { return (*vd.own)[i]; }
    static void scatter(VecDistributor &vd, DataType v, std::size_t i, int rank)
    {
      if (not vd.others.contains(rank)) {
        spdlog::get("all_ranks")->error("Rank {} is no neighbour", rank);
        MPI_Abort(MPI_COMM_WORLD, 17);
      }
      vd.others[rank][i] = v;
    }
  };

public:
  /**
   * @brief Constructor for the Galerkin preconditioner.
   * 
   * Sets up a Galerkin-type preconditioner that computes R^T (R A R^T)^-1 R,
   * where R is the restriction matrix built from template vectors.
   * 
   * @param A The overlapping matrix (must match the overlapping index set in ris)
   * @param ts Vector of template vectors (must be overlapping, size must match A.N())
   * @param ris Remote parallel indices describing the overlapping index set
   * 
   * @throws Dune::Exception if no template vectors are provided or if template vector size doesn't match matrix size
   */
  GalerkinPreconditioner(const Mat &A, const std::vector<Vec> &ts, RemoteParallelIndices<RemoteIndices> ris) : ris(ris), n(ris.second->size()), d_ovlp(n), x_ovlp(n), num_t(ts.size())
  {
    register_log_events();
    build_communication_interfaces(ris);

    if (ts.size() == 0) {
      DUNE_THROW(Dune::Exception, "Must at least pass one template vector");
    }

    if (ts[0].N() != A.N()) {
      DUNE_THROW(Dune::Exception, "Template vectors must match size of matrix");
    }

    auto max_num_t = get_max_template_vecs(ts.size());
    Vec zero(ts[0].N());
    zero = 0;
    restr_vecs.resize(max_num_t, Vec(ts[0].N()));
    for (int i = 0; i < num_t; ++i) {
      restr_vecs[i] = ts[i];
    }

    spdlog::info("Setting up GalerkinPreconditioner with {} template vector{} (max. one rank has is {})", num_t, (num_t == 1 ? "" : "s"), max_num_t);

    build_solver(A);
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void pre(Vec &, Vec &) override {}
  void post(Vec &) override {}

  void apply(Vec &x, const Vec &d) override
  {
    Logger::ScopedLog se(apply_event);

    MPI_Comm comm = ris.first->communicator();
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. Copy local values from non-overlapping to overlapping defect
    for (std::size_t i = 0; i < d.N(); ++i) {
      d_ovlp[i] = d[i];
    }

    // 1.5 Get remaining values from other ranks
    owner_copy_comm.forward<CopyGatherScatter>(d_ovlp);

    // 2. Compute local contribution of coarse defect
    std::vector<double> d_local(num_t, 0);
    for (int k = 0; k < num_t; ++k) {
      for (std::size_t i = 0; i < restr_vecs[0].N(); ++i) {
        d_local[k] += restr_vecs[k][i] * d_ovlp[i];
      }
    }

    // 3. Gather defect values on rank 0
    Dune::BlockVector<double> d0(total_num_t);
    MPI_Gatherv(d_local.data(), static_cast<int>(d_local.size()), MPI_DOUBLE, d0.data(), num_t_per_rank.data(), offset_per_rank.data(), MPI_DOUBLE, 0, comm);

    // 4. Solve on rank 0
    Dune::BlockVector<double> x0(total_num_t);
    if (rank == 0) {
      x0 = 0;
      Dune::InverseOperatorResult res;
      solver->apply(x0, d0, res);
    }

    // 5. Scatter the local solution back
    std::vector<double> coarse_solution(num_t);
    MPI_Scatterv(x0.data(), num_t_per_rank.data(), offset_per_rank.data(), MPI_DOUBLE, coarse_solution.data(), num_t, MPI_DOUBLE, 0, comm);

    // 6. Prolongate
    x_ovlp = 0;
    for (int k = 0; k < num_t; ++k) {
      for (std::size_t j = 0; j < x_ovlp.N(); ++j) {
        x_ovlp[j] += coarse_solution[k] * restr_vecs[k][j];
      }
    }

    // advdh.setVec(x_ovlp);
    all_all_comm.forward<AddGatherScatter>(x_ovlp);

    // 7. Restrict the solution to the non-overlapping subdomain
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = x_ovlp[i];
    }
  }

  /** @brief Get the assembled coarse matrix R A R^T for inspection or debugging */
  const Dune::BCRSMatrix<double> &get_coarse_matrix() const { return a0; }

private:
  /** @brief Register logging events for performance monitoring */
  void register_log_events()
  {
    auto *family = Logger::get().registerFamily("GalerkinPrec");
    apply_event = Logger::get().registerEvent(family, "apply");
    build_solver_event = Logger::get().registerEvent(family, "build Matrix");
  }

  /** 
   * @brief Build communication interfaces for parallel operations
   * @param ris Remote parallel indices describing the overlapping index set and communication patterns
   */
  void build_communication_interfaces(const RemoteParallelIndices<RemoteIndices> &ris)
  {
    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    const AttributeSet ownerAttribute{Attribute::owner};
    const AttributeSet copyAttribute{Attribute::copy};

    all_all_interface.build(*ris.first, allAttributes, allAttributes);
    all_all_comm.build<Vec>(all_all_interface);

    owner_copy_interface.build(*ris.first, ownerAttribute, copyAttribute);
    owner_copy_comm.build<Vec>(owner_copy_interface);
  }

  /** 
   * @brief Get the maximum number of template vectors across all MPI ranks
   * @param num_restr_vecs Number of template vectors on this rank
   * @return Maximum number of template vectors on any rank
   */
  std::size_t get_max_template_vecs(std::size_t num_restr_vecs) const
  {
    std::size_t max_num_t = 0;
    MPI_Allreduce(&num_restr_vecs, &max_num_t, 1, MPI_UNSIGNED_LONG, MPI_MAX, ris.first->communicator());
    return max_num_t;
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
   */
  // TODO: Remove some of the logging, this was just added to find out where the most time is spent,
  //       because this function can become the bottleneck for large simulations.
  void build_solver(const Mat &A)
  {
    Logger::ScopedLog se(build_solver_event);

    auto *comm_begin_event = Logger::get().registerEvent("GalerkinPrec", "begin comm");
    auto *comm_end_event = Logger::get().registerEvent("GalerkinPrec", "end comm");
    auto *local_local_sp_event = Logger::get().registerEvent("GalerkinPrec", "dot (local<>local)");
    auto *local_remote_sp_event = Logger::get().registerEvent("GalerkinPrec", "dot (local<>remote)");
    auto *gather_A0 = Logger::get().registerEvent("GalerkinPrec", "gather A0");
    auto *factor_A0 = Logger::get().registerEvent("GalerkinPrec", "factor A0");
    auto *prepare_event = Logger::get().registerEvent("GalerkinPrec", "prepare");

    Logger::get().startEvent(prepare_event);
    Dune::GlobalLookupIndexSet glis(*ris.second);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(ris.first->communicator(), &rank);
    MPI_Comm_size(ris.first->communicator(), &size);

    std::vector<bool> neighbours(size, false);
    for (auto nid : ris.first->getNeighbours()) {
      neighbours[nid] = true;
    }

    std::vector<bool> owner_mask(restr_vecs[0].N(), false);
    for (std::size_t l = 0; l < owner_mask.size(); ++l) {
      owner_mask[l] = glis.pair(l)->local().attribute() == Attribute::owner;
    }

    // Find out how many template vectors each rank has and how large coarse matrix will be
    num_t_per_rank.resize(size);
    MPI_Allgather(&num_t, 1, MPI_INT, num_t_per_rank.data(), 1, MPI_INT, ris.first->communicator());
    total_num_t = std::accumulate(num_t_per_rank.begin(), num_t_per_rank.end(), 0UL);

    std::vector<double> my_rows_flat(static_cast<std::size_t>(num_t) * total_num_t);
    Vec s(restr_vecs[0].N());
    Vec y(restr_vecs[0].N());

    offset_per_rank.resize(size);
    std::exclusive_scan(num_t_per_rank.begin(), num_t_per_rank.end(), offset_per_rank.begin(), 0);

    Logger::get().endEvent(prepare_event);
#if 1
    auto max_num_t = *std::max_element(num_t_per_rank.begin(), num_t_per_rank.end());

    Vec zerovec(s);
    zerovec = 0;
    std::vector<int> neighbour_vec(ris.first->getNeighbours().begin(), ris.first->getNeighbours().end());
    VecDistributor vd(zerovec, neighbour_vec);

    Dune::BufferedCommunicator bcomm;
    bcomm.build<VecDistributor>(all_all_interface);

    std::map<int, Vec> basis_vector_buffer; // TODO: Check if using this extra buffer actually helps (it allows to do more computation during communication but requires copying around some vectors).

    for (int idx = 0; idx < max_num_t; ++idx) {
      if (idx < num_t) {
        vd.own = &restr_vecs[idx];
      }
      else {
        vd.own = &zerovec;
      }

      Logger::get().startEvent(comm_begin_event);
      bcomm.forwardBegin<CopyGatherScatterWithRank>(vd);
      Logger::get().endEvent(comm_begin_event);

      if (idx > 0) {
        for (auto &&[rank, basis_vec] : vd.others) {
          basis_vector_buffer[rank] = basis_vec;
        }
      }

      // Compute scalar products for local * A * local vectors
      Logger::get().startEvent(local_local_sp_event);
      if (idx < num_t) {
        A.mv(restr_vecs[idx], y);
        for (int k = 0; k < num_t; ++k) {
          my_rows_flat[(offset_per_rank[rank] + idx) * num_t + k] = restr_vecs[k] * y;
        }
      }
      Logger::get().endEvent(local_local_sp_event);

      if (idx > 0) {
        // Compute scalar products for local * A * remote vectors.
        // The remote vectors are those that were send and received in the previous iteration.
        Logger::get().startEvent(local_remote_sp_event);
        for (const auto &nb : neighbour_vec) {
          if (idx - 1 >= num_t_per_rank[nb]) {
            continue;
          }

          A.mv(basis_vector_buffer[nb], y);
          for (int k = 0; k < num_t; ++k) {
            my_rows_flat[(offset_per_rank[nb] + idx - 1) * num_t + k] = restr_vecs[k] * y;
          }
        }
        Logger::get().endEvent(local_remote_sp_event);
      }

      vd.clear();
      // Wait for communication to finish
      Logger::get().startEvent(comm_end_event);
      bcomm.forwardEnd<CopyGatherScatterWithRank>(vd);
      Logger::get().endEvent(comm_end_event);
    }

    // Compute scalar products for local * A * remote vectors for the last basis vectors
    // that we missed above.
    Logger::get().startEvent(local_remote_sp_event);
    for (const auto &nb : neighbour_vec) {
      if (max_num_t - 1 >= num_t_per_rank[nb]) {
        continue;
      }

      A.mv(vd.others[nb], y);
      for (int k = 0; k < num_t; ++k) {
        my_rows_flat[(offset_per_rank[nb] + max_num_t - 1) * num_t + k] = restr_vecs[k] * y;
      }
    }
    Logger::get().endEvent(local_remote_sp_event);
#else
    auto *dot_vecs_event = Logger::get().registerEvent("GalerkinPrec", "dot products");
    auto *s_event = Logger::get().registerEvent("GalerkinPrec", "comm s");
    auto *y_event = Logger::get().registerEvent("GalerkinPrec", "comm y");
    auto *Asy_event = Logger::get().registerEvent("GalerkinPrec", "compute y=As");

    for (int col = 0; col < size; ++col) {
      for (std::size_t i = 0; i < num_t_per_rank[col]; ++i) {
        if (col == rank) {
          s = restr_vecs[i];
        }
        else {
          s = 0;
        }

        Logger::get().startEvent(s_event);
        all_all_comm.forward<AddGatherScatter>(s);
        Logger::get().endEvent(s_event);

        y = 0;
        // Below, we're computing products of the form s' * A * r, for all combinations of restriction vectors s and r.
        // This product will only be nonzero, if either we set some values in the 's' vector, or one of our neighbours
        // set some values in their 's' vector. For a rank that we share no degrees of freedom with, this will always be
        // zero and we can skip the computations. Unfortunately, we can't skip the communication.
        // TODO: Rewrite the whole communication interface so we can actually skip the communication here.
        if (col == rank or neighbours[col]) {
          Logger::get().startEvent(Asy_event);
          A.mv(s, y);

          for (std::size_t l = 0; l < y.N(); ++l) {
            y[l] *= owner_mask[l];
          }
          Logger::get().endEvent(Asy_event);
        }

        Logger::get().startEvent(y_event);
        all_all_comm.forward<AddGatherScatter>(y);
        Logger::get().endEvent(y_event);

        // Again, we skip the computation if we know it will be zero. In fact, these dot products are often
        // the longest taking step in this loop, so skipping unecessary computations here is crucial.
        // TODO: Is this actually true? We have to wait for the other ranks anyways so I'm not sure if
        //       saving some computations locally is really that important.

        // Note that my_rows is initialised with zeros, so just skipping here is fine.
        if (col == rank or neighbours[col]) {
          Logger::get().startEvent(dot_vecs_event);
          for (std::size_t k = 0; k < num_t; ++k) {
            my_rows[k][offset_per_rank[col] + i] = restr_vecs[k] * y;
          }
          Logger::get().endEvent(dot_vecs_event);
        }
      }
    }
#endif

    Logger::get().startEvent(gather_A0);
    a0 = gatherMatrixFromRowsFlat(my_rows_flat, total_num_t, ris.first->communicator());
    Logger::get().endEvent(gather_A0);

    Logger::get().startEvent(factor_A0);
    if (rank == 0) {
      spdlog::info("Size of coarse space matrix: {}x{}, nonzeros: {}", a0.N(), a0.M(), a0.nonzeroes());
      solver = std::make_unique<Solver>(a0);
      solver->setOption(UMFPACK_IRSTEP, 0);
    }
    Logger::get().endEvent(factor_A0);
  }

  /** @brief Direct solver for the coarse problem (UMFPack by default) */
  std::unique_ptr<Solver> solver;
  
  /** @brief Remote parallel indices describing the overlapping index set */
  RemoteParallelIndices<RemoteIndices> ris;

  /** @brief Template vectors used to build the restriction matrix */
  std::vector<Vec> restr_vecs;

  /** @brief Size of the overlapping index set */
  std::size_t n;
  
  /** @brief Overlapping defect vector for temporary storage */
  Vec d_ovlp;
  
  /** @brief Overlapping solution vector for temporary storage */
  Vec x_ovlp;

  /** @brief Number of template vectors owned by this rank */
  int num_t;
  
  /** @brief Total number of template vectors across all ranks */
  int total_num_t{};
  
  /** @brief Number of template vectors per rank */
  std::vector<int> num_t_per_rank;
  
  /** @brief Offset for each rank's template vectors in global numbering */
  std::vector<int> offset_per_rank;

  /** @brief Communication interface for all-to-all communication */
  Dune::Interface all_all_interface;
  
  /** @brief Buffered communicator for all-to-all communication */
  Dune::BufferedCommunicator all_all_comm;

  /** @brief Communication interface for owner-to-copy communication */
  Dune::Interface owner_copy_interface;
  
  /** @brief Buffered communicator for owner-to-copy communication */
  Dune::BufferedCommunicator owner_copy_comm;

  /** @brief Assembled coarse matrix R A R^T */
  Dune::BCRSMatrix<double> a0;

  /** @brief Logging event for timing the apply method */
  Logger::Event *apply_event{};
  
  /** @brief Logging event for timing the solver building process */
  Logger::Event *build_solver_event{};
};
