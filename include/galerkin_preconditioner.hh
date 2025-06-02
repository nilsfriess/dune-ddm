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

/** @brief A preconditioner that acts as R^T (R A R^T)^-1 R.

    The restriction matrix R (which is of size 'A.N()' x 'MPI ranks') is built from
      - a so called template vector \p t, and
      - a partition of unity \p pou.
    Here, the template vector should live on a non-overlapping index set whereas the
    partition of unity should be defined on an overlapping index set that matches the
    parallel index set stored in \p ris. The callee is responsible for passing a
    suitable template vector (e.g., zeroed out on Dirichlet dofs).

    One column of R^T is constructed by taking the template vector, extending it to
    the overlapping index set (by exchanging the values with other MPI ranks), multiplying
    this vector pointwise with the partition of unity and finally extending by zeros to
    the whole domain (of course the last step is not actually carried out). Each MPI rank
    constructs one column of R^T.

    To compute the Galerkin product A_0 = R A R^T, it is expected that A is already
    extended to an overlapping matrix (again matching the parallel index set).

    The typical use-case for this preconditioner is a two-level Schwarz method, where it
    is used in conjunction with a single-level Schwarz method using the helper class
    `CombinedPreconditioner`.
 */
template <class Vec, class Mat, class RemoteIndices, class Solver = Dune::UMFPack<Dune::BCRSMatrix<double>>>
class GalerkinPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  struct AddGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] += v; }
  };

  struct CopyGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] = v; }
  };

  struct VecDistributor {
    using value_type = double;

    VecDistributor(const Vec &temp, const std::vector<int> &neighbours) : temp{temp}
    {
      for (const auto &nb : neighbours) {
        others.emplace(nb, temp);
        others[nb] = 0;
      }
    }

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

    Vec *own = nullptr;
    std::map<int, Vec> others;
    Vec temp;
  };

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
  // GalerkinPreconditioner(const Mat &A, const Vec &pou, const Vec &t, RemoteParallelIndices<RemoteIndices> ris) : ris(ris), restr_vecs(1, Vec(pou.N(), 0)), N(ris.second->size()), d_ovlp(N),
  // x_ovlp(N)
  // {
  //   registerLogEvents();
  //   buildCommunicationInterfaces(ris);
  //   buildRestrictionVector(pou, t, restr_vecs[0]);
  //   buildSolver(A);
  // }

  GalerkinPreconditioner(const Mat &A, const std::vector<Vec> &ts, RemoteParallelIndices<RemoteIndices> ris) : ris(ris), N(ris.second->size()), d_ovlp(N), x_ovlp(N), num_t(ts.size())
  {
    registerLogEvents();
    buildCommunicationInterfaces(ris);

    if (ts.size() == 0) {
      DUNE_THROW(Dune::Exception, "Must at least pass one template vector");
    }

    if (ts[0].N() != A.N()) {
      DUNE_THROW(Dune::Exception, "Template vectors must match size of matrix");
    }

    auto max_num_t = getMaxTemplateVecs(ts.size());
    Vec zero(ts[0].N());
    zero = 0;
    restr_vecs.resize(max_num_t, Vec(ts[0].N()));
    for (int i = 0; i < num_t; ++i) {
      restr_vecs[i] = ts[i];
    }

    spdlog::info("Setting up GalerkinPreconditioner with {} template vector{} (max. one rank has is {})", num_t, (num_t == 1 ? "" : "s"), max_num_t);

    buildSolver(A);
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

  const Dune::BCRSMatrix<double> &getCoarseMatrix() const { return A0; }

private:
  void registerLogEvents()
  {
    auto *family = Logger::get().registerFamily("GalerkinPrec");
    apply_event = Logger::get().registerEvent(family, "apply");
    build_solver_event = Logger::get().registerEvent(family, "build Matrix");
  }

  void buildCommunicationInterfaces(const RemoteParallelIndices<RemoteIndices> &ris)
  {
    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    const AttributeSet ownerAttribute{Attribute::owner};
    const AttributeSet copyAttribute{Attribute::copy};

    all_all_interface.build(*ris.first, allAttributes, allAttributes);
    all_all_comm.build<Vec>(all_all_interface);

    owner_copy_interface.build(*ris.first, ownerAttribute, copyAttribute);
    owner_copy_comm.build<Vec>(owner_copy_interface);
  }

  std::size_t getMaxTemplateVecs(std::size_t num_restr_vecs) const
  {
    std::size_t max_num_t = 0;
    MPI_Allreduce(&num_restr_vecs, &max_num_t, 1, MPI_UNSIGNED_LONG, MPI_MAX, ris.first->communicator());
    return max_num_t;
  }

  // void buildRestrictionVector(const Vec &pou, const Vec &t, Vec &restr)
  // {
  //   // Initialise r to be the "template" vector and make consistent on the overlapping index set
  //   restr = 0;
  //   for (std::size_t i = 0; i < t.N(); ++i) {
  //     restr[i] = t[i];
  //   }

  //   cvdh.setVec(restr);
  //   owner_copy_comm->forward(cvdh);

  //   // Multiply with the partition of unity
  //   for (std::size_t i = 0; i < restr.N(); ++i) {
  //     restr[i] *= pou[i];
  //   }
  // }

  // TODO: Remove some of the logging, this was just added to find out where the most time is spent,
  //       because this function can become the bottleneck for large simulations.
  void buildSolver(const Mat &A)
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
    A0 = gatherMatrixFromRowsFlat(my_rows_flat, total_num_t, ris.first->communicator());
    Logger::get().endEvent(gather_A0);

    Logger::get().startEvent(factor_A0);
    if (rank == 0) {
      spdlog::info("Size of coarse space matrix: {}x{}", A0.N(), A0.M());
      solver = std::make_unique<Solver>(A0);
      solver->setOption(UMFPACK_IRSTEP, 0);
    }
    Logger::get().endEvent(factor_A0);
  }

  std::unique_ptr<Solver> solver;
  RemoteParallelIndices<RemoteIndices> ris;

  std::vector<Vec> restr_vecs;

  std::size_t N;
  Vec d_ovlp;
  Vec x_ovlp;

  int num_t;         // how many template vectors we own
  int total_num_t{}; // how many template vectors we own
  std::vector<int> num_t_per_rank;
  std::vector<int> offset_per_rank;

  Dune::Interface all_all_interface;
  Dune::BufferedCommunicator all_all_comm;

  Dune::Interface owner_copy_interface;
  Dune::BufferedCommunicator owner_copy_comm;

  Dune::BCRSMatrix<double> A0;

  Logger::Event *apply_event{};
  Logger::Event *build_solver_event{};
};
