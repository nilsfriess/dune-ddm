#pragma once

#include <cassert>
#include <cstddef>
#include <iostream>

#include <mpi.h>

#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/io.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>

#include "datahandles.hh"
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

    TODO: Extend from one template vector to a set of template vectors (of possibly different
          sizes on each MPI rank).
 */
template <class Vec, class Mat, class RemoteIndices, class Solver = Dune::UMFPack<Dune::BCRSMatrix<double>>>
class GalerkinPreconditioner : public Dune::Preconditioner<Vec, Vec> {
public:
  GalerkinPreconditioner(const Mat &A, const Vec &pou, const Vec &t, RemoteParallelIndices<RemoteIndices> ris) : ris(ris), restr_vecs(1, Vec(pou.N(), 0)), N(ris.second->size()), d_ovlp(N), x_ovlp(N)
  {
    registerLogEvents();
    buildCommunicationInterfaces(ris);
    buildRestrictionVector(pou, t, restr_vecs[0]);
    buildSolver(A);
  }

  GalerkinPreconditioner(const Mat &A, const Vec &pou, const std::vector<Vec> &ts, RemoteParallelIndices<RemoteIndices> ris)
      : ris(ris), restr_vecs(ts.size(), Vec(pou.N(), 0)), N(ris.second->size()), d_ovlp(N), x_ovlp(N)
  {
    registerLogEvents();
    buildCommunicationInterfaces(ris);
    for (std::size_t i = 0; i < ts.size(); ++i) {
      buildRestrictionVector(pou, ts[i], restr_vecs[i]);
    }
    buildSolver(A);
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void pre(Vec &, Vec &) override {}
  void post(Vec &) override {}

  void apply(Vec &x, const Vec &d) override
  {
    Logger::ScopedLog se(apply_event);

    // assert(restr_vecs.size() == 1);

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
    cvdh.setVec(d_ovlp);
    owner_copy_comm->forward(cvdh);

    // 2. Compute local contribution of coarse defect
    std::vector<double> d_local(restr_vecs.size(), 0);
    for (std::size_t k = 0; k < restr_vecs.size(); ++k) {
      for (std::size_t i = 0; i < restr_vecs[0].N(); ++i) {
        d_local[k] += restr_vecs[k][i] * d_ovlp[i];
      }
    }

    // 3. Gather defect values on rank 0
    Dune::BlockVector<double> d0(size * restr_vecs.size());
    MPI_Gather(d_local.data(), static_cast<int>(d_local.size()), MPI_DOUBLE, d0.data(), restr_vecs.size(), MPI_DOUBLE, 0, comm);

    // 4. Solve on rank 0
    Dune::BlockVector<double> x0(size * restr_vecs.size());
    if (rank == 0) {
      x0 = 0;
      Dune::InverseOperatorResult res;
      solver->apply(x0, d0, res);
    }

    // 5. Scatter the local solution back
    std::vector<double> coarse_solution(restr_vecs.size(), 0);
    MPI_Scatter(x0.data(), restr_vecs.size(), MPI_DOUBLE, coarse_solution.data(), coarse_solution.size(), MPI_DOUBLE, 0, comm);

    // 6. Prolongate
    x_ovlp = 0;
    for (std::size_t k = 0; k < restr_vecs.size(); ++k) {
      for (std::size_t j = 0; j < x_ovlp.N(); ++j) {
        x_ovlp[j] += coarse_solution[k] * restr_vecs[k][j];
      }
    }

    advdh.setVec(x_ovlp);
    all_all_comm->forward(advdh);

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
    mat_prod_event = Logger::get().registerEvent(family, "build Matrix");
  }

  void buildCommunicationInterfaces(const RemoteParallelIndices<RemoteIndices> &ris)
  {
    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    const AttributeSet ownerAttribute{Attribute::owner};
    const AttributeSet copyAttribute{Attribute::copy};

    all_all_interface.build(*ris.first, allAttributes, allAttributes);
    all_all_comm = std::make_unique<Dune::VariableSizeCommunicator<>>(all_all_interface);

    owner_copy_interface.build(*ris.first, ownerAttribute, copyAttribute);
    owner_copy_comm = std::make_unique<Dune::VariableSizeCommunicator<>>(owner_copy_interface);
  }

  void buildRestrictionVector(const Vec &pou, const Vec &t, Vec &restr)
  {
    // Initialise r to be the "template" vector and make consistent on the overlapping index set
    restr = 0;
    for (std::size_t i = 0; i < t.N(); ++i) {
      restr[i] = t[i];
    }

    cvdh.setVec(restr);
    owner_copy_comm->forward(cvdh);

    // Multiply with the partition of unity
    for (std::size_t i = 0; i < restr.N(); ++i) {
      restr[i] *= pou[i];
    }
  }

  void buildSolver(const Mat &A)
  {
    Logger::ScopedLog se(mat_prod_event);

    auto *s_event = Logger::get().registerEvent("GalerkinPrec", "comm s");
    auto *y_event = Logger::get().registerEvent("GalerkinPrec", "comm y");
    auto *Asy_event = Logger::get().registerEvent("GalerkinPrec", "compute y=As");
    auto *gather_A0 = Logger::get().registerEvent("GalerkinPrec", "gather A0");
    auto *factor_A0 = Logger::get().registerEvent("GalerkinPrec", "factor A0");

    Dune::GlobalLookupIndexSet glis(*ris.second);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(ris.first->communicator(), &rank);
    MPI_Comm_size(ris.first->communicator(), &size);

    // First check that all MPI ranks have the same number of template vectors
    std::size_t num_t = restr_vecs.size();
    std::size_t max_num_t = 0;
    std::size_t min_num_t = 0;
    MPI_Allreduce(&num_t, &max_num_t, 1, MPI_UNSIGNED_LONG, MPI_MAX, ris.first->communicator());
    MPI_Allreduce(&num_t, &min_num_t, 1, MPI_UNSIGNED_LONG, MPI_MIN, ris.first->communicator());
    if (max_num_t != min_num_t) {
      std::cout << "Error: All MPI ranks must have the same number of template vectors\n";
      MPI_Abort(ris.first->communicator(), 1);
    }

    // if (rank == 1) {
    //   for (auto &restr_vec : restr_vecs) {
    //     Dune::printvector(std::cout, restr_vec, "", "");
    //   }
    // }

    std::vector<bool> owner_mask(restr_vecs[0].N(), false);
    for (std::size_t l = 0; l < owner_mask.size(); ++l) {
      owner_mask[l] = glis.pair(l)->local().attribute() == Attribute::owner;
    }

    std::vector<Vec> my_rows(num_t, Vec(size * num_t));
    Vec s(restr_vecs[0].N());
    Vec y(restr_vecs[0].N());

    for (std::size_t k = 0; k < num_t; ++k) {
      const Vec &r = restr_vecs[k];
      for (int col = 0; col < size; ++col) {
        for (std::size_t i = 0; i < num_t; ++i) {
          if (col == rank) {
            s = restr_vecs[i];
          }
          else {
            s = 0;
          }

          Logger::get().startEvent(s_event);
          advdh.setVec(s);
          all_all_comm->forward(advdh);
          Logger::get().endEvent(s_event);

          Logger::get().startEvent(Asy_event);
          A.mv(s, y);

          for (std::size_t l = 0; l < y.N(); ++l) {
            y[l] *= owner_mask[l];
          }
          Logger::get().endEvent(Asy_event);

          Logger::get().startEvent(y_event);
          advdh.setVec(y);
          all_all_comm->forward(advdh);
          Logger::get().endEvent(y_event);

          my_rows[k][col * num_t + i] = r * y;
        }
      }
    }

    Logger::get().startEvent(gather_A0);
    A0 = gatherMatrixFromRows(my_rows, ris.first->communicator());
    Logger::get().endEvent(gather_A0);

    Logger::get().startEvent(factor_A0);
    if (rank == 0) {
      solver = std::make_unique<Solver>(A0);
    }
    Logger::get().endEvent(factor_A0);
  }

  std::unique_ptr<Solver> solver;
  RemoteParallelIndices<RemoteIndices> ris;

  std::vector<Vec> restr_vecs;

  std::size_t N;
  Vec d_ovlp;
  Vec x_ovlp;

  Dune::Interface all_all_interface;
  std::unique_ptr<Dune::VariableSizeCommunicator<>> all_all_comm;

  Dune::Interface owner_copy_interface;
  std::unique_ptr<Dune::VariableSizeCommunicator<>> owner_copy_comm;

  CopyVectorDataHandle<Vec> cvdh;
  AddVectorDataHandle<Vec> advdh;

  Dune::BCRSMatrix<double> A0;

  Logger::Event *apply_event{};
  Logger::Event *mat_prod_event{};
};
