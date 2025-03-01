#pragma once

#include <cassert>
#include <cstddef>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>

#include "datahandles.hh"
#include "helpers.hh"
#include "logger.hh"

/** @brief A preconditioner that acts as R^T (R A R^T)^-1 R

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

  GalerkinPreconditioner(const Mat &A, const Vec &pou, const std::vector<Vec> &ts, RemoteParallelIndices<RemoteIndices> ris) : ris(ris), restr_vecs(ts.size(), Vec(pou.N(), 0)), N(ris.second->size()), d_ovlp(N), x_ovlp(N)
  {
    registerLogEvents();
    buildCommunnicationInterfaces(ris);
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

    assert(restr_vecs.size() == 1);

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
    double d_local = 0;
    for (std::size_t i = 0; i < restr_vecs[0].N(); ++i) {
      d_local += restr_vecs[0][i] * d_ovlp[i];
    }

    // 3. Gather defect values on rank 0
    Dune::BlockVector<double> d0(size);
    MPI_Gather(&d_local, 1, MPI_DOUBLE, d0.data(), 1, MPI_DOUBLE, 0, comm);

    // 4. Solve on rank 0
    Dune::BlockVector<double> x0(size);
    if (rank == 0) {
      x0 = 0;
      Dune::InverseOperatorResult res;
      solver->apply(x0, d0, res);
    }

    // 5. Scatter the local solution back
    double coarse_solution{};
    MPI_Scatter(x0.data(), 1, MPI_DOUBLE, &coarse_solution, 1, MPI_DOUBLE, 0, comm);

    // 6. Prolongate
    x_ovlp = restr_vecs[0];
    x_ovlp *= coarse_solution;

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

    Dune::GlobalLookupIndexSet glis(*ris.second);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(ris.first->communicator(), &rank);
    MPI_Comm_size(ris.first->communicator(), &size);

    if (restr_vecs.size() == 1) {
      Vec my_row(size, 0); // Each process builds one row of the coarse matrix

      // Compute r * A * s, where s is the r vector of another rank
      Vec s(restr_vecs[0].N());
      Vec y(restr_vecs[0].N());
      for (int col = 0; col < size; ++col) {
        if (col == rank) {
          s = restr_vecs[0];
        }
        else {
          s = 0;
        }

        advdh.setVec(s);
        all_all_comm->forward(advdh);

        A.mv(s, y);

        for (std::size_t i = 0; i < y.N(); ++i) {
          if (glis.pair(i)->local().attribute() != Attribute::owner) {
            y[i] = 0;
          }
        }

        advdh.setVec(y);
        all_all_comm->forward(advdh);

        my_row[col] = restr_vecs[0] * y;
      }

      A0 = gatherMatrixFromRows(my_row, ris.first->communicator());

      if (rank == 0) {
        solver = std::make_unique<Solver>(A0);
      }
    }
    else {
      assert(false && "Not implemented yet");
    }
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
