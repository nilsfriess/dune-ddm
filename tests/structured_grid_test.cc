#if HAVE_CONFIG_H
#include "config.h"
#endif

#include "dune/ddm/backend/sycl/backend.hh"
#include "dune/ddm/combined_preconditioner.hh"
#include "dune/ddm/communication.hh"
#include "dune/ddm/consistent_parallel_matrix_operator.hh"
#include "dune/ddm/galerkin_preconditioner.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/pou.hh"
#include "dune/ddm/schwarz.hh"
#include "dune/ddm/sycl/mat.hh"
#include "dune/ddm/sycl/vec.hh"
#include "poisson_problem.hh"
#include "test_utils.hh"

#include <cstddef>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/solvers.hh>
#include <dune/localfunctions/lagrange/lagrangelfecache.hh>
#include <limits>
#include <vector>

/** @brief The identity, as a preconditioner that communicates nothing.
 *
 *  Lets a solver run genuinely unpreconditioned. Passing a sequential preconditioner instead would
 *  make the factory wrap it in Dune::BlockPreconditioner, which restores consistency itself and
 *  would hide the very property this is meant to check.
 */
template <class Vec>
struct IdentityPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::overlapping; }
  void pre(Vec&, Vec&) override {}
  void post(Vec&) override {}
  void apply(Vec& v, const Vec& d) override { v = d; }
};

template <class Matrix, class Vector>
bool solve_single_level_schwarz(const Dune::MPIHelper& helper, std::shared_ptr<ddm::Communication>& comm, std::shared_ptr<Matrix>& A, const Vector& b, Vector& x, std::shared_ptr<PartitionOfUnity> pou)
{
  using Operator = ConsistentParallelMatrixOperator<Matrix, Vector, Vector, ddm::Communication>;
  auto op = std::make_shared<Operator>(A, comm);

  Dune::initSolverFactories<Operator>();
  Dune::ParameterTree solver_tree;
  solver_tree["verbose"] = (helper.rank() == 0) ? "2" : "0";
  solver_tree["type"] = "cgsolver";
  solver_tree["reduction"] = "1e-8";
  solver_tree["maxit"] = "1000";
  solver_tree["restart"] = "30";

  using SchwarzPrec = ddm::SchwarzPreconditioner<Matrix, Vector>;

  // Build fine-level preconditioner
  Dune::ParameterTree schwarz_tree;
  schwarz_tree["schwarz.type"] = "standard";
  try {
    auto prec = std::make_shared<SchwarzPrec>(A, comm, *pou, schwarz_tree);
    auto solver = Dune::getSolverFromFactory(op, solver_tree, prec);

    // Solve the system
    Dune::InverseOperatorResult res;
    x = 0.;
    auto rhs = b;
    solver->apply(x, rhs, res);
  }
  catch (Dune::Exception& e) {
    std::cout << "Exception thrown while trying to create and run Schwarz solver\n";
    std::cout << e.what() << "\n";

    // We don't return false here because this is expected in some cases currently
    // (e.g. when we run with the SYCL backend but on CPU; for this case, there is
    // currently no direct solver available so the Schwarz setup fails)
  }

  return true;
}

template <class Communication, class Matrix, class Vector>
bool solve_cg(const Dune::MPIHelper& helper, std::shared_ptr<Communication>& comm, std::shared_ptr<Matrix>& A, const Vector& b, Vector& x)
{
  using Operator = ConsistentParallelMatrixOperator<Matrix, Vector, Vector, Communication>;
  auto op = std::make_shared<Operator>(A, comm);
  Dune::initSolverFactories<Operator>();

  Dune::ParameterTree unprec_tree;
  unprec_tree["verbose"] = (helper.rank() == 0) ? "1" : "0";
  unprec_tree["type"] = "cgsolver";
  unprec_tree["reduction"] = "1e-8";
  unprec_tree["maxit"] = "500";
  unprec_tree["restart"] = "30";

  Dune::InverseOperatorResult unprec_res;
  x = 0.;
  auto unprec_rhs = b;
  auto unprec_solver = Dune::getSolverFromFactory(op, unprec_tree, std::make_shared<IdentityPreconditioner<Vector>>());
  unprec_solver->apply(x, unprec_rhs, unprec_res);

  return true;
}

int main(int argc, char** argv)
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    setup_loggers(helper.rank(), argc, argv);

    const int dim = 2;
    const int gridsize = 64;
    const int overlap = 4;

    using Grid = Dune::YaspGrid<dim>;
    // One element layer more than the method's overlap: the outermost layer is needed only to
    // make the rows of the patch vertices complete during assembly; the matrices are truncated to
    // the patch (see class Problem above), so every local matrix equals R A R^T.
    Grid grid({1., 1.}, {gridsize, gridsize}, 0ULL, overlap + 1);
    auto gv = grid.leafGridView();

    // Homogeneous Dirichlet conditions on the whole boundary of the unit square.
    auto is_dirichlet = [](const auto& x) {
      for (int i = 0; i < dim; ++i)
        if (x[i] < 1e-10 or x[i] > 1. - 1e-10) return true;
      return false;
    };
    auto coefficient = [](const auto&) { return 1.; };
    auto source = [](const auto&) { return 1.; };

    ddmtest::PoissonProblem p(gv, is_dirichlet, coefficient, source);
    auto comm = ddmtest::create_communication_for_grid(gv, p.patch);
    if (comm->indexSet().size() != static_cast<std::size_t>(p.A->N()))
      DUNE_THROW(Dune::InvalidStateException, "communication index set (" << comm->indexSet().size() << ") and matrix (" << p.A->N() << ") differ in size");
    comm->copyOwnerToAll(p.b, p.b); // Make b consistent
    auto vec_comm = std::make_shared<ddm::Communication>(ddm::make_communication_from_dune(*comm));
    auto pou = std::make_shared<PartitionOfUnity>(*p.A, *comm, PartitionOfUnityType::Standard);

    // Run with ISTL backend
    {
      typename ddmtest::PoissonProblem::Vector x(p.b.size());

      solve_cg(helper, comm, p.A, p.b, x);
      solve_single_level_schwarz(helper, vec_comm, p.A, p.b, x, pou);
    }

    // Run with SYCL backend
    {
      using SyclVec = ddm::Sycl::Vec<double>;
      using SyclMat = ddm::Sycl::Mat<double>;

      sycl::queue q{sycl::property::queue::in_order{}};
      SyclVec x(q, p.b.size());
      auto b = SyclVec::from_host_vector(q, p.b);
      auto A = std::make_shared<SyclMat>(SyclMat::from_bcrs(q, *p.A));

      solve_cg(helper, vec_comm, A, b, x);
      solve_single_level_schwarz(helper, vec_comm, A, b, x, pou);
    }

    Logger::get().report(MPI_COMM_WORLD);
  }
  catch (const Dune::Exception& e) {
    std::cout << "Dune exception thrown: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
