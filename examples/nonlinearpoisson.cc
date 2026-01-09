#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dune/ddm/logger.hh"
#include "dune/ddm/twolevel_schwarz.hh"
#include "nonlinearpoisson.hh"

#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/pdelab.hh>

template <typename Number>
class NonlinearPoissonProblem {
  Number eta;

public:
  using value_type = Number;

  //! Constructor without arg sets nonlinear term to zero
  NonlinearPoissonProblem()
      : eta(0.0)
  {
  }

  //! Constructor takes eta parameter
  NonlinearPoissonProblem(const Number& eta_)
      : eta(eta_)
  {
  }

  //! nonlinearity
  Number q(Number u) const { return eta * u * u; }

  //! derivative of nonlinearity
  Number qprime(Number u) const { return 2 * eta * u; }

  //! right hand side
  template <typename E, typename X>
  Number f(const E& e, const X& x) const
  {
    auto global = e.geometry().global(x);
    return -2.0 * x.size() + eta * global.two_norm2() * global.two_norm2();
  }

  //! boundary condition type function (true = Dirichlet)
  template <typename I, typename X>
  bool b(const I& /*i*/, const X& /*x*/) const
  {
    return true;
  }

  //! Dirichlet extension
  template <typename E, typename X>
  Number g(const E& e, const X& x) const
  {
    auto global = e.geometry().global(x);
    return global.two_norm2();
  }

  //! Neumann boundary condition
  template <typename I, typename X>
  Number j(const I&, const X&) const
  {
    return 0.0;
  }
};

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  setup_loggers(helper.rank(), argc, argv);

  // Create parameter tree
  Dune::ParameterTree ptree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readINITree("nonlinearpoisson.ini", ptree);
  ptreeparser.readOptions(argc, argv, ptree);

  // Create grid
  constexpr auto dim = 2;
  using Grid = Dune::UGGrid<dim>;
  unsigned int gridsize = ptree.get("gridsize", 32);
  if (ptree.hasKey("gridsize_per_rank")) {
    auto grid_sqrt = static_cast<int>(std::pow(helper.size(), 1. / dim));
    gridsize = ptree.get<int>("gridsize_per_rank") * grid_sqrt;
  }

  auto grid = Dune::StructuredGridFactory<Grid>::createSimplexGrid({0, 0}, {1, 1}, {gridsize, gridsize});

  using GridView = typename Grid::LeafGridView;
  auto gv = grid->leafGridView();
  auto part = Dune::ParMetisGridPartitioner<GridView>::partition(gv, helper);
  grid->loadBalance(part, 0);
  grid->globalRefine(ptree.get("refine", 0));

  // Finite element map
  using EntitySet = Dune::PDELab::OverlappingEntitySet<GridView>;
  using DomainField = GridView::Grid::ctype;
  using RangeType = double;
  const int degree = 2;
  using FiniteElementMap = Dune::PDELab::PkLocalFiniteElementMap<EntitySet, DomainField, RangeType, degree>;
  EntitySet es(gv);
  FiniteElementMap finiteElementMap(es);

  // Grid function space
  using Constraints = Dune::PDELab::ConformingDirichletConstraints;
  using VectorBackend = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>;
  using GridFunctionSpace = Dune::PDELab::GridFunctionSpace<EntitySet, FiniteElementMap, Constraints, VectorBackend>;
  GridFunctionSpace gridFunctionSpace(es, finiteElementMap);
  gridFunctionSpace.name("numerical_solution");

  // Solution vector
  using CoefficientVector = Dune::PDELab::Backend::Vector<GridFunctionSpace, DomainField>;
  CoefficientVector coefficientVector(gridFunctionSpace);

  // Discrete grid function of solution vetor
  using DiscreteGridFunction = Dune::PDELab::DiscreteGridFunction<GridFunctionSpace, CoefficientVector>;
  DiscreteGridFunction discreteGridFunction(gridFunctionSpace, coefficientVector);

  // Local operator (problem is nonlinear and problem class depends on the solution)
  using Problem = NonlinearPoissonProblem<RangeType>;
  Problem problem(1.0);
  using LocalOperator = NonlinearPoissonFEM<Problem, FiniteElementMap>;
  LocalOperator localOperator(problem, 1);

  // Create constraints map
  using ConstraintsContainer = typename GridFunctionSpace::template ConstraintsContainer<RangeType>::Type;
  ConstraintsContainer constraintsContainer;
  auto blambda = [&](const auto& i, const auto& x) { return problem.b(i, x); };
  auto bctype = Dune::PDELab::makeBoundaryConditionFromCallable(gv, blambda);
  Dune::PDELab::constraints(bctype, gridFunctionSpace, constraintsContainer);

  // Grid operator
  using MatrixBackend = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  const auto dofestimate = 4 * gridFunctionSpace.maxLocalSize();
  MatrixBackend matrixBackend(dofestimate);
  using GridOperator = Dune::PDELab::GridOperator<GridFunctionSpace, GridFunctionSpace, LocalOperator, MatrixBackend, DomainField, RangeType, RangeType, ConstraintsContainer, ConstraintsContainer>;
  GridOperator gridOperator(gridFunctionSpace, constraintsContainer, gridFunctionSpace, constraintsContainer, localOperator, matrixBackend);

  // Get grid function from Dirichlet boundary condition
  auto glambda = [&](const auto& e, const auto& x) { return problem.g(e, x); };
  auto boundaryCondition = Dune::PDELab::makeGridFunctionFromCallable(gv, glambda);
  Dune::PDELab::interpolate(boundaryCondition, gridFunctionSpace, coefficientVector);
  Dune::PDELab::set_nonconstrained_dofs(constraintsContainer, 0.0, coefficientVector);

  // Create Netwon solver
  using LinearSolver = TwoLevelSchwarzSolver<GridOperator::Jacobian, CoefficientVector>;
  LinearSolver linearSolver(gridFunctionSpace, constraintsContainer, ptree, "twolevelschwarz");
  using Solver = Dune::PDELab::NewtonMethod<GridOperator, LinearSolver>;
  Solver solver(gridOperator, linearSolver);

  // Use some nonsense parameters to ensure that setting them explicitly works
  auto& newton_ptree = ptree.sub("newton");
  if (not newton_ptree.hasKey("VerbosityLevel")) newton_ptree["VerbosityLevel"] = "4";
  if (helper.rank() != 0) newton_ptree["VerbosityLevel"] = "0";
  solver.setParameters(newton_ptree);

  // Retrieve the terminate interface and set parameters
  auto terminate = solver.getTerminate();
  terminate->setParameters(ptree.sub("Terminate"));

  // Retrieve line search interface and set parameters
  auto line_search = solver.getLineSearch();
  line_search->setParameters(ptree.sub("LineSearch"));

  // Solve PDE
  solver.apply(coefficientVector);

  // Visualization
  using VTKWriter = Dune::SubsamplingVTKWriter<GridView>;
  Dune::RefinementIntervals subint(1);
  VTKWriter vtkwriter(gv, subint);
  std::string vtkfile("testnewton");
  Dune::PDELab::addSolutionToVTKWriter(vtkwriter, gridFunctionSpace, coefficientVector, Dune::PDELab::vtk::defaultNameScheme());

  // Write MPI partitioning
  std::vector<int> rankVec(gv.size(0), helper.rank());
  Dune::P0VTKFunction rankFunc(gv, rankVec, "Rank");
  vtkwriter.addCellData(Dune::stackobject_to_shared_ptr(rankFunc));

  vtkwriter.write(vtkfile, Dune::VTK::ascii);

  Logger::get().report(MPI_COMM_WORLD);
}
