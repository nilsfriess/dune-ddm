#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "convectiondiffusiondg.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/twolevel_schwarz.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/geometry/type.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/istl/solvercategory.hh>
#include <dune/pdelab/boilerplate/pdelab.hh>
#include <dune/pdelab/localoperator/convectiondiffusiondg.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>
#include <dune/pdelab/stationary/linearproblem.hh>

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  setup_loggers(helper.rank(), argc, argv);

  // Create parameter tree
  Dune::ParameterTree ptree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readINITree("convectiondiffusiondg.ini", ptree);
  ptreeparser.readOptions(argc, argv, ptree);

  // Create grid
  constexpr auto dim = 2;
  using Grid = Dune::UGGrid<dim>;
  unsigned int gridsize = ptree.get("grid.global_size", 32);
  if (ptree.hasKey("grid.size_per_rank")) {
    auto grid_sqrt = static_cast<int>(std::pow(helper.size(), 1. / dim));
    gridsize = ptree.get<int>("grid.size_per_rank") * grid_sqrt;
  }

  auto grid = Dune::StructuredGridFactory<Grid>::createCubeGrid({0, 0}, {1, 1}, {gridsize, gridsize});

  using GridView = typename Grid::LeafGridView;
  auto gv = grid->leafGridView();
  auto part = Dune::ParMetisGridPartitioner<GridView>::partition(gv, helper);
  grid->loadBalance(part, 0);
  grid->globalRefine(ptree.get("grid.refine", 0));

  using Problem = ConvectionDiffusionDGProblem<GridView, double>;
  Problem problem;

  constexpr auto solvertype = Dune::SolverCategory::nonoverlapping;
  using DGSpace = Dune::PDELab::DGLegendreSpace<Grid, double, 1, Dune::GeometryType::cube, solvertype>;
  DGSpace space(gv);

  DGSpace::DOF x(space.getGFS(), 0);

  using BCType = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<Problem>;
  BCType bctype(gv, problem);

  using G = Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter<Problem>;
  G g(gv, problem);
  // Dune::PDELab::interpolate(g, space.getGFS(), x);
  space.assembleConstraints(bctype);
  // space.setNonConstrainedDOFS(x, 0.0);

  using LOP = Dune::PDELab::ConvectionDiffusionDG<Problem, DGSpace::FEM>;
  LOP lop(problem);
  using Assembler = Dune::PDELab::GalerkinGlobalAssembler<DGSpace, LOP, solvertype>;
  Assembler assembler(space, lop, 27);

#if 1
  using LinearSolver = TwoLevelSchwarzSolver<typename Assembler::MAT, DGSpace::DOF>;
  LinearSolver ls(space.getGFS(), space.getCC(), ptree, "twolevelschwarz", false);
  using SLP = Dune::PDELab::StationaryLinearProblemSolver<typename Assembler::GO, LinearSolver, typename DGSpace::DOF>;
  SLP slp(*assembler, ls, x, 1e-5);
#else
  using LinearSolver = typename Dune::PDELab::ISTLSolverBackend_IterativeDefault<DGSpace, Assembler, solvertype>;
  using SLP = Dune::PDELab::StationaryLinearProblemSolver<typename Assembler::GO, typename LinearSolver::LS, typename DGSpace::DOF>;
  LinearSolver ls(space, assembler, 100, 4);
  SLP slp(*assembler, *ls, x, 1e-5);
#endif

  slp.apply();

  Dune::SubsamplingVTKWriter<GridView> vtkwriter(gv, Dune::refinementIntervals(1));
  typename DGSpace::DGF xdgf(space.getGFS(), x);
  vtkwriter.addVertexData(std::make_shared<typename DGSpace::VTKF>(xdgf, "solution"));

  // Write MPI partitioning
  std::vector<int> rankVec(grid->leafGridView().size(0), grid->leafGridView().comm().rank());
  Dune::P0VTKFunction rankFunc(grid->leafGridView(), rankVec, "Rank");
  vtkwriter.addCellData(Dune::stackobject_to_shared_ptr(rankFunc));

  vtkwriter.write("convectiondifusiondg", Dune::VTK::appendedraw);
}
