#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "examples/ddm_utilities.hh"
#include "examples/generic_ddm_problem.hh"
#include "examples/pdelab_schwarz.hh"
#include "examples/poisson_problems.hh"
#include "examples/problem_traits.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/ddm/logger.hh>
#include <dune/ddm/nonoverlapping_operator.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/solvers.hh>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#include <dune/pdelab.hh>
#pragma GCC diagnostic pop

using Dune::PDELab::Backend::native;
using Dune::PDELab::Backend::Native;

int main(int argc, char* argv[])
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    setup_loggers(helper.rank(), argc, argv);

    auto* total_event = Logger::get().registerEvent("Total", "Total time");
    auto* setup_grid_event = Logger::get().registerEvent("Total", "Setup grid");
    auto* setup_prec_event = Logger::get().registerEvent("Total", "Setup preconditioner");
    auto* solve_event = Logger::get().registerEvent("Total", "Linear solve");

    Logger::get().startEvent(total_event);
    Logger::get().startEvent(setup_grid_event);

    // Read parameters and create grid
    Dune::ParameterTree ptree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readINITree("pdelab_example.ini", ptree);
    ptreeparser.readOptions(argc, argv, ptree);

    logger::info("MPI ranks: {}", helper.size());

    using Grid = Dune::UGGrid<2>;
    auto grid = DDMUtilities::make_grid<Grid>(ptree, helper, "");
    auto gv = grid->leafGridView();
    using GridView = decltype(gv);

    helper.getCommunication().barrier();
    Logger::get().endEvent(setup_grid_event);

    // Create problem classes, assemble and setup preconditioner
    Logger::get().startEvent(setup_prec_event);

    using ProblemParams = IslandsProblem<GridView, double>;
    using Traits = ConvectionDiffusionTraits<GridView, ProblemParams, true>;
    using Problem = GenericDDMProblem<GridView, Traits>;

    Problem problem(gv, helper);

    // Create the two-level Schwarz preconditioner
    using Prec = TwoLevelSchwarzPreconditioner<Native<Problem::Vec>>;
    auto prec = std::make_shared<Prec>(problem, ptree, helper);

    // Get the non-overlapping operator from the preconditioner
    auto op = prec->getNonOverlappingOperator();

    // Set up the solver
    using Op = Prec::NonOverlappingOp;
    Dune::initSolverFactories<Op>();
    auto solver_subtree = ptree.sub("solver");
    solver_subtree["verbose"] = helper.rank() == 0 ? solver_subtree["verbose"] : "0";

    auto solver = Dune::getSolverFromFactory(op, solver_subtree, prec);

    helper.getCommunication().barrier();
    Logger::get().endEvent(setup_prec_event);

    // Solve the system
    Logger::get().startEvent(solve_event);
    Dune::InverseOperatorResult res;
    Prec::NativeVec v(native(problem.getX()));
    Prec::NativeVec b = problem.getD();

    v = 0;
    solver->apply(v, b, res);
    problem.getX() -= v;
    Logger::get().endEvent(solve_event);

    // Visualization
    if (ptree.get("visualise", true)) {
      Dune::SubsamplingVTKWriter writer(gv, Dune::refinementLevels(0));

      using P = decltype(problem);
      using DGF = Dune::PDELab::DiscreteGridFunction<typename P::GFS, typename P::Vec>;
      using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGF>;

      // Write solution
      Dune::PDELab::addSolutionToVTKWriter(writer, *problem.getGFS(), problem.getXVec());

      // Write the rhs
      DGF dgfb(*problem.getGFS(), problem.getDVec());
      writer.addVertexData(std::make_shared<VTKF>(dgfb, "RHS"));

      // Write MPI partitioning
      std::vector<int> rankVec(gv.size(0), helper.rank());
      Dune::P0VTKFunction rankFunc(gv, rankVec, "Rank");
      writer.addCellData(Dune::stackobject_to_shared_ptr(rankFunc));

      writer.write(ptree.get("filename", "pdelab_example"), Dune::VTK::appendedraw);
    }

    Logger::get().endEvent(total_event);
    if (ptree.get("view_report", true)) Logger::get().report(helper.getCommunicator());
  }
  catch (Dune::Exception& e) {
    std::cerr << "Error in DUNE: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 1;
  }

  return 0;
}
