#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define DUNE_DDM_HAVE_LUA 1

#include "convection_diffusion_problems.hh"
#include "ddm_utilities.hh"
#include "generic_ddm_problem.hh"
#include "linearelasticity.hh"
#include "pdelab_schwarz.hh"
#include "poisson_problems.hh"
#include "problem_traits.hh"

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

template <class GridView, class Problem, class Params, class SymmParams = Params>
void driver(GridView gv, const Dune::MPIHelper& helper, const Dune::ParameterTree& ptree, std::shared_ptr<Params> params, std::shared_ptr<SymmParams> symm_params = nullptr)
{
  auto* total_event = Logger::get().registerEvent("Total", "Total time");
  auto* setup_grid_event = Logger::get().registerEvent("Total", "Setup grid");
  auto* setup_prec_event = Logger::get().registerEvent("Total", "Setup preconditioner");
  auto* solve_event = Logger::get().registerEvent("Total", "Linear solve");

  Logger::get().startEvent(total_event);
  Logger::get().startEvent(setup_grid_event);

  logger::info("MPI ranks: {}", helper.size());

  helper.getCommunication().barrier();
  Logger::get().endEvent(setup_grid_event);

  // Create problem classes, assemble and setup preconditioner
  Logger::get().startEvent(setup_prec_event);

  std::unique_ptr<Problem> problem;

  if constexpr (Problem::Traits::is_symmetric) problem = std::make_unique<Problem>(gv, helper, params);
  else problem = std::make_unique<Problem>(gv, helper, params, symm_params);

  // Create the two-level Schwarz preconditioner
  using Prec = TwoLevelSchwarzPreconditioner<Native<typename Problem::Vec>>;
  auto prec = std::make_shared<Prec>(*problem, ptree, helper);

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
  typename Prec::NativeVec v(native(problem->getX()));
  typename Prec::NativeVec b = problem->getD();

  v = 0;
  solver->apply(v, b, res);
  problem->getX() -= v;
  Logger::get().endEvent(solve_event);

  // First get the number of basis vectors from the debug rank
  auto n_basis = prec->get_basis().size();
  helper.getCommunication().broadcast(&n_basis, 1, ptree.get("debug_rank", 0));

  // Visualization
  if (ptree.get("visualise", true)) {
    Dune::SubsamplingVTKWriter writer(gv, Dune::refinementLevels(0));
    using DGF = std::conditional_t<Problem::Traits::vector_valued, Dune::PDELab::VectorDiscreteGridFunction<typename Problem::GFS, typename Problem::Vec>,
                                   Dune::PDELab::DiscreteGridFunction<typename Problem::GFS, typename Problem::Vec>>;
    using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGF>;

    // Storage to keep visualization data alive until writer.write() is called
    std::vector<std::shared_ptr<typename Prec::NativeVec>> vec_storage;
    std::vector<std::shared_ptr<typename Problem::Vec>> gf_storage;
    std::vector<std::shared_ptr<DGF>> dgf_storage;

    const auto write_overlapping_vector = [&](const auto& vec, const std::string& name, bool zero_if_not_debug_rank = true) {
      auto vec_vis = vec;
      if (zero_if_not_debug_rank and (helper.rank() != ptree.get("debug_rank", 0))) vec_vis = 0;
      if (zero_if_not_debug_rank) prec->getOverlappingCommunication()->addOwnerCopyToAll(vec_vis, vec_vis);

      auto vec_small = std::make_shared<typename Prec::NativeVec>(problem->getX().N());
      for (std::size_t i = 0; i < vec_small->N(); ++i) (*vec_small)[i] = vec_vis[i];
      auto gf = std::make_shared<typename Problem::Vec>(problem->getGFS());
      gf->attach(vec_small);
      auto dgf = std::make_shared<DGF>(problem->getGFS(), gf);
      writer.addVertexData(std::make_shared<VTKF>(dgf, name));

      // Keep the data alive
      vec_storage.push_back(vec_small);
      gf_storage.push_back(gf);
      dgf_storage.push_back(dgf);
    };

    // Write solution
    Dune::PDELab::addSolutionToVTKWriter(writer, *problem->getGFS(), problem->getXVec());

    // Write the rhs
    DGF dgfb(*problem->getGFS(), problem->getDVec());
    writer.addVertexData(std::make_shared<VTKF>(dgfb, "RHS"));

    // Write MPI partitioning
    std::vector<int> rankVec(gv.size(0), helper.rank());
    Dune::P0VTKFunction rankFunc(gv, rankVec, "Rank");
    writer.addCellData(Dune::stackobject_to_shared_ptr(rankFunc));

    // Write ring region (for ring coarse spaces)
    if (problem->get_neumann_region_to_subdomain().size() > 0) {
      typename Prec::NativeVec v(prec->getOverlappingCommunication()->indexSet().size());
      v = 0;
      for (const auto& idx : problem->get_neumann_region_to_subdomain()) v[idx] = 1;

      write_overlapping_vector(v, "Ring region");
    }

    // Visualise the basis vectors
    for (std::size_t i = 0; i < n_basis; ++i) {
      auto istr = std::to_string(i);
      auto i_padded = std::string(4 - std::min(4UL, istr.length()), '0') + istr;
      auto name = "Basis " + i_padded;

      typename Prec::NativeVec v(prec->getOverlappingCommunication()->indexSet().size());
      if (helper.rank() == ptree.get("debug_rank", 0)) v = prec->get_basis()[i];
      else v = 0;
      write_overlapping_vector(v, name);
    }

    // Write A_neu * 1 (should be zero)
    auto A_neu = problem->get_first_neumann_matrix();
    typename Prec::NativeVec ones(A_neu->N()), rowsums(A_neu->N());
    ones = 1;
    rowsums = 0;
    A_neu->mv(ones, rowsums);
    typename Prec::NativeVec v1(prec->getOverlappingCommunication()->indexSet().size());
    if (problem->get_neumann_region_to_subdomain().size() > 0)
      for (std::size_t i = 0; i < problem->get_neumann_region_to_subdomain().size(); ++i) v1[problem->get_neumann_region_to_subdomain()[i]] = rowsums[i];
    write_overlapping_vector(v1, "A_neu * 1");

    // Write POU
    write_overlapping_vector(prec->get_pou()->vector(), "POU");

    writer.write(ptree.get("filename", "pdelab_example"), Dune::VTK::appendedraw);
  }

  Logger::get().endEvent(total_event);
  if (ptree.get("view_report", true)) Logger::get().report(helper.getCommunicator());
}

int main(int argc, char* argv[])
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    setup_loggers(helper.rank(), argc, argv);

    Dune::ParameterTree configptree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readOptions(argc, argv, configptree);
    if (!configptree.hasKey("problem") or !configptree.hasKey("ini_file")) {
      logger::error("Usage: {} -problem <problem> -ini_file <filename.ini>", argv[0]);
      logger::error("  Supported problems:");
      logger::error("  - Convecton diffusion (-problem convection_diffusion)");
      logger::error("  - Poisson (-problem poisson)");
      helper.getCommunication().barrier();
      return 1;
    }

    // Convection diffusion example
    if (configptree.get<std::string>("problem") == "convection_diffusion") {
      logger::info("################################################################");
      logger::info("##########    Running convection diffusion example    ##########");
      logger::info("################################################################");
      helper.getCommunication().barrier();

      // Read parameters and create grid
      Dune::ParameterTree ptree;
      Dune::ParameterTreeParser ptreeparser;
      ptreeparser.readINITree(configptree.get<std::string>("ini_file"), ptree);
      ptreeparser.readOptions(argc, argv, ptree);

      // Create grid
      using Grid = Dune::UGGrid<2>;
      // using Grid = Dune::YaspGrid<2>;
      auto grid = DDMUtilities::make_grid<Grid>(ptree, helper, "");
      auto gv = grid->leafGridView();
      using GridView = decltype(gv);

      using ProblemParams = LuaConvectionDiffusionProblem<GridView, double>;
      using SymmetricProblemParams = LuaConvectionDiffusionProblem<GridView, double, true>; // true means symmetrise the problem (i.e. ignore convection)
      static constexpr bool is_symmetric = false;
      static constexpr bool use_dg = true;
      using Traits = ConvectionDiffusionTraits<GridView, ProblemParams, SymmetricProblemParams, is_symmetric, use_dg>;
      using Problem = GenericDDMProblem<GridView, Traits>;

      auto params = std::make_shared<ProblemParams>("convection_diffusion_coefficient.lua");
      auto symm_params = std::make_shared<SymmetricProblemParams>("convection_diffusion_coefficient.lua");

      driver<GridView, Problem, ProblemParams, SymmetricProblemParams>(gv, helper, ptree, params, symm_params);
    }
    else if (configptree.get<std::string>("problem") == "poisson") {
      logger::info("################################################################");
      logger::info("##########          Running Poisson example           ##########");
      logger::info("################################################################");
      helper.getCommunication().barrier();

      // Read parameters and create grid
      Dune::ParameterTree ptree;
      Dune::ParameterTreeParser ptreeparser;
      ptreeparser.readINITree(configptree.get<std::string>("ini_file"), ptree);
      ptreeparser.readOptions(argc, argv, ptree);

      // Create grid
      using Grid = Dune::YaspGrid<2>;
      auto grid = DDMUtilities::make_grid<Grid>(ptree, helper, "");
      auto gv = grid->leafGridView();
      using GridView = decltype(gv);

      using ProblemParams = LuaProblem<GridView, double>;
      static constexpr bool is_symmetric = true;
      static constexpr bool use_dg = false;
      using Traits = ConvectionDiffusionTraits<GridView, ProblemParams>;
      using Problem = GenericDDMProblem<GridView, Traits>;

      auto params = std::make_shared<ProblemParams>("poisson_coefficient.lua");

      driver<GridView, Problem, ProblemParams, ProblemParams>(gv, helper, ptree, params);
    }
    else if (configptree.get<std::string>("problem") == "linearelasticity") {
      logger::info("################################################################");
      logger::info("##########     Running linear elasticity example      ##########");
      logger::info("################################################################");
      helper.getCommunication().barrier();

      // Read parameters and create grid
      Dune::ParameterTree ptree;
      Dune::ParameterTreeParser ptreeparser;
      ptreeparser.readINITree(configptree.get<std::string>("ini_file"), ptree);
      ptreeparser.readOptions(argc, argv, ptree);

      // Create grid
      using Grid = Dune::UGGrid<3>;
      const int gridsize = 3;
      auto grid = Dune::StructuredGridFactory<Grid>::createSimplexGrid({0, 0, 0}, {10, 1, 1.5}, {10 * gridsize, gridsize, (unsigned int)(1.5 * gridsize)});

      auto gv = grid->leafGridView();
      auto part = Dune::ParMetisGridPartitioner<decltype(gv)>::partition(gv, helper);
      grid->loadBalance(part, 0);
      grid->globalRefine(ptree.get("refine", 0));
      using GridView = decltype(gv);

      using ProblemParams = LinearElasticityParametersLua<GridView>;
      using Traits = LinearElasticityTraits<GridView, ProblemParams>;
      using Problem = GenericDDMProblem<GridView, Traits>;

      auto params = std::make_shared<ProblemParams>();

      driver<GridView, Problem, ProblemParams, ProblemParams>(gv, helper, ptree, params);
    }
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
