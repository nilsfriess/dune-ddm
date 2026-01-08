#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dune/ddm/coarsespaces/coarse_spaces.hh"
#include "dune/ddm/combined_preconditioner.hh"
#include "dune/ddm/galerkin_preconditioner.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/nonoverlapping_operator.hh"
#include "dune/ddm/overlap_extension.hh"
#include "dune/ddm/pou.hh"
#include "dune/ddm/schwarz.hh"
#include "examples/pdelab_helper.hh"
#include "linearelasticity.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/istl/io.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspaceutilities.hh>
#include <dune/pdelab/localoperator/linearelasticityparameter.hh>
#include <taskflow/core/taskflow.hpp>

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  setup_loggers(helper.rank(), argc, argv);

  // Read parameters from ini file and command line
  Dune::ParameterTree ptree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readINITree("linearelasticity.ini", ptree);
  ptreeparser.readOptions(argc, argv, ptree);

  // Create the grid
  using Grid = Dune::UGGrid<3>;
  const int gridsize = 8;
  auto grid = Dune::StructuredGridFactory<Grid>::createSimplexGrid({0, 0, 0}, {10, 1, 1.5}, {10 * gridsize, gridsize, (unsigned int)(1.5 * gridsize)});

  auto gv = grid->leafGridView();
  auto part = Dune::ParMetisGridPartitioner<decltype(gv)>::partition(gv, helper);
  grid->loadBalance(part, 0);
  grid->globalRefine(ptree.get("grid.refine", 0));
  logger::info("Created grid with {} elements", gv.size(0));

  // Create the problem class (which contains the finite element information and performs the assembly later)
  LinearElasticityProblem problem(gv, helper);
  using Mat = decltype(problem)::NativeMat;
  using Vec = decltype(problem)::NativeVec;

  // Using the PDELab grid function space, create a globally unqiue numbering of the dofs
  const int overlap = 1;
  auto novlp_comm = make_communication(*problem.gfs());
  auto [ovlp_comm, boundary_mask] = make_overlapping_communication(*novlp_comm, problem.A(), overlap);
  using Communication = std::decay_t<decltype(*ovlp_comm)>;

  // Assemble the overlapping Dirichlet and Neumann matrices
  auto [A_dir, A_neu, B_neu, dirichlet_mask] = problem.assemble_overlapping_matrices(*ovlp_comm, overlap);

  // Create fine-level Schwarz preconditioner
  auto pou = std::make_shared<PartitionOfUnity>(*A_dir, *ovlp_comm, ptree, overlap);
  auto schwarz = std::make_shared<SchwarzPreconditioner<Mat, Vec, Communication>>(A_dir, ovlp_comm, pou, ptree);

  // Compute coarse space basis functions
  tf::Taskflow taskflow("Main taskflow");
  auto coarse_space = std::make_unique<GenEOCoarseSpace<Mat>>(A_neu, B_neu, pou, ptree, taskflow);

  // Create coarse level solver
  using CoarseLevel = GalerkinPreconditioner<Vec, Communication>;
  const auto zero_at_dirichlet = [&](auto&& x) {
    for (std::size_t i = 0; i < x.size(); ++i)
      if ((*dirichlet_mask)[i] > 0) x[i] = 0;
  };

  tf::Task prec_setup_task;
  std::shared_ptr<CoarseLevel> coarse;
  prec_setup_task = taskflow
                        .emplace([&]() {
                          auto basis = coarse_space->get_basis();
                          std::ranges::for_each(basis, zero_at_dirichlet);
                          coarse = std::make_shared<CoarseLevel>(*A_dir, basis, ovlp_comm, ptree, "coarse_solver");
                        })
                        .name("Build coarse preconditioner")
                        .succeed(coarse_space->get_setup_task());

  // Now actually run the tasks
  tf::Executor executor(1);
  executor.run(taskflow).get();

  // Build the parallel operator and solver objects
  using Op = NonOverlappingOperator<Mat, Vec, Vec, Communication>;
  Dune::initSolverFactories<Op>(); // register all DUNE solvers so we can choose them via the command line

  auto prec = std::make_shared<CombinedPreconditioner<Vec>>(ptree);
  auto op = std::make_shared<Op>(problem.A_ptr(), novlp_comm);

  prec->set_op(op);
  prec->add(schwarz);
  prec->add(coarse);

  auto solver_subtree = ptree.sub("solver");
  solver_subtree["verbose"] = helper.rank() == 0 ? solver_subtree["verbose"] : "0";

  auto solver = Dune::getSolverFromFactory(op, solver_subtree, prec);

  Dune::InverseOperatorResult res;
  auto& x = problem.x();
  Vec v(x.N());
  Vec b(problem.r());

  v = 0;
  solver->apply(v, b, res);
  x -= v;

  // Visualisation
  auto debug_rank = ptree.get("debug_rank", 0);
  using P = decltype(problem);
  using DGF = Dune::PDELab::VectorDiscreteGridFunction<P::GFS, P::Vec>;
  using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGF>;
  using PVec = P::Vec;
  Dune::SubsamplingVTKWriter writer(gv, Dune::refinementLevels(0));

  std::vector<std::shared_ptr<PVec>> pvecs;
  const auto write_overlapping_vector = [&](const auto& vec, const std::string& name) {
    auto vec_vis = vec;
    if (helper.rank() != debug_rank) vec_vis = 0;
    ovlp_comm->addOwnerCopyToAll(vec_vis, vec_vis);

    auto vec_small = std::make_shared<Vec>(problem.x().N());
    for (std::size_t i = 0; i < vec_small->N(); ++i) (*vec_small)[i] = vec_vis[i];
    auto gf = std::make_shared<PVec>(problem.gfs());
    pvecs.push_back(gf);
    gf->attach(vec_small);
    auto dgf = std::make_shared<DGF>(problem.gfs(), gf);
    writer.addVertexData(std::make_shared<VTKF>(dgf, name));
  };

  std::vector<int> rankVec(gv.size(0), helper.rank());
  Dune::P0VTKFunction rankFunc(gv, rankVec, "Rank");
  writer.addCellData(Dune::stackobject_to_shared_ptr(rankFunc));

  problem.gfs()->name("Solution");
  Dune::PDELab::addSolutionToVTKWriter(writer, *problem.gfs(), problem.x_pdelab());

  using Params = P::Parameters<decltype(gv)>;
  Params params;
  using ParamsVTKGF = Dune::PDELab::VTKGridFunctionAdapter<Params>;
  writer.addCellData(std::make_shared<ParamsVTKGF>(params, "Lambda"));

  // Write the overlapping subdomain
  Vec ovlp_subdomain(ovlp_comm->indexSet().size());
  ovlp_subdomain = helper.rank() == debug_rank ? 1 : 0;
  write_overlapping_vector(ovlp_subdomain, "Ovlp. subdomain");

  writer.write("linearelasticity");
}
