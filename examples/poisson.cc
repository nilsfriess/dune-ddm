#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::int64_t

#define USE_UGGRID 1 // Set to zero to use YASPGrid
#define GRID_DIM 2

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <dune/common/densevector.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/iteratorfacades.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/mpicommunication.hh>
#include <dune/common/parallel/plocalindex.hh>
#include <dune/common/parallel/remoteindices.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>
#include <dune/grid/common/geometry.hh>
#include <dune/grid/common/gridview.hh>
#include <dune/grid/io/file/vtk/function.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/istl/basearray.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solvercategory.hh>
#include <dune/istl/solvers.hh>
#include <dune/pdelab/backend/istl/vector.hh>
#include <dune/pdelab/common/function.hh>
#include <dune/pdelab/common/multiindex.hh>
#include <dune/pdelab/common/vtkexport.hh>
#include <dune/pdelab/gridfunctionspace/vtk.hh>
#include <dune/pdelab/localoperator/permeability_adapter.hh>
#include <dune/pdelab/ordering/transformations.hh>
#include <iostream>
#include <mpi.h>
#include <string>

#if USE_UGGRID
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/uggrid.hh>
#else
#include <dune/grid/yaspgrid.hh>
#include <dune/grid/yaspgrid/coordinates.hh>
#endif

#include "pdelab_helper.hh"
#include "poisson.hh"

#include <dune/ddm/coarsespaces/coarse_spaces.hh>
#include <dune/ddm/combined_preconditioner.hh>
#include <dune/ddm/datahandles.hh>
#include <dune/ddm/galerkin_preconditioner.hh>
#include <dune/ddm/helpers.hh>
#include <dune/ddm/logger.hh>
#include <dune/ddm/nonoverlapping_operator.hh>
#include <dune/ddm/overlap_extension.hh>
#include <dune/ddm/pou.hh>
#include <dune/ddm/schwarz.hh>

#if DUNE_DDM_HAVE_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif

namespace {
auto make_grid(const Dune::ParameterTree& ptree, [[maybe_unused]] const Dune::MPIHelper& helper)
{
  auto* event = Logger::get().registerEvent("Grid", "create");
  Logger::ScopedLog sl(event);

#if USE_UGGRID
  using Grid = Dune::UGGrid<GRID_DIM>;
  std::unique_ptr<Grid> grid;
  if (ptree.hasKey("meshfile")) {
    logger::info("Loading mesh from file");
    const auto meshfile = ptree.get("meshfile", "../data/unitsquare.msh");
    const auto verbose = ptree.get("verbose", 0);

    grid = Dune::GmshReader<Grid>::read(meshfile, verbose > 2);
  }
  else {
    auto gridsize = static_cast<unsigned int>(ptree.get("gridsize", 32));
    if (ptree.hasKey("gridsize_per_rank")) {
      auto grid_sqrt = static_cast<int>(std::pow(helper.size(), 1. / GRID_DIM)); // This is okay because Yaspgrid will complain if the gridsize is not a power of GRID_DIM
      gridsize = ptree.get<int>("gridsize_per_rank") * grid_sqrt;
    }
    grid = Dune::StructuredGridFactory<Grid>::createSimplexGrid({0, 0}, {1, 1}, {gridsize, gridsize});
  }
#else
  using Grid = Dune::YaspGrid<GRID_DIM>;
  auto gridsize = ptree.get("gridsize", 32);
  if (ptree.hasKey("gridsize_per_rank")) {
    auto grid_sqrt = static_cast<int>(std::pow(helper.size(), 1. / GRID_DIM)); // This is okay because Yaspgrid will complain if the gridsize is not a power of GRID_DIM
    gridsize = ptree.get<int>("gridsize_per_rank") * grid_sqrt;
  }
#if GRID_DIM == 2
  Dune::Yasp::PowerDPartitioning<GRID_DIM> partitioner;
  auto grid = std::unique_ptr<Grid>(new Grid({1.0, 1.0}, {gridsize, gridsize}, std::bitset<2>(0ULL), 0, Grid::Communication(), &partitioner));
#elif GRID_DIM == 3
  Dune::Yasp::PowerDPartitioning<GRID_DIM> partitioner;
  auto grid = std::unique_ptr<Grid>(new Grid({1.0, 1.0, 1.0}, {gridsize, gridsize, gridsize}, std::bitset<3>(0ULL), 0, Grid::Communication(), &partitioner));
#endif
#endif

  grid->globalRefine(ptree.get("serial_refine", 0));

#if USE_UGGRID
  auto gv = grid->leafGridView();
  auto part = Dune::ParMetisGridPartitioner<decltype(gv)>::partition(gv, helper);

  grid->loadBalance(part, 0);
#else
  grid->loadBalance();
#endif

  grid->globalRefine(ptree.get("refine", 0));

  return grid;
}

template <class Vec, class RemoteIndices>
bool is_pou(const Vec& pou, const RemoteIndices& remote_indices)
{
  AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  Dune::Interface all_all_interface;
  all_all_interface.build(remote_indices, allAttributes, allAttributes);
  Dune::VariableSizeCommunicator all_all_comm(all_all_interface);
  AddVectorDataHandle<Vec> advdh;
  Vec v = pou;
  advdh.setVec(v);
  all_all_comm.forward(advdh);
  return std::all_of(v.begin(), v.end(), [](auto val) { return std::abs(val - 1.0) < 1e10; });
}
} // namespace

int main(int argc, char* argv[])
{
  try {
    using Dune::PDELab::Backend::native;
    using Dune::PDELab::Backend::Native;

    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    setup_loggers(helper.rank(), argc, argv);

    // Create the taskflow that will hold all the tasks
    tf::Taskflow taskflow("Main taskflow");

    auto* matrix_setup = Logger::get().registerEvent("Total", "Setup problem");
    auto* prec_setup = Logger::get().registerEvent("Total", "Setup preconditioner");
    auto* solve = Logger::get().registerEvent("Total", "Linear solve");
    auto* total = Logger::get().registerEvent("Total", "Total time");
    Logger::get().startEvent(total);

    Logger::get().startEvent(matrix_setup);
    Dune::ParameterTree ptree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readINITree("poisson.ini", ptree);
    ptreeparser.readOptions(argc, argv, ptree);

    // Create the grid view
    auto grid = make_grid(ptree, helper);
    auto gv = grid->leafGridView();

    // Set up the finite element problem. This also assembles the sparsity pattern of the matrix.
    PoissonProblem problem(gv, helper);
    using Vec = decltype(problem)::Vec;
    using Mat = decltype(problem)::Mat;

    // Get the overlap from the config file
    const int overlap = ptree.get("overlap", 1);

    // Using the sparsity pattern of the matrix, create the overlapping subdomains by extending this index set
    auto remoteids = make_remote_indices(*problem.getGFS(), helper);
    using RemoteIndices = std::remove_cvref_t<decltype(*remoteids)>;
    ExtendedRemoteIndices ext_indices(*remoteids, native(problem.getA()), overlap);

    // The nonzero pattern of the non-overlapping matrix is now set up, and we have a parallel index set
    // on the overlapping subdomains. Now we can assemble the overlapping matrices.
    const auto& coarsespace_subtree = ptree.sub("coarsespace");
    auto coarsespace = coarsespace_subtree.get("type", "geneo");
    if (coarsespace == "geneo" or coarsespace == "constraint_geneo") problem.assemble_overlapping_matrices(ext_indices, NeumannRegion::All, NeumannRegion::All, true);
    else if (coarsespace == "msgfem") problem.assemble_overlapping_matrices(ext_indices, NeumannRegion::All, NeumannRegion::All, true);
    else if (coarsespace == "pou" or coarsespace == "harmonic_extension" or coarsespace == "algebraic_geneo" or coarsespace == "algebraic_msgfem" or coarsespace == "none")
      problem.assemble_dirichlet_matrix_only(ext_indices);
    else if (coarsespace == "geneo_ring") problem.assemble_overlapping_matrices(ext_indices, NeumannRegion::ExtendedOverlap, NeumannRegion::ExtendedOverlap, false);
    else if (coarsespace == "msgfem_ring") problem.assemble_overlapping_matrices(ext_indices, NeumannRegion::Overlap, NeumannRegion::Overlap, false);
    else DUNE_THROW(Dune::NotImplemented, "Unknown coarse space");

    // Extract the three matrices that have been assembled
    auto A_dir = problem.get_dirichlet_matrix();
    auto A_neu = problem.get_first_neumann_matrix();
    auto B_neu = problem.get_second_neumann_matrix();

    MPI_Barrier(MPI_COMM_WORLD);
    Logger::get().endEvent(matrix_setup);

    // Next, create a partition of unity
    auto pou = std::make_shared<PartitionOfUnity>(*A_dir, ext_indices, ptree);
    if (!is_pou(pou->vector(), ext_indices.get_remote_indices())) logger::warn("POU does not add up to 1");
    else logger::debug("POU adds up to 1");

    // Now we can create the preconditioner. First the fine level overlapping Schwarz method
    logger::info("Setting up tasks");
    auto schwarz = std::make_shared<SchwarzPreconditioner<Native<Vec>, Native<Mat>>>(A_dir, ext_indices.get_remote_indices(), pou, ptree);
    logger::info("After schwarz");

    using CoarseLevel = GalerkinPreconditioner<Native<Vec>>;
    std::shared_ptr<CoarseLevel> coarse;

    const auto zero_at_dirichlet = [&](auto&& x) {
      for (std::size_t i = 0; i < x.size(); ++i)
        if (problem.get_overlapping_dirichlet_mask()[i] > 0) x[i] = 0;
    };

    std::unique_ptr<CoarseSpaceBuilder<>> coarse_space = nullptr;

    if (coarsespace == "geneo") { coarse_space = std::make_unique<GenEOCoarseSpace<Native<Mat>>>(A_neu, B_neu, pou, ptree, taskflow); }
    else if (coarsespace == "constraint_geneo") {
      coarse_space = std::make_unique<ConstraintGenEOCoarseSpace<Native<Mat>, std::remove_reference_t<decltype(ext_indices.get_overlapping_boundary_mask())>>>(
          A_dir, A_neu, B_neu, pou, ext_indices.get_overlapping_boundary_mask(), ptree, taskflow);
    }
    else if (coarsespace == "geneo_ring") {
      coarse_space = std::make_unique<GenEORingCoarseSpace<Native<Mat>>>(A_dir, A_neu, pou, problem.get_neumann_region_to_subdomain(), ptree, taskflow);
    }
    else if (coarsespace == "msgfem") {
      // We only pass one matrix here, because the oversampling is simulated using the partition of unity
      coarse_space = std::make_unique<
          MsGFEMCoarseSpace<Native<Mat>, std::remove_reference_t<decltype(problem.get_overlapping_dirichlet_mask())>, std::remove_reference_t<decltype(ext_indices.get_overlapping_boundary_mask())>>>(
          A_neu, pou, problem.get_overlapping_dirichlet_mask(), ext_indices.get_overlapping_boundary_mask(), ptree, taskflow);
    }
    else if (coarsespace == "msgfem_ring") {
      coarse_space = std::make_unique<MsGFEMRingCoarseSpace<Native<Mat>, std::remove_reference_t<decltype(problem.get_overlapping_dirichlet_mask())>,
                                                            std::remove_reference_t<decltype(ext_indices.get_overlapping_boundary_mask())>>>(
          A_dir, A_neu, overlap, pou, problem.get_overlapping_dirichlet_mask(), ext_indices.get_overlapping_boundary_mask(), problem.get_neumann_region_to_subdomain(), ptree, taskflow);
    }
    else if (coarsespace == "algebraic_geneo") {
      coarse_space =
          std::make_unique<AlgebraicGenEOCoarseSpace<Native<Mat>>>(problem.getA().storage(), A_dir, pou, problem.get_overlapping_dirichlet_mask(), ext_indices, ptree, taskflow);
    }
    else if (coarsespace == "algebraic_msgfem") {
      coarse_space = std::make_unique<AlgebraicMsGFEMCoarseSpace<Native<Mat>, std::remove_reference_t<decltype(problem.get_overlapping_dirichlet_mask())>,
                                                                 std::remove_reference_t<decltype(ext_indices.get_overlapping_boundary_mask())>>>(
          problem.getA().storage(), A_dir, pou, problem.get_overlapping_dirichlet_mask(), ext_indices.get_overlapping_boundary_mask(), ext_indices, ptree, taskflow);
    }
    else if (coarsespace == "pou") {
      coarse_space = std::make_unique<POUCoarseSpace<>>(pou, taskflow);
    }
    else if (coarsespace == "none") {
      // Nothing to do here
    }
    else {
      DUNE_THROW(Dune::NotImplemented, "Unknown coarse space");
    }

    tf::Task prec_setup_task;
    std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> basis;
    if (coarse_space) {
      prec_setup_task = taskflow.emplace([&]() {
        basis = coarse_space->get_basis();
        std::ranges::for_each(basis, zero_at_dirichlet);
        coarse = std::make_shared<CoarseLevel>(*A_dir, basis, ext_indices.get_remote_indices(), ptree, "coarse_solver");
      });

      prec_setup_task.name("Build coarse preconditioner");
      prec_setup_task.succeed(coarse_space->get_setup_task());
    }

    Logger::get().startEvent(prec_setup);
    logger::info("Starting taskflow execution");
    tf::Executor executor(ptree.get("taskflow_executor_threads", 2));
    std::shared_ptr<tf::TFProfObserver> observer;
    if (helper.rank() == 0) observer = executor.make_observer<tf::TFProfObserver>();
    executor.run(taskflow).get();

    if (helper.rank() == 0) {
      std::ofstream dot_file("poisson.dot");
      taskflow.dump(dot_file);

      std::ofstream json_file("poisson.json");
      observer->dump(json_file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    Logger::get().endEvent(prec_setup);

    // Build the parallel operator
    using Op = NonOverlappingOperator<Native<Mat>, Native<Vec>>;
    Dune::initSolverFactories<Op>(); // register all DUNE solvers so we can choose them via the command line

    auto prec = std::make_shared<CombinedPreconditioner<Native<Vec>>>(ptree);
    auto op = std::make_shared<Op>(problem.getA().storage(), *remoteids);

    prec->set_op(op);
    prec->add(schwarz);
    if (coarse) prec->add(coarse);

    auto solver_subtree = ptree.sub("solver");
    solver_subtree["verbose"] = helper.rank() == 0 ? solver_subtree["verbose"] : "0";

    auto solver = Dune::getSolverFromFactory(op, solver_subtree, prec);

    Logger::get().startEvent(solve);
    Dune::InverseOperatorResult res;
    Native<Vec> v(native(problem.getX()));
    Native<Vec> b = problem.getD();

    v = 0;
    solver->apply(v, b, res);
    problem.getX() -= v;
    Logger::get().endEvent(solve);

    // Visualisation
    if (ptree.get("visualise", true)) {
      Dune::SubsamplingVTKWriter writer(gv, Dune::refinementLevels(0));

      using P = decltype(problem);
      using DGF = Dune::PDELab::DiscreteGridFunction<P::GFS, P::Vec>;
      using VTKF = Dune::PDELab::VTKGridFunctionAdapter<DGF>;

      auto debug_rank = ptree.get("debug_rank", 0);
      if (debug_rank > helper.size() - 1) debug_rank = 0;

      const auto write_overlapping_vector = [&](const auto& vec, const std::string& name) {
        AddVectorDataHandle<Native<Vec>> advdh;
        auto vec_vis = vec;
        if (helper.rank() != debug_rank) vec_vis = 0;
        advdh.setVec(vec_vis);
        ext_indices.get_overlapping_communicator().forward(advdh);

        auto vec_small = std::make_shared<Native<Vec>>(problem.getX().N());
        for (std::size_t i = 0; i < vec_small->N(); ++i) (*vec_small)[i] = vec_vis[i];
        auto gf = std::make_shared<P::Vec>(problem.getGFS());
        gf->attach(vec_small);
        auto dgf = std::make_shared<DGF>(problem.getGFS(), gf);
        writer.addVertexData(std::make_shared<VTKF>(dgf, name));
      };

      // Write solution
      Dune::PDELab::addSolutionToVTKWriter(writer, *problem.getGFS(), problem.getXVec());

      // Write the rhs
      DGF dgfb(*problem.getGFS(), problem.getDVec());
      writer.addVertexData(std::make_shared<VTKF>(dgfb, "RHS"));

      // Write MPI partitioning
      std::vector<int> rankVec(gv.size(0), helper.rank());
      Dune::P0VTKFunction rankFunc(gv, rankVec, "Rank");
      writer.addCellData(Dune::stackobject_to_shared_ptr(rankFunc));

      // Write some additional output
      write_overlapping_vector(pou->vector(), "POU");

      Native<Vec> ovlp_subdomain(ext_indices.size());
      ovlp_subdomain = helper.rank() == debug_rank ? 1 : 0;
      write_overlapping_vector(ovlp_subdomain, "Ovlp. subdomain");

      // Write coarse space basis vectors. We need to take into account that the ranks might have a different number of basis vectors
      auto n_basis = basis.size();
      auto bcast_root = debug_rank;
      MPI_Bcast(&n_basis, 1, MPI_UNSIGNED_LONG, bcast_root, helper.getCommunicator());
      for (std::size_t k = 0; k < n_basis; ++k) {
        auto bvec_vis = basis[0];
        if (helper.rank() != bcast_root) bvec_vis = 0;
        else bvec_vis = basis[k];
        write_overlapping_vector(bvec_vis, "Basis vec " + std::format("{:04}", k));
      }

      // Write ring region (might be all zero for non-ring coarse spaces)
      Native<Vec> ring_region(ext_indices.size());
      ring_region = 0;
      if (helper.rank() == ptree.get("debug_rank", 0))
        for (const auto& idx : problem.get_neumann_region_to_subdomain()) ring_region[idx] = 1;
      write_overlapping_vector(ring_region, "Ring region");

      writer.write(ptree.get("filename", "Poisson"));
    }

    Logger::get().endEvent(total);
    if (ptree.get("view_report", true)) Logger::get().report(helper.getCommunicator());
  }
  catch (const Dune::Exception& e) {
    std::cerr << "Caught Dune exception: " << e << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  catch (const std::exception& e) {
    std::cerr << "Caught std exception: " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 1;
  }
  catch (...) {
    std::cerr << "Caught unknown exception" << '\n';
    MPI_Abort(MPI_COMM_WORLD, 3);
    return 1;
  }

  return 0;
}
