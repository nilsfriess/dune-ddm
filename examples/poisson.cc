#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <cstdint>
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::int64_t

#include <dune-pdelab-config.hh>

#include "metis.hh"
#include "overlap_extension.hh"
#include "poisson.hh"

#define USE_UGGRID 0 // Set to zero to use YASPGrid
#define GRID_DIM 2
#define GRID_OVERLAP 0

#include <cmath>
#include <mpi.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <bitset>
#include <cstddef>

#include <dune/common/densevector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/iteratorfacades.hh>
#include <dune/common/parallel/communicator.hh>
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
#if USE_UGGRID
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/uggrid.hh>
#else
#include <dune/grid/yaspgrid.hh>
#include <dune/grid/yaspgrid/coordinates.hh>
#endif
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
#include <dune/pdelab/localoperator/permeability_adapter.hh>
#include <dune/pdelab/ordering/transformations.hh>

#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <utility>

#include "coarsespaces/geneo.hh"
#include "coarsespaces/msgfem.hh"
#include "coarsespaces/nicolaides.hh"
#include "combined_preconditioner.hh"
#include "datahandles.hh"
#include "galerkin_preconditioner.hh"
#include "helpers.hh"
#include "logger.hh"
#include "schwarz.hh"
#include "spdlog/common.h"

template <class Vec, class Communication>
class MaskedScalarProduct : public Dune::ScalarProduct<Vec> {
  using Base = Dune::ScalarProduct<Vec>;

public:
  MaskedScalarProduct(const std::vector<unsigned> &mask, Communication comm) : mask(&mask), comm(comm) { dot_event = Logger::get().registerEvent("MaskedScalarProduct", "dot"); }

  typename Base::field_type dot(const Vec &x, const Vec &y) const override
  {
    Logger::ScopedLog se(dot_event);

    typename Base::field_type res{0.0};
    for (typename Vec::size_type i = 0; i < x.size(); i++) {
      res += x[i] * y[i] * (*mask)[i];
    }
    return comm.sum(res);
  }

  typename Base::real_type norm(const Vec &x) const override
  {
    auto res = dot(x, x);
    return std::sqrt(res);
  }

  typename Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

private:
  const std::vector<unsigned> *mask;
  Communication comm;

  Logger::Event *dot_event;
};

namespace {
auto makeGrid(const Dune::ParameterTree &ptree, [[maybe_unused]] const Dune::MPIHelper &helper)
{
  auto *event = Logger::get().registerEvent("Grid", "create");
  Logger::ScopedLog sl(event);

#if USE_UGGRID
  const auto meshfile = ptree.get("meshfile", "../data/unitsquare.msh");
  const auto verbose = ptree.get("verbose", 0);

  using Grid = Dune::UGGrid<GRID_DIM>;
  auto grid = Dune::GmshReader<Grid>::read(meshfile, verbose > 2, false);
#else
  using Grid = Dune::YaspGrid<GRID_DIM>;
  auto gridsize = ptree.get("gridsize", 32);
  if (ptree.hasKey("gridsize_per_rank")) {
    auto grid_sqrt = static_cast<int>(std::pow(helper.size(), 1. / GRID_DIM)); // This is okay because Yaspgrid will complain if the gridsize is not a power of GRID_DIM
    gridsize = ptree.get<int>("gridsize_per_rank") * grid_sqrt;
  }
#if GRID_DIM == 2
  Dune::Yasp::PowerDPartitioning<GRID_DIM> partitioner;
  auto grid = std::unique_ptr<Grid>(new Grid({1.0, 1.0}, {gridsize, gridsize}, std::bitset<2>(0ULL), GRID_OVERLAP, Grid::Communication(), &partitioner));
#elif GRID_DIM == 3
  Dune::Yasp::PowerDPartitioning<GRID_DIM> partitioner;
  auto grid = std::unique_ptr<Grid>(new Grid({1.0, 1.0, 1.0}, {gridsize, gridsize, gridsize}, std::bitset<3>(0ULL), GRID_OVERLAP, Grid::Communication(), &partitioner));
#endif
#endif

  grid->globalRefine(ptree.get("serial_refine", 2));

#if USE_UGGRID
  auto gv = grid->leafGridView();
  auto part = partitionMETIS(gv, helper);

  grid->loadBalance(part, 0);
#else
  grid->loadBalance();
#endif

  grid->globalRefine(ptree.get("refine", 2));

  return grid;
}

template <class GFS>
auto makeRemoteIndices(const GFS &gfs, const Dune::MPIHelper &helper)
{
  using Dune::PDELab::Backend::native;

  // Using the grid function space, we can generate a globally unique numbering
  // of the dofs. This is done by taking the local index, shifting it to the
  // upper 32 bits of a 64 bit number and taking our MPI rank as the lower 32
  // bits.
  using GlobalIndexVec = Dune::PDELab::Backend::Vector<GFS, std::uint64_t>;
  GlobalIndexVec giv(gfs);
  for (std::size_t i = 0; i < giv.N(); ++i) {
    native(giv)[i] = (static_cast<std::uint64_t>(i + 1) << 32ULL) + helper.rank();
  }

  // Now we have a unique global indexing scheme in the interior of each process
  // subdomain; at the process boundary we take the smallest among all
  // processes.
  GlobalIndexVec giv_before(gfs);
  giv_before = giv; // Copy the vector so that we can find out if we are the
                    // owner of a border index after communication
  Dune::PDELab::MinDataHandle mindh(gfs, giv);
  gfs.gridView().communicate(mindh, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);

  using BooleanVec = Dune::PDELab::Backend::Vector<GFS, bool>;
  BooleanVec isPublic(gfs);
  isPublic = false;
  Dune::PDELab::SharedDOFDataHandle shareddh(gfs, isPublic);
  gfs.gridView().communicate(shareddh, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);

  using AttributeLocalIndex = Dune::ParallelLocalIndex<Attribute>;
  using GlobalIndex = std::uint64_t;
  using ParallelIndexSet = Dune::ParallelIndexSet<GlobalIndex, AttributeLocalIndex>;
  using RemoteIndices = Dune::RemoteIndices<ParallelIndexSet>;

  ParallelIndexSet paridxs;
  paridxs.beginResize();
  for (std::size_t i = 0; i < giv.N(); ++i) {
    paridxs.add(native(giv)[i],
                {i,                                                                            // Local index is just i
                 native(giv)[i] == native(giv_before)[i] ? Attribute::owner : Attribute::copy, // If the index didn't change above, we own it
                 native(isPublic)[i]}                                                          // SharedDOFDataHandle determines if an index is public
    );
  }
  paridxs.endResize();

  const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  const AttributeSet ownerAttribute{Attribute::owner};
  const AttributeSet copyAttribute{Attribute::copy};

  std::set<int> neighboursset;
  Dune::PDELab::GFSNeighborDataHandle nbdh(gfs, helper.rank(), neighboursset);
  gfs.gridView().communicate(nbdh, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);
  std::vector<int> neighbours(neighboursset.begin(), neighboursset.end());

  auto remoteindices = std::make_shared<RemoteIndices>(paridxs, paridxs, helper.getCommunicator(), neighbours);
  return makeRemoteParallelIndices(remoteindices);
}
} // namespace

int main(int argc, char *argv[])
{
  using Dune::PDELab::Backend::native;
  using Dune::PDELab::Backend::Native;

  const auto &helper = Dune::MPIHelper::instance(argc, argv);
  setup_loggers(helper.rank(), argc, argv);

  auto *matrix_setup = Logger::get().registerEvent("Total", "Setup problem");
  auto *prec_setup = Logger::get().registerEvent("Total", "Setup preconditioner");
  auto *solve = Logger::get().registerEvent("Total", "Linear solve");
  auto *total = Logger::get().registerEvent("Total", "Total time");
  Logger::get().startEvent(total);

  Logger::get().startEvent(matrix_setup);
  Dune::ParameterTree ptree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readOptions(argc, argv, ptree);
  const auto verbose = ptree.get("verbose", 0);

  auto grid = makeGrid(ptree, helper);
  auto gv = grid->leafGridView();

  PoissonProblem problem(gv, helper);
  const auto &gfs = problem.getGFS();

  auto remoteparidxs = makeRemoteIndices(gfs, helper);
  const auto &remoteindices = *remoteparidxs.first;
  const auto &paridxs = *remoteparidxs.second;

  ExtendedRemoteIndices ext_indices(remoteindices, problem.getA(), ptree.get("overlap", 1));
  const auto [Aovlp, remote_ncorr_triples, own_ncorr_triples, interior_dof_mask] = problem.assembleJacobian(remoteindices, ext_indices, ptree.get("overlap", 1));

  using Vec = decltype(problem)::Vec;
  using Mat = decltype(problem)::Mat;

  // Create a mask for the owned indices for the scalar product
  std::vector<unsigned> mask(problem.getD().N(), 1);
  for (const auto &idx : paridxs) {
    if (idx.local().attribute() != Attribute::owner) {
      mask[idx.local()] = 0;
    }
  }

  auto my_dofs = std::count_if(mask.begin(), mask.end(), [](auto e) { return e > 0; });
  auto sum = helper.getCommunication().sum(my_dofs);
  spdlog::info("Total dofs: {}", sum);

  MaskedScalarProduct<Native<Vec>, decltype(helper.getCommunication())> sp(mask, helper.getCommunication());

  // Build the parallel operator
  const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  Dune::Interface all_all_interface;
  all_all_interface.build(remoteindices, allAttributes, allAttributes);
  auto communicator = std::make_shared<Dune::BufferedCommunicator>();
  communicator->build<Native<Vec>>(all_all_interface);
  auto op = std::make_shared<NonoverlappingOperator<Native<Mat>, Native<Vec>, Native<Vec>>>(problem.getA(), communicator);

  // Construct the preconditioner
  double start = 0;
  double end = 0;
  start = MPI_Wtime();

  ApplyMode applymode = ApplyMode::Additive;
  auto applymode_param = ptree.get("applymode", "additive");
  if (applymode_param == "additive") {
    applymode = ApplyMode::Additive;
  }
  else if (applymode_param == "multiplicative") {
    applymode = ApplyMode::Multiplicative;
  }
  else {
    applymode = ApplyMode::Additive;
    spdlog::warn("Unknown apply mode for combined preconditioner, using 'additive' instead");
  }
  Logger::get().endEvent(matrix_setup);

  MPI_Barrier(MPI_COMM_WORLD);

  Logger::get().startEvent(prec_setup);
  auto schwarz = std::make_shared<SchwarzPreconditioner<Native<Vec>, Native<Mat>, std::remove_reference_t<decltype(ext_indices)>>>(Aovlp, ext_indices, ptree);

  CombinedPreconditioner<Native<Vec>> prec(applymode, {schwarz}, op);

  // Check if partition of unity is actually a partition of unity
  Native<Vec> pou_vis;
  {
    AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    Dune::Interface all_all_interface;
    all_all_interface.build(ext_indices.get_remote_indices(), allAttributes, allAttributes);
    Dune::VariableSizeCommunicator all_all_comm(all_all_interface);
    AddVectorDataHandle<Native<Vec>> advdh;

    auto pou = *schwarz->getPartitionOfUnity();
    pou_vis = pou;

    advdh.setVec(pou);
    all_all_comm.forward(advdh);

    auto all_one = std::all_of(pou.begin(), pou.end(), [](auto val) { return std::abs(val - 1.0) < 1e-10; });
    if (!all_one) {
      spdlog::get("all_ranks")->error("Partition of unity does not add up to 1, max is {}, min is {}", *std::max_element(pou.begin(), pou.end()), *std::min_element(pou.begin(), pou.end()));
      // MPI_Abort(MPI_COMM_WORLD, 12);
    }
    else {
      spdlog::get("all_ranks")->debug("Partition of unity does add up to 1");
    }
  }

  Native<Vec> visuvec(pou_vis.N());
  const auto coarsespace = ptree.get("coarsespace", "geneo");
  if (coarsespace == "nicolaides") {
    int nvecs = ptree.get("nvecs", 1);
    if (nvecs < 1 or nvecs > 3) {
      nvecs = 1;
      spdlog::warn("Wrong number of template vectors, using 1 instead");
    }

    std::vector<Vec> template_vecs(nvecs, Vec(gfs));
    Dune::PDELab::interpolate([](const auto &) { return 1; }, gfs, template_vecs[0]);
    if (nvecs > 1) {
      Dune::PDELab::interpolate([](const auto &x) { return x[0]; }, gfs, template_vecs[1]);
    }
    if (nvecs > 2) {
      Dune::PDELab::interpolate([](const auto &x) { return x[1]; }, gfs, template_vecs[2]);
    }
    for (auto &template_vec : template_vecs) {
      for (std::size_t i = 0; i < template_vec.N(); ++i) {
        if (native(problem.getDirichletMask())[i] > 0) {
          native(template_vec)[i] = 0;
        }
      }
    }

    std::vector<Native<Vec>> native_template_vecs(template_vecs.size());
    for (std::size_t i = 0; i < template_vecs.size(); ++i) {
      native_template_vecs[i] = native(template_vecs[i]);
    }

    auto basis_vecs = buildNicolaidesCoarseSpace(ext_indices.get_remote_indices(), *schwarz->getOverlappingMat(), native_template_vecs, interior_dof_mask, *schwarz->getPartitionOfUnity(), ptree);

    auto nicolaides = std::make_shared<GalerkinPreconditioner<Native<Vec>, Native<Mat>, std::remove_reference_t<decltype(remoteindices)>>>(*schwarz->getOverlappingMat(), basis_vecs,
                                                                                                                                           ext_indices.get_remote_par_indices());
    prec.add(nicolaides);
  }
  else if (coarsespace == "geneo") {
    auto basis_vecs = buildGenEOCoarseSpace(ext_indices.get_remote_par_indices(), *schwarz->getOverlappingMat(), remote_ncorr_triples, own_ncorr_triples, interior_dof_mask,
                                            native(problem.getDirichletMask()), *schwarz->getPartitionOfUnity(), ptree);

    auto geneo = std::make_shared<GalerkinPreconditioner<Native<Vec>, Native<Mat>, std::remove_reference_t<decltype(remoteindices)>>>(*schwarz->getOverlappingMat(), basis_vecs,
                                                                                                                                      ext_indices.get_remote_par_indices());
    prec.add(geneo);
  }
  else if (coarsespace == "msgfem") {
    auto basis_vecs = buildMsGFEMCoarseSpace(ext_indices.get_remote_par_indices(), *schwarz->getOverlappingMat(), remote_ncorr_triples, own_ncorr_triples, interior_dof_mask,
                                             native(problem.getDirichletMask()), *schwarz->getPartitionOfUnity(), ptree);

    visuvec = basis_vecs[ptree.get("n_vis", 0)];

    auto msgfem = std::make_shared<GalerkinPreconditioner<Native<Vec>, Native<Mat>, std::remove_reference_t<decltype(remoteindices)>>>(*schwarz->getOverlappingMat(), basis_vecs,
                                                                                                                                       ext_indices.get_remote_par_indices());
    prec.add(msgfem);
  }
  else if (coarsespace == "none") {
    // Nothing to do here
  }
  else {
    spdlog::error("Unknown coarse space type '{}'", coarsespace);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  end = MPI_Wtime();
  spdlog::info("Done setting up preconditioner, took {:.5f}s", (end - start));

#if !USE_UGGRID && GRID_OVERLAP > 0
  using SolverVec = Vec;
#else
  using SolverVec = Native<Vec>;
#endif
  Logger::get().endEvent(prec_setup);

  MPI_Barrier(MPI_COMM_WORLD);

  Logger::get().startEvent(solve);
  std::unique_ptr<Dune::IterativeSolver<SolverVec, SolverVec>> solver;
  auto maxit = ptree.get("maxit", 1000);
  auto tol = ptree.get("tolerance", 1e-8);
  auto solvertype = ptree.get("solver", "gmres");
  if (solvertype == "cg") {
    solver = std::make_unique<Dune::CGSolver<SolverVec>>(*op, sp, prec, tol, maxit, helper.rank() == 0 ? verbose : 0, helper.rank() == 0 ? verbose > 0 : false);
  }
  else if (solvertype == "none") {
    solver = std::make_unique<Dune::LoopSolver<SolverVec>>(*op, sp, prec, tol, maxit, helper.rank() == 0 ? verbose : 0);
  }
  else if (solvertype == "bicgstab") {
    solver = std::make_unique<Dune::BiCGSTABSolver<SolverVec>>(*op, sp, prec, tol, maxit, helper.rank() == 0 ? verbose : 0);
  }
  else {
    solver = std::make_unique<Dune::RestartedGMResSolver<SolverVec>>(*op, sp, prec, tol, 50, maxit, helper.rank() == 0 ? verbose : 0);
    if (solvertype != "gmres") {
      if (helper.rank() == 0) {
        std::cout << "WARNING: Unknown solver type '" << solvertype << "', using GMRES instead\n";
      }
    }
  }

  Dune::InverseOperatorResult res;
  Native<Vec> v(native(problem.getX()));
  Native<Vec> b = problem.getD();

  v = 0;
  solver->apply(v, b, res);
  problem.getX() -= v;
  Logger::get().endEvent(solve);

  // Visualisation
  if (ptree.get("visualise", true)) {
    Dune::SubsamplingVTKWriter writer(problem.getEntitySet(), Dune::refinementLevels(0));

    // Write MPI partitioning
    std::vector<int> rankVec(problem.getEntitySet().size(0), helper.rank());
    Dune::P0VTKFunction rankFunc(problem.getEntitySet(), rankVec, "Rank");
    writer.addCellData(Dune::stackobject_to_shared_ptr(rankFunc));

    // Write interior cells
    auto dof_mask = interior_dof_mask;
    Dune::P1VTKFunction interiorFunc(problem.getEntitySet(), dof_mask, "Interior");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(interiorFunc));

    // Plot the finite element solution
    Dune::P1VTKFunction residualFunc(problem.getEntitySet(), problem.getD(), "Residual");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(residualFunc));

    // Plot the finite element solution
    Dune::P1VTKFunction solutionFunc(problem.getEntitySet(), problem.getX(), "Solution");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(solutionFunc));

    PermeabilityAdapter permdgf(problem.getEntitySet(), problem.getUnderlyingProblem()); // This is defined in PDELab but not in a namespace
    typedef Dune::PDELab::VTKGridFunctionAdapter<decltype(permdgf)> PermVTKDGF;
    writer.addCellData(std::make_shared<PermVTKDGF>(permdgf, "log(K)"));

    // Write some additional output
    AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    Dune::Interface all_all_interface;
    all_all_interface.build(ext_indices.get_remote_indices(), allAttributes, allAttributes);
    Dune::VariableSizeCommunicator all_all_comm(all_all_interface);
    AddVectorDataHandle<Native<Vec>> advdh;

    // Overlapping subdomain
    Native<Vec> ovlp_subdomain(ext_indices.size());
    ovlp_subdomain = helper.rank() == ptree.get("debug_rank", 0) ? 1 : 0;
    advdh.setVec(ovlp_subdomain);
    all_all_comm.forward(advdh);
    Native<Vec> ovlp_subdomain_small(paridxs.size());
    for (std::size_t i = 0; i < ovlp_subdomain_small.size(); ++i) {
      ovlp_subdomain_small[i] = ovlp_subdomain[i];
    }
    Dune::P1VTKFunction ovlp_subdomain_function(problem.getEntitySet(), ovlp_subdomain_small, "Overlapping Subdomain");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(ovlp_subdomain_function));

    // Inner ring corrections
    Native<Vec> inner_ring_corrections(dof_mask.size());
    inner_ring_corrections = 0;
    for (const auto &triple : own_ncorr_triples) {
      inner_ring_corrections[triple.row] = 1;
      inner_ring_corrections[triple.col] = 1;
    }
    Dune::P1VTKFunction inner_ring_corrections_function(problem.getEntitySet(), inner_ring_corrections, "Own corrections");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(inner_ring_corrections_function));

    // Visualise pou
    if (helper.rank() != ptree.get("debug_rank", 0)) {
      pou_vis = 0;
    }
    advdh.setVec(pou_vis);
    all_all_comm.forward(advdh);

    Native<Vec> pou_vec_small(problem.getX().N());
    for (std::size_t i = 0; i < pou_vec_small.N(); ++i) {
      pou_vec_small[i] = pou_vis[i];
    }
    Dune::P1VTKFunction pouFunc(problem.getEntitySet(), pou_vec_small, "POU");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(pouFunc));

    // Add a vector with the dof numbering for debugging
    Native<Vec> dof_numbers(problem.getX().N());
    dof_numbers = 0;
    if (helper.rank() == ptree.get("debug_rank", 0)) {
      std::iota(dof_numbers.begin(), dof_numbers.end(), 0);
    }
    Dune::P1VTKFunction dofFunc(problem.getEntitySet(), dof_numbers, "DOFs");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(dofFunc));

    // Visualise one of the basis vectors extracted after setup of the coarse space
    if (helper.rank() != ptree.get("debug_rank", 0)) {
      visuvec = 0;
    }
    advdh.setVec(visuvec);
    all_all_comm.forward(advdh);
    visuvec.resize(problem.getX().N());
    Dune::P1VTKFunction basisFunc(problem.getEntitySet(), visuvec, "Basis function");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(basisFunc));

    writer.write(ptree.get("filename", "Poisson"));
  }

  Logger::get().endEvent(total);
  if (ptree.get("view_report", true)) {
    Logger::get().report(helper.getCommunicator());
  }
}
