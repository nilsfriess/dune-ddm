#include <cstddef>
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::int64_t

#include "logger.hh" // Must be included at the very top if MPI calls should be logged

#include <algorithm>
#include <cassert>
#include <iostream>

#include <spdlog/cfg/argv.h>
#include <spdlog/spdlog.h>

#include <mpi.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define USE_UGGRID 0 // Set to zero to use YASPGrid
#define GRID_DIM 3
#define GRID_OVERLAP 0

#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/istl/bccsmatrixinitializer.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/gsetc.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/solver.hh>
#include <dune/pdelab.hh>

#include "coarsespaces/geneo.hh"
#include "combined_preconditioner.hh"
#include "datahandles.hh"
#include "galerkin_preconditioner.hh"
#include "helpers.hh"
#include "schwarz.hh"

#include "poisson.hh"

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
  const auto gridsize = ptree.get("gridsize", 32);
#if GRID_DIM == 2
  auto grid = std::unique_ptr<Grid>(new Grid({1.0, 1.0}, {gridsize, gridsize}, std::bitset<2>(0ULL), GRID_OVERLAP));
#elif GRID_DIM == 3
  auto grid = std::unique_ptr<Grid>(new Grid({1.0, 1.0, 0.5}, {gridsize, gridsize, gridsize / 2}, std::bitset<3>(0ULL), GRID_OVERLAP));
#endif
#endif

  grid->globalRefine(ptree.get("serial_refine", 2));

#if USE_UGGRID
  auto gv = grid->leafGridView();
  auto part = Dune::ParMetisGridPartitioner<decltype(gv)>::partition(gv, helper);

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
  for (int i = 0; i < giv.N(); ++i) {
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

  // Now we have a set of remote indices on the non-overlapping grid, and a stiffness matrix with the correct sparsity pattern.
  // We can use those to find out which dofs we need to treat differently when assembling the matrix. This is all done in the
  // following method.
  const auto [remote_ncorr_triples, own_ncorr_triples, interior_dof_mask] = problem.assembleJacobian(remoteindices, ptree.get("overlap", 1));

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

  auto schwarz = std::make_shared<SchwarzPreconditioner<Native<Vec>, Native<Mat>, std::remove_reference_t<decltype(remoteindices)>>>(problem.getA(), remoteindices, ptree);

  CombinedPreconditioner<Native<Vec>> prec(applymode, {schwarz}, op);

  Vec visuvec(gfs); // A vector that can be used for visualisation

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

    // Extend to overlapping index set and multiply with partition of unity
    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    const AttributeSet ownerAttribute{Attribute::owner};
    const AttributeSet copyAttribute{Attribute::copy};

    Dune::Interface owner_copy_interface;
    owner_copy_interface.build(*schwarz->getOverlappingIndices().first, ownerAttribute, copyAttribute);
    Dune::VariableSizeCommunicator owner_copy_comm(owner_copy_interface);

    CopyVectorDataHandle<Native<Vec>> cvdh{};

    std::vector<Native<Vec>> extended_template_vecs(native_template_vecs.size(), Native<Vec>(schwarz->getPartitionOfUnity()->N()));
    for (std::size_t i = 0; i < native_template_vecs.size(); ++i) {
      extended_template_vecs[i] = 0;
      for (std::size_t j = 0; j < native_template_vecs[i].N(); ++j) {
        extended_template_vecs[i][j] = native_template_vecs[i][j];
      }

      cvdh.setVec(extended_template_vecs[i]);
      owner_copy_comm.forward(cvdh);

      // Multiply with the partition of unity
      for (std::size_t j = 0; j < extended_template_vecs[i].N(); ++j) {
        extended_template_vecs[i][j] *= (*schwarz->getPartitionOfUnity())[j];
      }
    }

    native(visuvec) = extended_template_vecs[0]; // Save one vector for visualisation

    auto nicolaides = std::make_shared<GalerkinPreconditioner<Native<Vec>, Native<Mat>, std::remove_reference_t<decltype(remoteindices)>>>(*schwarz->getOverlappingMat(), extended_template_vecs,
                                                                                                                                           schwarz->getOverlappingIndices());
    prec.add(nicolaides);
  }
  else if (coarsespace == "geneo") {
    auto basis_vecs = buildGenEOCoarseSpace(schwarz->getOverlappingIndices(), *schwarz->getOverlappingMat(), remote_ncorr_triples, own_ncorr_triples, interior_dof_mask,
                                            native(problem.getDirichletMask()), *schwarz->getPartitionOfUnity(), ptree);
    native(visuvec) = basis_vecs[ptree.get("n_vis", 0)]; // Save one vectors for visualisation

    auto geneo = std::make_shared<GalerkinPreconditioner<Native<Vec>, Native<Mat>, std::remove_reference_t<decltype(remoteindices)>>>(*schwarz->getOverlappingMat(), basis_vecs,
                                                                                                                                      schwarz->getOverlappingIndices());
    prec.add(geneo);
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

  std::unique_ptr<Dune::IterativeSolver<SolverVec, SolverVec>> solver;
  auto maxit = ptree.get("maxit", 1000);
  auto tol = ptree.get("tolerance", 1e-8);
  auto solvertype = ptree.get("solver", "gmres");
  if (solvertype == "cg") {
    solver = std::make_unique<Dune::CGSolver<SolverVec>>(*op, sp, prec, tol, maxit, helper.rank() == 0 ? verbose : 0);
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

  // Visualisation
  if (ptree.get("visualise", true)) {
    Dune::SubsamplingVTKWriter writer(problem.getEntitySet(), Dune::refinementLevels(0));

    // Write MPI partitioning
    std::vector<int> rankVec(problem.getEntitySet().size(0), helper.rank());
    Dune::P0VTKFunction rankFunc(problem.getEntitySet(), rankVec, "Rank");
    writer.addCellData(Dune::stackobject_to_shared_ptr(rankFunc));

    // Plot the finite element solution
    Dune::P1VTKFunction residualFunc(problem.getEntitySet(), problem.getD(), "Residual");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(residualFunc));

    // Plot the finite element solution
    Dune::P1VTKFunction solutionFunc(problem.getEntitySet(), problem.getX(), "Solution");
    writer.addVertexData(Dune::stackobject_to_shared_ptr(solutionFunc));

    PermeabilityAdapter permdgf(problem.getEntitySet(), problem.getUnderlyingProblem()); // This is defined in PDELab but not in a namespace
    typedef Dune::PDELab::VTKGridFunctionAdapter<decltype(permdgf)> PermVTKDGF;
    writer.addCellData(std::make_shared<PermVTKDGF>(permdgf, "Permeability"));

    writer.write(ptree.get("filename", "Poisson"));
  }

  if (ptree.get("view_report", true)) {
    Logger::get().report(helper.getCommunicator());
  }
}
