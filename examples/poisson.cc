#include "logger.hh"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>

#include <mpi.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define USE_UGGRID 1
#define GRID_OVERLAP 0

#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/grid/io/file/gmshreader.hh>
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

#include "combined_preconditioner.hh"
#include "galerkin_preconditioner.hh"
#include "helpers.hh"
#include "schwarz.hh"

#include "poisson.hh"

template <class Vec, class Communication>
class MaskedScalarProduct : public Dune::ScalarProduct<Vec> {
  using Base = Dune::ScalarProduct<Vec>;

public:
  MaskedScalarProduct(const std::vector<unsigned> &mask, Communication comm) : mask(&mask), comm(comm) { dot_event = Logger::get().registerEvent(Logger::get().registerFamily("MaskedScalarProduct"), "dot"); }

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
#if USE_UGGRID
  const auto meshfile = ptree.get("meshfile", "../data/unitsquare.msh");
  const auto verbose = ptree.get("verbose", 0);

  using Grid = Dune::UGGrid<2>;
  auto grid = Dune::GmshReader<Grid>::read(meshfile, verbose > 1, false);
#else
  using Grid = Dune::YaspGrid<2>;
  const auto gridsize = ptree.get("gridsize", 32);
  auto grid = std::unique_ptr<Grid>(new Grid({1.0, 1.0}, {gridsize, gridsize}, std::bitset<2>(0ULL), GRID_OVERLAP));
#endif

  grid->globalRefine(ptree.get("refine", 2));

#if USE_UGGRID
  auto gv = grid->leafGridView();
  auto part = Dune::ParMetisGridPartitioner<decltype(gv)>::partition(gv, helper);
  grid->loadBalance(part, 0);
#else
  grid->loadBalance();
#endif

  return grid;
}

template <class GFS>
auto makeRemoteIndices(const GFS &gfs, const Dune::MPIHelper &helper)
{
  using Dune::PDELab::Backend::native;

  // Using the grid function space, we can generate a globally unique numbering of the dofs.
  // This is done by taking the local index, shifting it to the upper 32 bits of a 64 bit number and
  // taking our MPI rank as the lower 32 bits.
  using GlobalIndexVec = Dune::PDELab::Backend::Vector<GFS, std::uint64_t>;
  GlobalIndexVec giv(gfs);
  for (int i = 0; i < giv.N(); ++i) {
    native(giv)[i] = (static_cast<std::uint64_t>(i + 1) << 32ULL) + helper.rank();
  }

  // Now we have a unique global indexing scheme in the interior of each process
  // subdomain; at the process boundary we take the smallest among all processes.
  GlobalIndexVec giv_before(gfs);
  giv_before = giv; // Copy the vector so that we can find out if we are the owner of a border index after communication
  Dune::PDELab::MinDataHandle mindh(gfs, giv);
  gfs.gridView().communicate(mindh, Dune::All_All_Interface, Dune::ForwardCommunication);

  using BooleanVec = Dune::PDELab::Backend::Vector<GFS, bool>;
  BooleanVec isPublic(gfs);
  Dune::PDELab::SharedDOFDataHandle shareddh(gfs, isPublic);
  gfs.gridView().communicate(shareddh, Dune::All_All_Interface, Dune::ForwardCommunication);

  using AttributeLocalIndex = Dune::ParallelLocalIndex<Attribute>;
  using GlobalIndex = std::uint64_t;
  using ParallelIndexSet = Dune::ParallelIndexSet<GlobalIndex, AttributeLocalIndex>;
  using RemoteIndices = Dune::RemoteIndices<ParallelIndexSet>;

  ParallelIndexSet paridxs;
  paridxs.beginResize();
  for (std::size_t i = 0; i < giv.N(); ++i) {
    paridxs.add(native(giv)[i], {i,                                                                            // Local index is just i
                                 native(giv)[i] == native(giv_before)[i] ? Attribute::owner : Attribute::copy, // If the index didn't change above, we own it
                                 native(isPublic)[i]}                                                          // Index is public if multiple ranks added a value above
    );
  }
  paridxs.endResize();

  const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  const AttributeSet ownerAttribute{Attribute::owner};
  const AttributeSet copyAttribute{Attribute::copy};

  std::set<int> neighboursset;
  Dune::PDELab::GFSNeighborDataHandle nbdh(gfs, helper.rank(), neighboursset);
  gfs.gridView().communicate(nbdh, Dune::All_All_Interface, Dune::ForwardCommunication);
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

  using Vec = decltype(problem)::Vec;
  using Mat = decltype(problem)::Mat;

  // Create a mask for the owned indices for the scalar product
  std::vector<unsigned> mask(problem.getD().N(), 1);
  for (const auto &idx : paridxs) {
    if (idx.local().attribute() != Attribute::owner) {
      mask[idx.local()] = 0;
    }
  }

  if (verbose > 0) {
    auto my_dofs = std::count_if(mask.begin(), mask.end(), [](auto e) { return e > 0; });
    auto sum = helper.getCommunication().sum(my_dofs);
    if (helper.rank() == 0) {
      std::cout << "Total dofs: " << sum << "\n";
    }
  }

  MaskedScalarProduct<Native<Vec>, decltype(helper.getCommunication())> sp(mask, helper.getCommunication());

  // Build the parallel operator
  const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  Dune::Interface all_all_interface;
  all_all_interface.build(remoteindices, allAttributes, allAttributes);
  auto communicator = std::make_shared<Dune::BufferedCommunicator>();
  communicator->build<Native<Vec>>(all_all_interface);
  auto op = std::make_shared<NonoverlappingOperator<Native<Mat>, Native<Vec>, Native<Vec>>>(problem.getA(), std::move(communicator));

  // Construct the preconditioner
  double start = 0;
  double end = 0;
  if (verbose > 0 and helper.rank() == 0) {
    std::cout << "Setting up preconditioner... " << std::flush;
  }
  start = MPI_Wtime();

  auto applymode = (ptree.get("applymode", "additive") == "additive") ? ApplyMode::Additive : ApplyMode::Multiplicative;
  CombinedPreconditioner<Native<Vec>> prec(applymode);
  prec.setMat(op);

  auto schwarz = std::make_shared<SchwarzPreconditioner<Native<Vec>, Native<Mat>, std::remove_reference_t<decltype(remoteindices)>>>(*problem.getA(), remoteindices, ptree);
  prec.add(schwarz);

  Vec one(gfs);
  if (ptree.get("coarsespace", false)) {
    // Build vector of constant 1s, except for the Dirichlet dofs which are zeroed out
    Dune::PDELab::interpolate([]([[maybe_unused]] const auto &x) { return 1; }, gfs, one);
    for (std::size_t i = 0; i < one.N(); ++i) {
      if (native(problem.getDirichletMask())[i] > 0) {
        native(one)[i] = 0;
      }
    }

    auto nicolaides = std::make_shared<GalerkinPreconditioner<Native<Vec>, Native<Mat>, std::remove_reference_t<decltype(remoteindices)>>>(*schwarz->getOverlappingMat(), *schwarz->getPartitionOfUnity(), native(one), schwarz->getOverlappingIndices());
    prec.add(nicolaides);
  }
  end = MPI_Wtime();
  if (verbose > 0 and helper.rank() == 0) {
    std::cout << "Done. Took " << (end - start) << "s\n";
  }

#if !USE_UGGRID && GRID_OVERLAP > 0
  using SolverVec = Vec;
#else
  using SolverVec = Native<Vec>;
#endif

  std::unique_ptr<Dune::IterativeSolver<SolverVec, SolverVec>> solver;
  auto tol = ptree.get("tolerance", 1e-8);
  auto solvertype = ptree.get("solver", "gmres");
  if (solvertype == "cg") {
    solver = std::make_unique<Dune::CGSolver<SolverVec>>(*op, sp, prec, tol, 1000, helper.rank() == 0 ? verbose : 0);
  }
  else if (solvertype == "none") {
    solver = std::make_unique<Dune::LoopSolver<SolverVec>>(*op, sp, prec, tol, 1000, helper.rank() == 0 ? verbose : 0);
  }
  else if (solvertype == "bicgstab") {
    solver = std::make_unique<Dune::BiCGSTABSolver<SolverVec>>(*op, sp, prec, tol, 1000, helper.rank() == 0 ? verbose : 0);
  }
  else {
    solver = std::make_unique<Dune::RestartedGMResSolver<SolverVec>>(*op, sp, prec, tol, 50, 1000, helper.rank() == 0 ? verbose : 0);
    if (solvertype != "gmres") {
      if (helper.rank() == 0) {
        std::cout << "WARNING: Unknown solver type '" << solvertype << "', using GMRES instead\n";
      }
    }
  }

  Dune::InverseOperatorResult res;
  auto v = problem.getX();
  auto b = problem.getD();
  v = 0;

  solver->apply(v, b, res);
  problem.getX() -= v;

  std::vector<int> rankVec(grid->leafGridView().size(0), helper.rank());

  Dune::VTKWriter writer(grid->leafGridView());
  writer.addCellData(rankVec, "Rank");

  // #if GRID_OVERLAP == 0
  //   Vec pouvec(gfs, *schwarz->getPartitionOfUnity());
  //   if (helper.rank() != ptree.get("debug_rank", 0))
  //     native(pouvec) = 0;
  //   Dune::PDELab::DiscreteGridFunction poudgf(gfs, pouvec);
  //   auto pougfadapter = std::make_shared<Dune::PDELab::VTKGridFunctionAdapter<decltype(poudgf)>>(poudgf, "POU");
  //   writer.addVertexData(pougfadapter);
  // #endif

  Dune::PDELab::DiscreteGridFunction dgf(gfs, problem.getX());
  auto gfadapter = std::make_shared<Dune::PDELab::VTKGridFunctionAdapter<decltype(dgf)>>(dgf, "Solution");
  writer.addVertexData(gfadapter);

  Dune::PDELab::DiscreteGridFunction dgfone(gfs, one);
  auto gfadapterone = std::make_shared<Dune::PDELab::VTKGridFunctionAdapter<decltype(dgfone)>>(dgfone, "Template vec");
  writer.addVertexData(gfadapterone);

  writer.write("Poisson");

  
  Logger::get().report(helper.getCommunicator());

  return 0;
}
