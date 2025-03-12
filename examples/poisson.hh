#pragma once

#include <algorithm>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/version.hh>
#include <dune/istl/io.hh>
#include <dune/pdelab/backend/interface.hh>
#include <dune/pdelab/backend/istl/bcrsmatrixbackend.hh>
#include <dune/pdelab/backend/istl/parallelhelper.hh>
#include <dune/pdelab/constraints/common/constraints.hh>
#include <dune/pdelab/constraints/conforming.hh>
#include <dune/pdelab/finiteelementmap/pkfem.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/pdelab/gridfunctionspace/interpolate.hh>
#include <dune/pdelab/gridoperator/gridoperator.hh>
#include <dune/pdelab/localoperator/convectiondiffusionfem.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>

#include <memory>
#include <mpi.h>
#include <type_traits>

#include "assemblewrapper.hh"
#include "datahandles.hh"
#include "helpers.hh"

template <class GridView, class RF>
class PoissonModelProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  typename Traits::PermTensorType A(const typename Traits::ElementType &e, const typename Traits::DomainType &x) const
  {
    auto xg = e.geometry().global(x);
    const auto nx = 8;

    RF Dglobal = 1;
    if ((((int)(xg[0] * nx) % 2 != 0) && ((int)(xg[1] * nx) % 2 != 0)) || (((int)(xg[0] * nx) % 2 == 0) && ((int)(xg[1] * nx) % 2 == 0))) {
      Dglobal = 1;
    }

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++) {
      for (std::size_t j = 0; j < Traits::dimDomain; j++) {
        I[i][j] = (i == j) ? Dglobal : 0;
      }
    }
    return I;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType &, const typename Traits::DomainType &) const { return 1.0; }

  typename Traits::RangeFieldType g(const typename Traits::ElementType &, const typename Traits::DomainType &) const { return 0.0; }

  BC bctype(const typename Traits::IntersectionType &is, const typename Traits::IntersectionDomainType &x) const
  {
    auto center = is.geometry().global(x);
    if (true or center[0] < 1e-6 or center[1] < 1e-6) {
      return BC::Dirichlet;
    }
    else {
      return BC::Neumann;
    }
  }
};

template <class Grid>
constexpr bool isYASPGrid()
{
  return requires(Grid g) { g.torus(); }; // We distinguish YASPGrid and UGGrid using the method "torus" which only exists in YASPGrid
}

template <class GridView>
class PoissonProblem {
public:
  using RF = double;
  using Grid = typename GridView::Grid;
  using DF = typename Grid::ctype;

  using ModelProblem = PoissonModelProblem<GridView, RF>;
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<ModelProblem>;

  using FEM = std::conditional_t<isYASPGrid<Grid>(),                                          // If YASP grid...
                                 Dune::PDELab::QkLocalFiniteElementMap<GridView, DF, RF, 1>,  // ... then use quadrilaterals
                                 Dune::PDELab::PkLocalFiniteElementMap<GridView, DF, RF, 1>>; // ... otherwise use triangles
  using LOP = Dune::PDELab::ConvectionDiffusionFEM<ModelProblem, FEM>;

  using CON = Dune::PDELab::ConformingDirichletConstraints;
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;

  using GFS = Dune::PDELab::GridFunctionSpace<GridView, FEM, CON, Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>>;

  using CC = typename GFS::template ConstraintsContainer<RF>::Type;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, AssembleWrapper<LOP>, MBE, RF, RF, RF, CC, CC>;

  using Vec = Dune::PDELab::Backend::Vector<GFS, RF>;
  using Mat = typename GO::Jacobian;
  using NativeMat = Dune::PDELab::Backend::Native<Mat>;

  PoissonProblem(const GridView &gv, const Dune::MPIHelper &helper) : fem(gv), gfs(gv, fem), bc(gv, modelProblem), lop(modelProblem), x(std::make_unique<Vec>(gfs))
  {
    // Find out which dofs are constrained
    Dune::PDELab::constraints(bc, gfs, cc);

    // Create solution vector and initialise with Dirichlet conditions
    Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter g(gv, modelProblem);
    Dune::PDELab::interpolate(g, gfs, *x);
    Dune::PDELab::ISTL::ParallelHelper parhelper(gfs);
    parhelper.maskForeignDOFs(*x);
    Dune::PDELab::AddDataHandle adddh(gfs, *x);
    if (helper.size() > 1) {
      gfs.gridView().communicate(adddh, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);
    }

    // Create Dirichlet mask
    dirichlet_mask = std::make_unique<Vec>(gfs, 0);
    Dune::PDELab::set_constrained_dofs(cc, 1., *dirichlet_mask);

    // Create the grid operator, assemble the residual and setup the nonzero pattern of the matrix
    spdlog::info("Assembling residual");
    wrapper = std::make_unique<AssembleWrapper<LOP>>(&lop);
    go = std::make_unique<GO>(gfs, cc, gfs, cc, *wrapper, MBE(9));
    d = std::make_unique<Vec>(gfs);
    go->residual(*x, *d);

    spdlog::info("Assembling sparsity pattern");
    As = std::make_unique<Mat>(*go);
  }

  template <class RemoteIndices>
  std::pair<std::vector<TripleWithRank>, std::vector<TripleWithRank>> assembleJacobian(const RemoteIndices &remoteids, int overlap)
  {
    // TODO: This is almost the same code as in the overlapExtend function, with the exception that here we identify the distance
    //       to the boundary separately for each neighbour. This should be refactored into a single function.
    using Dune::PDELab::Backend::native;

    // The sparsity pattern is already assembled and we can use it to algebraically increase the overlap.
    if (overlap <= 0) {
      DUNE_THROW(Dune::Exception, "Overlap must be greater than zero");
    }

    int ownrank{};
    MPI_Comm_rank(remoteids.communicator(), &ownrank);

    const auto ovlpindices_before = extendOverlap(remoteids, native(*As), overlap - 1);
    auto Aovlp = createOverlappingMatrix(native(*As), *ovlpindices_before.first);

    const auto ovlpindices = extendOverlap(*ovlpindices_before.first, Aovlp, 1);
    Aovlp = std::move(createOverlappingMatrix(Aovlp, *ovlpindices.first));

    const auto ovlpindices_after = extendOverlap(*ovlpindices.first, Aovlp, 1);

    const auto paridxs2vec = [](const auto &paridxs) {
      std::vector<std::size_t> indices(paridxs.size());
      std::transform(paridxs.begin(), paridxs.end(), indices.begin(), [](auto &idx) { return idx.local().local(); });
      return indices;
    };

    // TODO: This is pretty inefficient and we could get the same information for free during overlap generation.
    //       Maybe the whole overlap extension should be factored into a class that stores some of the information
    //       we construct below.
    auto indices_before = paridxs2vec(*ovlpindices_before.second);
    std::sort(indices_before.begin(), indices_before.end());
    auto indices = paridxs2vec(*ovlpindices.second);
    std::sort(indices.begin(), indices.end());
    auto indices_after = paridxs2vec(*ovlpindices_after.second);
    std::sort(indices_after.begin(), indices_after.end());

    std::vector<std::size_t> indices_at_boundary(indices.size() - indices_before.size());
    std::set_difference(indices.begin(), indices.end(), indices_before.begin(), indices_before.end(), indices_at_boundary.begin());
    std::vector<std::size_t> indices_outside_boundary(indices_after.size(), indices.size());
    std::set_difference(indices_after.begin(), indices_after.end(), indices.begin(), indices.end(), indices_outside_boundary.begin());

    /* Now we have two sets of indices that contain:
       1. All indices that lie on the boundary of our overlapping subdomain.
       2. All indices that lie just outside the boundary of our overlapping subdomain.
       We know send those around so that the other ranks know which elements they have to treat differently
       when assembling the stiffness matrix.

       TODO: It's possible to generate this information locally, I tried and it didn't match the information
             that we get doing it the way it's done above. Would be good to know why the two approaches differ.
    */

    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    // First create a boolean mask that marks all indices on the subdomain boundary of each of our neighbouring ranks
    std::map<int, std::vector<bool>> boundary_mask_for_rank;
    {
      Dune::Interface interface;
      interface.build(*ovlpindices.first, allAttributes, allAttributes);
      Dune::VariableSizeCommunicator communicator(interface);

      MarkIndicesForRank handle(indices_at_boundary, ownrank, *ovlpindices.second);
      communicator.forward(handle);
      boundary_mask_for_rank = std::move(handle.mask_from_rank);
    }

    // Next, create a boolean mask that marks all indices just outside the subdomain boundary of each of our neighbouring ranks
    std::map<int, std::vector<bool>> outside_mask_for_rank;
    {
      Dune::Interface interface;
      interface.build(*ovlpindices_after.first, allAttributes, allAttributes);
      Dune::VariableSizeCommunicator communicator(interface);

      MarkIndicesForRank handle(indices_outside_boundary, ownrank, *ovlpindices_after.second);
      communicator.forward(handle);
      outside_mask_for_rank = std::move(handle.mask_from_rank);
    }

    assert(outside_mask_for_rank.size() == boundary_mask_for_rank.size() && "Both boolean masks should have equal size. Might actually happen that one is larger, have to think about that");

    spdlog::info("Assembling full stiffness matrix");
    go->jacobian(*x, *As);

    // Full stiffness matrix is assembled, now we can proceed assembling the individual Neumann correction matrices for each of our MPI neighbours.
    spdlog::info("Assembling Neumann correction matrices");
    for (const auto &[rank, boundary_mask] : boundary_mask_for_rank) {
      neumannCorrForRank[rank] = *As;
      neumannCorrForRank[rank] = 0.0;

      wrapper->setMasks(&boundary_mask, &(outside_mask_for_rank[rank]), rank);
      go->jacobian(*x, neumannCorrForRank[rank]);
    }

    // Now we have a set of matrices that represent corrections that remote ranks have to apply after overlap extension to turn
    // the matrices that they've obtained into matrices that correspond to a PDE with Neumann boundary conditions at the overlapping
    // subdomain boundary. Let's exchange this info with our neighbours.
    constexpr int nitems = 4;
    std::array<int, nitems> blocklengths = {1, 1, 1, 1};
    std::array<MPI_Datatype, nitems> types = {MPI_INT, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_DOUBLE};
    MPI_Datatype triple_type = MPI_DATATYPE_NULL;
    std::array<MPI_Aint, nitems> offsets{0};
    offsets[0] = offsetof(TripleWithRank, rank);
    offsets[1] = offsetof(TripleWithRank, row);
    offsets[2] = offsetof(TripleWithRank, col);
    offsets[3] = offsetof(TripleWithRank, val);
    MPI_Type_create_struct(nitems, blocklengths.data(), offsets.data(), types.data(), &triple_type);
    MPI_Type_commit(&triple_type);

    Dune::GlobalLookupIndexSet glis(remoteids.sourceIndexSet());

    std::vector<MPI_Request> requests;
    requests.reserve(neumannCorrForRank.size());
    std::map<int, std::vector<TripleWithRank>> my_triples;
    for (const auto &[rank, Anp] : neumannCorrForRank) {
      const auto &An = native(Anp);
      std::size_t cnt = 0;
      for (auto ri = An.begin(); ri != An.end(); ++ri) {
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          // We can skip all contributions from outside the overlapping subdomain, they won't be present in the overlapping matrix anyways.
          if ((not boundary_mask_for_rank[rank][ri.index()]) or (not boundary_mask_for_rank[rank][ci.index()])) {
            continue;
          }

          if (std::abs(*ci) > 1e-12) {
            cnt++;
          }
        }
      }

      my_triples[rank].resize(cnt);
      cnt = 0;
      for (auto ri = An.begin(); ri != An.end(); ++ri) {
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if ((not boundary_mask_for_rank[rank][ri.index()]) or (not boundary_mask_for_rank[rank][ci.index()])) {
            continue;
          }

          if (std::abs(*ci) > 1e-12) {
            auto col = ci.index();
            auto row = ri.index();

            // Convert the row and col to global indices
            auto gcol = glis.pair(col)->global();
            auto grow = glis.pair(row)->global();
            my_triples[rank][cnt++] = TripleWithRank{.rank = ownrank, .row = grow, .col = gcol, .val = *ci};

            spdlog::trace("New triple: ({}, {}) => {}", grow, gcol, static_cast<double>(*ci));
          }
        }
      }

      // Here we can already post the asynchronous sends
      MPI_Isend(my_triples[rank].data(), my_triples[rank].size(), triple_type, rank, 0, remoteids.communicator(), &requests.emplace_back());
    }

    std::map<int, std::vector<TripleWithRank>> remote_triples;
    std::size_t num_triples = 0;
    for (const auto &[rank, Anp] : neumannCorrForRank) {
      MPI_Status status;
      int count{};

      MPI_Probe(rank, 0, remoteids.communicator(), &status);
      MPI_Get_count(&status, triple_type, &count);
      num_triples += count;

      remote_triples[rank].resize(count);
      MPI_Recv(remote_triples[rank].data(), count, triple_type, rank, 0, remoteids.communicator(), MPI_STATUS_IGNORE);
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // Now combine all triples we received into a single list
    std::vector<TripleWithRank> all_triples;
    all_triples.reserve(num_triples);
    for (const auto &[rank, triples] : remote_triples) {
      all_triples.insert(all_triples.end(), triples.begin(), triples.end());
    }

    // TOOD: Remove, this is just for visualisation.
    interesting_elements.clear();
    for (const auto &[rank, elems] : wrapper->interesting_elements) {
      if (interesting_elements.size() == 0) {
        interesting_elements.resize(elems.size(), 0);
      }
      for (std::size_t i = 0; i < elems.size(); ++i) {
        interesting_elements[i] += elems[i];
      }
    }

    // Combine all of our own triples into one vector
    std::size_t cnt = 0;
    for (const auto &[rank, triples] : my_triples) {
      cnt += triples.size();
    }
    std::vector<TripleWithRank> all_own_triples(cnt);
    for (const auto &[rank, triples] : my_triples) {
      all_own_triples.insert(all_own_triples.end(), triples.begin(), triples.end());
    }

    return {all_triples, all_own_triples};
  }

  Vec &getX() { return *x; }
  const Vec &getD() const { return *d; }
  const Vec &getDirichletMask() const { return *dirichlet_mask; }

  const NativeMat &getA() { return Dune::PDELab::Backend::native(*As); }

  const GFS &getGFS() const { return gfs; }

  std::vector<int> interesting_elements;

private:
  FEM fem;
  GFS gfs;
  BC bc;
  CC cc;
  ModelProblem modelProblem; // The underlying problem describing the PDE

  LOP lop;
  std::unique_ptr<AssembleWrapper<LOP>> wrapper;
  std::unique_ptr<GO> go;

  std::unique_ptr<Vec> x;              // solution vector
  std::unique_ptr<Vec> d;              // residual vector
  std::unique_ptr<Vec> dirichlet_mask; // vector with ones at the dirichlet dofs
  std::unique_ptr<Mat> As;
  std::map<int, Mat> neumannCorrForRank;
  //  std::shared_ptr<NativeMat> A; // stiffness matrix
};
