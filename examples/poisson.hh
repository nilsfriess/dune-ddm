#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <mpi.h>

#include <dune/common/version.hh>

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/grid/common/gridenums.hh>
#include <dune/istl/io.hh>
#include <dune/pdelab/backend/interface.hh>
#include <dune/pdelab/backend/istl/bcrsmatrixbackend.hh>
#include <dune/pdelab/backend/istl/vector.hh>
#include <dune/pdelab/common/partitionviewentityset.hh>
#include <dune/pdelab/constraints/common/constraints.hh>
#include <dune/pdelab/constraints/conforming.hh>
#include <dune/pdelab/finiteelementmap/pkfem.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/pdelab/gridfunctionspace/genericdatahandle.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include <dune/pdelab/gridfunctionspace/interpolate.hh>
#include <dune/pdelab/gridoperator/gridoperator.hh>
#include <dune/pdelab/localoperator/convectiondiffusionfem.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>

#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "assemblewrapper.hh"
#include "datahandles.hh"
#include "helpers.hh"
#include "overlap_extension.hh"
#include "spdlog/spdlog.h"

template <class GridView, class RF>
class PoissonModelProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  typename Traits::PermTensorType A(const typename Traits::ElementType &e, const typename Traits::DomainType &x) const
  {
    auto xg = e.geometry().global(x);
    const auto radius = 0.02;

    const auto square = [](auto &&v) { return v * v; };

    RF val = 1;
    int divx = 10;
    int divy = 10;
    for (int i = 0; i < divx - 1; ++i) {
      for (int j = 0; j < divy - 1; ++j) {
        typename Traits::DomainType centre{(i + 1.) / divx, (j + 1.) / divy};

        if (square(xg[0] - centre[0]) + square(xg[1] - centre[1]) < square(radius)) {
          val = 1e6;
        }
      }
    }

    if constexpr (GridView::dimension == 3) {
      if (xg[2] < 0.05 or xg[2] > 1 - 0.05) {
        val = 1;
      }
    }

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++) {
      for (std::size_t j = 0; j < Traits::dimDomain; j++) {
        I[i][j] = (i == j) ? val : 0;
      }
    }
    return I;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType &, const typename Traits::DomainType &) const { return 1.0; }

  typename Traits::RangeFieldType g(const typename Traits::ElementType &, const typename Traits::DomainType &) const { return 0; }

  BC bctype(const typename Traits::IntersectionType &, const typename Traits::IntersectionDomainType &) const { return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet; }
};

template <class GridView, class RF>
class IslandsModelProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  typename Traits::PermTensorType A(const typename Traits::ElementType &e, const typename Traits::DomainType &) const
  {
    auto xg = e.geometry().center();

    int ix = std::floor(15.0 * xg[0]);
    int iy = std::floor(15.0 * xg[1]);
    auto x = xg[0];
    auto y = xg[1];

    double kappa = 1.0;

    if (x > 0.3 && x < 0.9 && y > 0.6 - (x - 0.3) / 6 && y < 0.8 - (x - 0.3) / 6) {
      kappa = pow(10, 5.0) * (x + y) * 10.0;
    }

    if (x > 0.1 && x < 0.5 && y > 0.1 + x && y < 0.25 + x) {
      kappa = pow(10, 5.0) * (1.0 + 7.0 * y);
    }

    if (x > 0.5 && x < 0.9 && y > 0.15 - (x - 0.5) * 0.25 && y < 0.35 - (x - 0.5) * 0.25) {
      kappa = pow(10, 5.0) * 2.5;
    }

    if (ix % 2 == 0 && iy % 2 == 0) {
      kappa = pow(10, 5.0) * (1.0 + ix + iy);
    }

    if (GridView::Grid::dimension == 3) {
      const auto radius = 0.02;
      const auto square = [](auto &&v) { return v * v; };

      int divx = 9;
      int divz = 9;
      for (int i = 0; i < divx - 1; ++i) {
        for (int j = 0; j < divz - 1; ++j) {
          typename Traits::DomainType centre{(i + 1.) / divx, (j + 1.) / divz, 0};

          if (square(xg[0] - centre[0]) + square(xg[2] - centre[1]) < square(radius)) {
            kappa = 1e6;
          }
        }
      }
    }

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++) {
      for (std::size_t j = 0; j < Traits::dimDomain; j++) {
        I[i][j] = (i == j) ? kappa : 0;
      }
    }

    return I;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType &, const typename Traits::DomainType &) const { return 0.0; }

  typename Traits::RangeFieldType g(const typename Traits::ElementType &e, const typename Traits::DomainType &xlocal) const
  {
    auto xglobal = e.geometry().global(xlocal);
    return 1. - xglobal[0];
  }

  BC bctype(const typename Traits::IntersectionType &is, const typename Traits::IntersectionDomainType &x) const
  {
    auto xglobal = is.geometry().global(x);
    if (xglobal[0] < 1e-6) {
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    }
    if (xglobal[0] > 1.0 - 1e-6) {
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    }
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
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

  using ES = Dune::PDELab::NonOverlappingEntitySet<GridView>; // Skip ghost elements during assembly

  // using ModelProblem = PoissonModelProblem<ES, RF>;
  using ModelProblem = IslandsModelProblem<ES, RF>;
  // using ModelProblem = Dune::PDELab::ConvectionDiffusionModelProblem<ES, RF>;
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<ModelProblem>;

  using FEM = std::conditional_t<isYASPGrid<Grid>(),                                          // If YASP grid...
                                 Dune::PDELab::QkLocalFiniteElementMap<GridView, DF, RF, 1>,  // ... then use quadrilaterals
                                 Dune::PDELab::PkLocalFiniteElementMap<GridView, DF, RF, 1>>; // ... otherwise use triangles
  using LOP = Dune::PDELab::ConvectionDiffusionFEM<ModelProblem, FEM>;

  using CON = Dune::PDELab::ConformingDirichletConstraints;
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;

  using GFS = Dune::PDELab::GridFunctionSpace<ES, FEM, CON, Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>>;

  using CC = typename GFS::template ConstraintsContainer<RF>::Type;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, AssembleWrapper<LOP>, MBE, RF, RF, RF, CC, CC>;

  using Vec = Dune::PDELab::Backend::Vector<GFS, RF>;
  using Mat = typename GO::Jacobian;
  using NativeMat = Dune::PDELab::Backend::Native<Mat>;
  using NativeVec = Dune::PDELab::Backend::Native<Vec>;

  PoissonProblem(const GridView &gv, const Dune::MPIHelper &helper)
      : es(gv), fem(gv), gfs(es, fem), bc(es, modelProblem), lop(modelProblem), wrapper(std::make_unique<AssembleWrapper<LOP>>(&lop)), x(std::make_unique<Vec>(gfs, 0.0)),
        x0(std::make_unique<Vec>(gfs, 0.)), d(std::make_unique<Vec>(gfs, 0.)), dirichlet_mask(std::make_unique<Vec>(gfs, 0))
  {
    using Dune::PDELab::Backend::native;
    // Name the GFS for visualisation
    gfs.name("Solution");

    // Create solution vector and initialise with Dirichlet conditions
    Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter g(es, modelProblem);
    cc.clear();
    Dune::PDELab::interpolate(g, gfs, *x);
    Dune::PDELab::constraints(bc, gfs, cc);

    // Set Dirichlet mask
    Dune::PDELab::set_constrained_dofs(cc, 1., *dirichlet_mask);

    // Create the grid operator, assemble the residual and setup the nonzero pattern of the matrix
    go = std::make_unique<GO>(gfs, cc, gfs, cc, *wrapper, MBE(9));

    spdlog::info("Assembling sparsity pattern");
    As = std::make_unique<Mat>(*go);

    spdlog::info("Assembling residual");
    go->residual(*x, *d);

    *x0 = *x;

    if (helper.size() > 1) {
      Dune::PDELab::AddDataHandle adddhd(gfs, *d);
      gfs.gridView().communicate(adddhd, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);
    }
  }

  template <class RemoteIndices>
  std::tuple<std::vector<TripleWithRank>, std::vector<TripleWithRank>, std::vector<bool>> assembleJacobian(const RemoteIndices &remoteids, int overlap)
  {
    using Dune::PDELab::Backend::native;

    int ownrank{};
    MPI_Comm_rank(remoteids.communicator(), &ownrank);
    const auto &paridxs = remoteids.sourceIndexSet();

    // Create a vector with ones on the overlapping subdomain boundary, and one vector with ones
    // "one layer further"
    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    Dune::Interface interface;
    interface.build(remoteids, allAttributes, allAttributes);
    Dune::VariableSizeCommunicator varcomm(interface);

    IdentifyBoundaryDataHandle ibdh(native(*As), remoteids.sourceIndexSet());
    varcomm.forward(ibdh);

    // First we create (for each rank) two boolean masks:
    //   - One marks the dofs that have distance 'overlap' from the subdomain boundary
    //     that we share with a given rank.
    //   - The other one marks the dofs that have distance 'overlap + 1' from the
    //     subdomain boundary that we share with a given rank.
    // All elements that contain dofs from both sets corresponding to these masks are
    // exactly those elements that we would need to skip during assembly in order to
    // assemble a matrix that corresponds to a "Neumann" matrix for the given remote
    // rank. Instead of skipping those elements, we compute the contributions from
    // them and send those contributions around. Our neighbouring ranks can then
    // subtract those contributions from their matrices if they need Neumann matrices.
    auto subdomain_boundary_mask_for_rank = ibdh.getBoundaryMaskForRank();
    std::map<int, std::vector<int>> boundary_dst_for_rank;
    for (const auto &[rank, mask] : subdomain_boundary_mask_for_rank) {
      boundary_dst_for_rank[rank].resize(paridxs.size(), std::numeric_limits<int>::max() - 1);

      for (const auto &idxpair : paridxs) {
        auto li = idxpair.local();
        if (mask[li]) {
          boundary_dst_for_rank[rank][li] = 0;
        }
      }

      for (int round = 0; round <= overlap + 2; ++round) {
        for (std::size_t i = 0; i < boundary_dst_for_rank[rank].size(); ++i) {
          for (auto cIt = native(*As)[i].begin(); cIt != native(*As)[i].end(); ++cIt) {
            boundary_dst_for_rank[rank][i] = std::min(boundary_dst_for_rank[rank][i], boundary_dst_for_rank[rank][cIt.index()] + 1); // Increase distance from boundary by one
          }
        }
      }
    }

    std::map<int, std::vector<bool>> on_boundary_mask_for_rank;
    std::map<int, std::vector<bool>> outside_boundary_mask_for_rank;
    for (const auto &[rank, dstmap] : boundary_dst_for_rank) {
      on_boundary_mask_for_rank[rank].resize(paridxs.size(), false);
      outside_boundary_mask_for_rank[rank].resize(paridxs.size(), false);

      for (std::size_t i = 0; i < paridxs.size(); ++i) {
        on_boundary_mask_for_rank[rank][i] = dstmap[i] == overlap;
        outside_boundary_mask_for_rank[rank][i] = dstmap[i] == overlap + 1;
      }
    }

    const auto &boundary_mask = ibdh.get_boundary_mask();

    std::vector<int> boundary_dst(remoteids.sourceIndexSet().size(), std::numeric_limits<int>::max() - 1);
    for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
      if (boundary_mask[i]) {
        boundary_dst[i] = 0;
      }
    }

    for (int round = 0; round < overlap + 2; ++round) {
      for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
        for (auto cit = native(*As)[i].begin(); cit != native(*As)[i].end(); ++cit) {
          boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cit.index()] + 1); // Increase distance from boundary by one
        }
      }
    }

    // Now we have the two boolean masks as explained above. Since we might also have to apply these
    // "Neumann corrections" to some of our own indices (e.g., in the case of the GenEO ring coarse
    // space), we create two additional boolean masks corresponding to those interior corrections.
    std::vector<bool> on_boundary_mask(As->N(), false);
    std::vector<bool> outside_boundary_mask(As->N(), false);
    for (std::size_t i = 0; i < on_boundary_mask.size(); ++i) {
      on_boundary_mask[i] = boundary_dst[i] == overlap;
      outside_boundary_mask[i] = boundary_dst[i] == (overlap + 1);
    }

    spdlog::info("Assembling full stiffness matrix and corrections");
    wrapper->setMasks(native(*As), &on_boundary_mask_for_rank, &outside_boundary_mask_for_rank, &on_boundary_mask, &outside_boundary_mask);
    go->jacobian(*x, *As);
    spdlog::info("Stiffness matrix assembly done");

    Dune::GlobalLookupIndexSet glis(remoteids.sourceIndexSet());
    auto triples_for_rank = wrapper->get_correction_triples(glis);

    // Now we have a set of {row, col, value} triples that represent corrections that remote ranks have to apply after overlap extension to turn
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

    std::vector<MPI_Request> requests;
    requests.reserve(triples_for_rank.size());
    for (const auto &[rank, triples] : triples_for_rank) {
      if (rank < 0) {
        // rank == -1 corresponds to corrections that we have to apply locally, so we can skip them here
        continue;
      }

      MPI_Isend(triples.data(), triples.size(), triple_type, rank, 0, remoteids.communicator(), &requests.emplace_back());
    }

    std::map<int, std::vector<TripleWithRank>> remote_triples;
    std::size_t num_triples = 0;
    for (const auto &[rank, triples] : triples_for_rank) {
      if (rank < 0) {
        // rank < 0 corresponds to corrections that we have to apply locally, so we can skip them here
        continue;
      }

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

    std::vector<TripleWithRank> all_own_triples(std::move(triples_for_rank.at(-1)));

    std::vector<bool> interior(As->N(), false);
    for (std::size_t i = 0; i < interior.size(); ++i) {
      if (boundary_dst[i] > overlap) {
        interior[i] = true;
      }
    }

    MPI_Type_free(&triple_type);
    return {all_triples, all_own_triples, interior};
  }

  Vec &getXVec() { return *x0; }
  NativeVec &getX() { return Dune::PDELab::Backend::native(*x0); }
  NativeVec &getD() const { return Dune::PDELab::Backend::native(*d); }
  Vec &getDVec() const { return *d; }
  const Vec &getDirichletMask() const { return *dirichlet_mask; }

  NativeMat &getA() { return Dune::PDELab::Backend::native(*As); }

  const GFS &getGFS() const { return gfs; }
  const ES &getEntitySet() const { return es; }

  const ModelProblem &getUnderlyingProblem() const { return modelProblem; }

private:
  ES es;
  FEM fem;
  GFS gfs;

  BC bc;
  CC cc;
  ModelProblem modelProblem; // The underlying problem describing the PDE

  LOP lop;
  std::unique_ptr<AssembleWrapper<LOP>> wrapper;
  std::unique_ptr<GO> go;

  std::unique_ptr<Vec> x;
  std::unique_ptr<Vec> x0;             // solution vector
  std::unique_ptr<Vec> d;              // residual vector
  std::unique_ptr<Vec> dirichlet_mask; // vector with ones at the dirichlet dofs
  std::unique_ptr<Mat> As;
};
