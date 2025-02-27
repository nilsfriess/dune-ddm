#pragma once

#include <dune/common/version.hh>
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
#include <type_traits>

template <class GridView, class RF>
class Problem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
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
    if (center[0] < 1e-6 or center[1] < 1e-6) {
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
  using RF = double;
  using Grid = typename GridView::Grid;
  using DF = typename Grid::ctype;

  using CDProblem = Problem<GridView, RF>;
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<CDProblem>;

  using FEM = std::conditional_t<isYASPGrid<Grid>(),                                          // If YASP grid...
                                 Dune::PDELab::QkLocalFiniteElementMap<GridView, DF, RF, 2>,  // ... then use quadrilaterals
                                 Dune::PDELab::PkLocalFiniteElementMap<GridView, DF, RF, 2>>; // ... otherwise use triangles
  using LOP = Dune::PDELab::ConvectionDiffusionFEM<CDProblem, FEM>;

  using CON = Dune::PDELab::ConformingDirichletConstraints;
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;

public:
  using GFS = Dune::PDELab::GridFunctionSpace<GridView, FEM, CON, Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>>;

private:
  using CC = typename GFS::template ConstraintsContainer<RF>::Type;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, RF, RF, RF, CC, CC>;

public:
  using Vec = Dune::PDELab::Backend::Vector<GFS, RF>;
  using Mat = typename GO::Jacobian;
  using NativeMat = Dune::PDELab::Backend::Native<Mat>;

  PoissonProblem(const GridView &gv, const Dune::MPIHelper &helper) : fem(gv), gfs(gv, fem), x(std::make_unique<Vec>(gfs))
  {
    // Set up underlying PDE and boundary conditions
    CDProblem problem;
    BC bc(gv, problem);

    // Find out which dofs are constrained
    CC cc;
    Dune::PDELab::constraints(bc, gfs, cc);

    // Create solution vector and initialise with Dirichlet conditions
    Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter g(gv, problem);
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

    // Create local and global operators
    LOP lop(problem);
    GO go(gfs, cc, gfs, cc, lop, MBE(9));

    // Assemble residual
    d = std::make_unique<Vec>(gfs);
    go.residual(*x, *d);

    // Assemble stiffness matrix
    Mat As(go);
    go.jacobian(*x, As);
    A = std::make_shared<NativeMat>(Dune::PDELab::Backend::native(As));
  }

  Vec &getX() { return *x; }
  const Vec &getD() const { return *d; }
  const Vec &getDirichletMask() const { return *dirichlet_mask; }

  std::shared_ptr<NativeMat> getA() const { return A; }

  const GFS &getGFS() const { return gfs; }

private:
  FEM fem;
  GFS gfs;

  std::unique_ptr<Vec> x;              // solution vector
  std::unique_ptr<Vec> d;              // residual vector
  std::unique_ptr<Vec> dirichlet_mask; // vector with ones at the dirichlet dofs
  std::shared_ptr<NativeMat> A;        // stiffness matrix
};
