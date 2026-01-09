/**
 * @file poisson_problems.hh
 * @brief Problem parameter definitions for convection-diffusion examples
 *
 * This file defines the PDE coefficients, boundary conditions, and source terms
 * for various convection-diffusion test problems used in DDM examples.
 */

#pragma once

#include <cmath>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>

/**
 * @brief Simple Poisson problem (Laplace equation with source term)
 *
 * -∆u = f in Ω
 *  u = g on ∂Ω
 *
 * With f = 1.0 (constant source) and g = 0 (homogeneous Dirichlet BC).
 */
template <class GridView, class RF>
class SimplePoissonProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  // Diffusion tensor: identity matrix (isotropic diffusion)
  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++)
      for (std::size_t j = 0; j < Traits::dimDomain; j++) I[i][j] = (i == j) ? 1.0 : 0.0;
    return I;
  }

  // Source term: constant
  typename Traits::RangeFieldType f(const typename Traits::ElementType&, const typename Traits::DomainType&) const { return 1.0; }

  // Dirichlet boundary value: zero
  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return 0.0; }

  // Boundary condition type: Dirichlet everywhere
  BC bctype(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const { return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet; }
};

/**
 * @brief Heterogeneous diffusion problem with vertical "beams"
 *
 * This creates a challenging problem for domain decomposition with
 * strong coefficient variation (1 to 1e6).
 */
template <class GridView, class RF>
class PoissonBeamsProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    auto xg = e.geometry().global(x);
    const auto width = 0.02;

    const RF small_coeff = 1.0;
    const RF large_coeff = 1e6;
    RF coeff = small_coeff;

    int num_beams = 8;
    double space_between = 0.1;

    if (xg[1] <= 0.95) {
      for (int i = 1; i <= num_beams; ++i)
        if (xg[0] >= i * space_between && xg[0] <= i * space_between + width) coeff = large_coeff;

      if (xg[1] >= 0.95 - width) {
        for (int i = 1; i <= num_beams; ++i)
          if (xg[0] >= i * space_between && xg[0] <= i * space_between + 3 * width) coeff = large_coeff;
      }

      if (xg[1] >= 0.95 - 2 * width) {
        for (int i = 1; i <= num_beams; ++i)
          if (xg[0] >= i * space_between + 2 * width && xg[0] <= i * space_between + 3 * width) coeff = large_coeff;
      }
    }

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++)
      for (std::size_t j = 0; j < Traits::dimDomain; j++) I[i][j] = (i == j) ? coeff : 0.0;
    return I;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType&, const typename Traits::DomainType&) const { return 1.0; }

  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return 0.0; }

  BC bctype(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const { return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet; }
};

/**
 * @brief Islands problem with complex heterogeneous coefficient patterns
 *
 * This is the exact same problem as IslandsModelProblem in the original poisson.hh.
 * Features multiple high-conductivity regions with different patterns and variable coefficients.
 */
template <class GridView, class RF>
class IslandsProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType&) const
  {
    auto xg = e.geometry().center();

    int ix = std::floor(15.0 * xg[0]);
    int iy = std::floor(15.0 * xg[1]);
    auto x = xg[0];
    auto y = xg[1];

    double kappa = 1.0;

    // Diagonal band with variable coefficient
    if (x > 0.3 && x < 0.9 && y > 0.6 - (x - 0.3) / 6 && y < 0.8 - (x - 0.3) / 6) kappa = std::pow(10.0, 5.0) * (x + y) * 10.0;

    // Lower left triangular region with variable coefficient
    if (x > 0.1 && x < 0.5 && y > 0.1 + x && y < 0.25 + x) kappa = std::pow(10.0, 5.0) * (1.0 + 7.0 * y);

    // Lower right diagonal band with constant high coefficient
    if (x > 0.5 && x < 0.9 && y > 0.15 - (x - 0.5) * 0.25 && y < 0.35 - (x - 0.5) * 0.25) kappa = std::pow(10.0, 5.0) * 2.5;

    // Checkerboard pattern with variable coefficient
    if (ix % 2 == 0 && iy % 2 == 0) kappa = std::pow(10.0, 5.0) * (1.0 + ix + iy);

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++)
      for (std::size_t j = 0; j < Traits::dimDomain; j++) I[i][j] = (i == j) ? kappa : 0.0;
    return I;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType&, const typename Traits::DomainType&) const { return 0.0; }

  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, const typename Traits::DomainType& xlocal) const
  {
    auto xglobal = e.geometry().global(xlocal);
    return 1.0 - xglobal[0];
  }

  BC bctype(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    auto xglobal = is.geometry().global(x);
    if (xglobal[0] < 1e-6) return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    if (xglobal[0] > 1.0 - 1e-6) return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
  }
};
