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
  typename Traits::PermTensorType A(const typename Traits::ElementType& e, 
                                     const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++)
      for (std::size_t j = 0; j < Traits::dimDomain; j++) 
        I[i][j] = (i == j) ? 1.0 : 0.0;
    return I;
  }

  // Source term: constant
  typename Traits::RangeFieldType f(const typename Traits::ElementType&, 
                                     const typename Traits::DomainType&) const 
  { 
    return 1.0; 
  }

  // Dirichlet boundary value: zero
  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, 
                                     const typename Traits::DomainType& x) const
  {
    return 0.0;
  }

  // Boundary condition type: Dirichlet everywhere
  BC bctype(const typename Traits::IntersectionType& is, 
            const typename Traits::IntersectionDomainType& x) const
  {
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
  }
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

  typename Traits::PermTensorType A(const typename Traits::ElementType& e, 
                                     const typename Traits::DomainType& x) const
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
        if (xg[0] >= i * space_between && xg[0] <= i * space_between + width) 
          coeff = large_coeff;

      if (xg[1] >= 0.95 - width) {
        for (int i = 1; i <= num_beams; ++i)
          if (xg[0] >= i * space_between && xg[0] <= i * space_between + 3 * width) 
            coeff = large_coeff;
      }

      if (xg[1] >= 0.95 - 2 * width) {
        for (int i = 1; i <= num_beams; ++i)
          if (xg[0] >= i * space_between + 2 * width && xg[0] <= i * space_between + 3 * width) 
            coeff = large_coeff;
      }
    }

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++)
      for (std::size_t j = 0; j < Traits::dimDomain; j++) 
        I[i][j] = (i == j) ? coeff : 0.0;
    return I;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType&, 
                                     const typename Traits::DomainType&) const 
  { 
    return 1.0; 
  }

  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, 
                                     const typename Traits::DomainType& x) const
  {
    return 0.0;
  }

  BC bctype(const typename Traits::IntersectionType& is, 
            const typename Traits::IntersectionDomainType& x) const
  {
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
  }
};

/**
 * @brief Islands problem with complex heterogeneous coefficient patterns
 * 
 * Features diagonal strips, triangular regions, and checkerboard patterns.
 */
template <class GridView, class RF>
class IslandsProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  typename Traits::PermTensorType A(const typename Traits::ElementType& e, 
                                     const typename Traits::DomainType&) const
  {
    auto xg = e.geometry().center();

    int ix = std::floor(15.0 * xg[0]);
    int iy = std::floor(15.0 * xg[1]);
    auto x = xg[0];
    auto y = xg[1];

    double kappa = 1.0;
    
    // Diagonal strip pattern
    if (std::abs(x - y) < 0.1) 
      kappa = 1e5 * (1.0 + x);
    
    // Triangular high-conductivity region  
    if (x < 0.5 && y < 0.5 && y < x)
      kappa = 1e5;
      
    // Checkerboard pattern in upper right
    if (x > 0.5 && y > 0.5)
      if ((ix + iy) % 2 == 0)
        kappa = 1e4;

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++)
      for (std::size_t j = 0; j < Traits::dimDomain; j++) 
        I[i][j] = (i == j) ? kappa : 0.0;
    return I;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType&, 
                                     const typename Traits::DomainType&) const 
  { 
    return 1.0; 
  }

  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, 
                                     const typename Traits::DomainType& x) const
  {
    return 0.0;
  }

  BC bctype(const typename Traits::IntersectionType& is, 
            const typename Traits::IntersectionDomainType& x) const
  {
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
  }
};
