#ifndef DUNE_DDM_CONVECTIONDIFFUSIONDG_HH
#define DUNE_DDM_CONVECTIONDIFFUSIONDG_HH

#include <cmath>
#include <dune/common/fvector.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>

/** @brief Heterogeneous convection-diffusion problem with interesting solution features
 *
 *  This problem models a convection-diffusion equation with:
 *  - Heterogeneous diffusion coefficient (varying by region)
 *  - Moderate convection field creating transport effects
 *  - Dirichlet boundary conditions creating boundary layers
 *  - Source term producing internal features
 *
 *  The setup creates a solution with:
 *  - Smooth variations in regions with high diffusion
 *  - Sharper gradients in low-diffusion regions
 *  - Transport effects from the convection field
 *  - Interesting interaction between source and boundaries
 */
template <typename GV, typename RF>
class ConvectionDiffusionDGProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF> {
public:
  using Traits = Dune::PDELab::ConvectionDiffusionParameterTraits<GV, RF>;

  ConvectionDiffusionDGProblem() = default;

  //! Tensor diffusion coefficient (here scalar, but could be anisotropic)
  auto A(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType I;

    RF coeff = 0.01;
    auto xg = e.geometry().global(x);
    if (xg[0] > 0.3 && xg[0] < 0.4 && xg[1] > 0.3 && xg[1] < 0.4) coeff = 1e5;

    for (std::size_t i = 0; i < Traits::dimDomain; i++) I[i][i] = coeff;
    return I;
  }

  //! Velocity field (convection)
  auto b(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::RangeType v(0.0);
    v[0] = 1.0;
    v[1] = 1.0;
    return v;
  }

  //! Sink term (here zero, could be used for decay)
  auto c(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return 0.0; }

  //! Source term
  auto f(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    const auto global = e.geometry().global(x);

    // Single Gaussian source at the bottom left
    auto sx = 0.2;
    auto sy = 0.2;
    auto r2 = (global[0] - sx) * (global[0] - sx) + (global[1] - sy) * (global[1] - sy);
    return 100.0 * std::exp(-r2 / (0.05 * 0.05));
  }

  //! Boundary condition type
  auto bctype(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    auto global = is.geometry().global(x);
    if (global[0] > 1.0 - 1e-6 || global[1] > 1.0 - 1e-6)
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Outflow;
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
  }

  //! Dirichlet boundary condition value
  auto g(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    return 0.0;
  }

  //! Neumann boundary condition (not used here, but required by interface)
  auto j(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const { return 0.0; }

  //! Outflow boundary condition (not used here)
  auto o(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const { return 0.0; }
};

#endif // DUNE_DDM_CONVECTIONDIFFUSIONDG_HH
