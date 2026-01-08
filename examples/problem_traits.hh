#pragma once

/**
 * @file problem_traits.hh
 * @brief Trait definitions for different PDE types in the unified DDM framework
 * 
 * This file defines trait structs that specify all the type information needed
 * to set up a PDELab problem for domain decomposition methods. Each trait struct
 * provides:
 * - Finite element map (FEM) type
 * - Local operator type  
 * - Problem parameter interface
 * - Entity set type (for CG vs DG assembly)
 * - Constraints type
 * - Block size (scalar=1, vector=dim)
 * 
 * Supported PDEs:
 * - Convection-Diffusion (CG and DG)
 * - Linear Elasticity
 * 
 * Part of the unified PDELab example framework.
 */

#include <type_traits>

#include <dune/pdelab/constraints/conforming.hh>
#include <dune/pdelab/constraints/noconstraints.hh>
#include <dune/pdelab/finiteelementmap/pkfem.hh>
#include <dune/pdelab/finiteelementmap/qkdg.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/pdelab/common/partitionviewentityset.hh>
#include <dune/pdelab/localoperator/convectiondiffusiondg.hh>
#include <dune/pdelab/localoperator/convectiondiffusionfem.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>
#include <dune/pdelab/localoperator/linearelasticity.hh>
#include <dune/pdelab/localoperator/linearelasticityparameter.hh>

/**
 * @brief Traits for Convection-Diffusion problems (scalar elliptic/parabolic PDEs)
 * 
 * Supports both continuous Galerkin (CG) and discontinuous Galerkin (DG) discretizations.
 * The discretization type is controlled by the UseDG template parameter.
 * 
 * @tparam GridView DUNE grid view type
 * @tparam ProblemParameters User-provided class defining PDE coefficients and boundary conditions
 *                           Must inherit from Dune::PDELab::ConvectionDiffusionModelProblem
 * @tparam UseDG If true, use DG discretization; if false, use CG
 * @tparam degree Polynomial degree of finite elements (default: 1)
 */
template <class GridView, class ProblemParameters, bool UseDG = false, int degree = 1>
struct ConvectionDiffusionTraits {
  using RF = double;  ///< Range field type
  using Grid = typename GridView::Grid;
  using DF = typename Grid::ctype;  ///< Domain field type
  
  static constexpr int dim = GridView::dimension;
  static constexpr int blocksize = 1;  ///< Scalar problem
  static constexpr bool is_dg = UseDG;
  
  /// Entity set: AllEntitySet for DG (includes ghost elements), OverlappingEntitySet for CG
  using EntitySet = std::conditional_t<UseDG,
      Dune::PDELab::AllEntitySet<GridView>,
      Dune::PDELab::OverlappingEntitySet<GridView>>;
  
  /// Finite element map: QkDG for DG, Qk for CG (works for both structured and unstructured grids)
  using FEM = std::conditional_t<UseDG,
      Dune::PDELab::QkDGLocalFiniteElementMap<DF, RF, degree, dim>,
      Dune::PDELab::QkLocalFiniteElementMap<EntitySet, DF, RF, degree>>;
  
  /// Model problem defining PDE coefficients (diffusion, convection, reaction, source, BC)
  using ModelProblem = ProblemParameters;
  
  /// Boundary condition adapter
  using BCType = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<ModelProblem>;
  
  /// Local operator: ConvectionDiffusionDG for DG, ConvectionDiffusionFEM for CG
  using LocalOperator = std::conditional_t<UseDG,
      Dune::PDELab::ConvectionDiffusionDG<ModelProblem, FEM>,
      Dune::PDELab::ConvectionDiffusionFEM<ModelProblem, FEM>>;
  
  /// Constraints: None for DG (no DOF coupling across elements), Conforming Dirichlet for CG
  using Constraints = std::conditional_t<UseDG,
      Dune::PDELab::NoConstraints,
      Dune::PDELab::ConformingDirichletConstraints>;
  
  /// Information about coarse space compatibility
  struct CoarseSpaceInfo {
    static constexpr bool supports_geneo = true;     ///< GenEO requires symmetric operator
    static constexpr bool supports_msgfem = true;    ///< MsGFEM works with general operators
    static constexpr bool supports_pou = true;       ///< POU always works
  };
  
  /// Default assembly parameters
  struct AssemblyDefaults {
    static constexpr int nonzeros_per_row = (2 * degree + 1) * (2 * degree + 1);  ///< For 2D
    static constexpr int quadrature_order = 2 * degree;  ///< Sufficient for polynomial degree
  };
};

/**
 * @brief Traits for Linear Elasticity problems (vector-valued elliptic PDE)
 * 
 * Solves the linear elasticity equations for small deformations.
 * This is a vector-valued problem with dim unknowns per DOF.
 * 
 * @tparam GridView DUNE grid view type  
 * @tparam ProblemParameters User-provided class defining material parameters (lambda, mu)
 *                           Must inherit from Dune::PDELab::LinearElasticityParameterInterface
 * @tparam degree Polynomial degree of finite elements (default: 1)
 */
template <class GridView, class ProblemParameters, int degree = 1>
struct LinearElasticityTraits {
  using RF = double;
  using Grid = typename GridView::Grid;
  using DF = typename Grid::ctype;
  
  static constexpr int dim = GridView::dimension;
  static constexpr int blocksize = dim;  ///< Vector problem: one block per spatial dimension
  static constexpr bool is_dg = false;   ///< Linear elasticity typically uses CG
  
  /// Entity set: AllEntitySet to include all elements
  using EntitySet = Dune::PDELab::AllEntitySet<GridView>;
  
  /// Finite element map: Pk for simplicial grids, Qk for structured grids
  /// Note: For now we use Pk which works on both simplex and cube elements
  using FEM = Dune::PDELab::PkLocalFiniteElementMap<EntitySet, DF, RF, degree>;
  
  /// Model problem defining material parameters
  using ModelProblem = ProblemParameters;
  
  /// Local operator for linear elasticity
  using LocalOperator = Dune::PDELab::LinearElasticity<ModelProblem>;
  
  /// Constraints: Conforming Dirichlet constraints
  using Constraints = Dune::PDELab::ConformingDirichletConstraints;
  
  /// Information about coarse space compatibility
  struct CoarseSpaceInfo {
    static constexpr bool supports_geneo = true;     ///< Symmetric positive definite operator
    static constexpr bool supports_msgfem = true;    ///< Works well
    static constexpr bool supports_pou = true;       ///< Always works
  };
  
  /// Default assembly parameters  
  struct AssemblyDefaults {
    static constexpr int nonzeros_per_row = dim * dim * (2 * degree + 1) * (2 * degree + 1);
    static constexpr int quadrature_order = 2 * degree;
  };
};

/**
 * @brief Helper to check if a traits class supports a specific coarse space type
 * 
 * Usage:
 * @code
 * if constexpr (SupportsCoarseSpace<MyTraits, CoarseSpaceType::GenEO>) {
 *   // Can use GenEO
 * }
 * @endcode
 */
enum class CoarseSpaceType : std::uint8_t {
  GenEO,
  MsGFEM,
  POU
};

template <class Traits, CoarseSpaceType CSType>
constexpr bool SupportsCoarseSpace()
{
  if constexpr (CSType == CoarseSpaceType::GenEO)
    return Traits::CoarseSpaceInfo::supports_geneo;
  else if constexpr (CSType == CoarseSpaceType::MsGFEM)
    return Traits::CoarseSpaceInfo::supports_msgfem;
  else if constexpr (CSType == CoarseSpaceType::POU)
    return Traits::CoarseSpaceInfo::supports_pou;
  return false;
}
