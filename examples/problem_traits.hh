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

#include <dune/pdelab/common/partitionviewentityset.hh>
#include <dune/pdelab/constraints/conforming.hh>
#include <dune/pdelab/constraints/noconstraints.hh>
#include <dune/pdelab/finiteelementmap/opbfem.hh>
#include <dune/pdelab/finiteelementmap/pkfem.hh>
#include <dune/pdelab/finiteelementmap/qkdg.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/pdelab/localoperator/convectiondiffusiondg.hh>
#include <dune/pdelab/localoperator/convectiondiffusionfem.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>
#include <dune/pdelab/localoperator/linearelasticity.hh>
#include <dune/pdelab/localoperator/linearelasticityparameter.hh>
#include <type_traits>

/**
 * @brief Traits for Convection-Diffusion problems (scalar elliptic/parabolic PDEs)
 *
 * Supports both continuous Galerkin (CG) and discontinuous Galerkin (DG) discretizations.
 * The discretization type is controlled by the UseDG template parameter. Assumes a quadrilateral
 * mesh both for CG and DG.
 *
 * @tparam GridView DUNE grid view type
 * @tparam ProblemParameters User-provided class defining PDE coefficients and boundary conditions
 *                           Must inherit from Dune::PDELab::ConvectionDiffusionModelProblem
 * @tparam UseDG If true, use DG discretization; if false, use CG
 * @tparam degree Polynomial degree of finite elements (default: 1)
 */
template <class GridView, class ProblemParameters, bool UseDG = false, int degree = 1, bool QkElements = true>
struct ConvectionDiffusionTraits {
  using RF = double; ///< Range field type
  using Grid = typename GridView::Grid;
  using DF = typename Grid::ctype; ///< Domain field type

  static constexpr int dim = GridView::dimension;
  static constexpr bool assembled_matrix_is_consistent = UseDG; ///< For CG, assume assembled matrix is additive, for DG it's consistent by because we assemble including ghost elements

  /// Entity set: AllEntitySet for DG (includes ghost elements), OverlappingEntitySet for CG
  using EntitySet = std::conditional_t<UseDG, Dune::PDELab::AllEntitySet<GridView>, Dune::PDELab::OverlappingEntitySet<GridView>>;

  /// Finite element map:
  /// - DG + Qk elements (cubes/quads): QkDGLocalFiniteElementMap
  /// - DG + simplex elements: OPBLocalFiniteElementMap (orthogonal polynomials)
  /// - CG + Qk elements (cubes/quads): QkLocalFiniteElementMap
  /// - CG + simplex elements: PkLocalFiniteElementMap
  // clang-format off
  using FEM = std::conditional_t<
      UseDG,
      std::conditional_t<QkElements,
                         Dune::PDELab::QkDGLocalFiniteElementMap<DF, RF, degree, dim>,
                         Dune::PDELab::OPBLocalFiniteElementMap<DF, RF, degree, dim, Dune::GeometryType::simplex>>,
      std::conditional_t<QkElements, 
                         Dune::PDELab::QkLocalFiniteElementMap<EntitySet, DF, RF, degree>, 
                         Dune::PDELab::PkLocalFiniteElementMap<EntitySet, DF, RF, degree>>>;
  // clang-format on

  /// Model problem defining PDE coefficients (diffusion, convection, reaction, source, BC)
  using ModelProblem = ProblemParameters;

  /// Boundary condition adapter
  using BCType = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<ModelProblem>;

  /// Local operator: ConvectionDiffusionDG for DG, ConvectionDiffusionFEM for CG
  using LocalOperator = std::conditional_t<UseDG, Dune::PDELab::ConvectionDiffusionDG<ModelProblem, FEM>, Dune::PDELab::ConvectionDiffusionFEM<ModelProblem, FEM>>;

  /// Constraints: None for DG (no DOF coupling across elements), Conforming Dirichlet for CG
  using Constraints = std::conditional_t<UseDG, Dune::PDELab::NoConstraints, Dune::PDELab::ConformingDirichletConstraints>;

  /// Default assembly parameters
  struct AssemblyDefaults {
    static constexpr int nonzeros_per_row = (2 * degree + 1) * (2 * degree + 1); ///< For 2D
    static constexpr int quadrature_order = 2 * degree;                          ///< Sufficient for polynomial degree
  };

  /// A function to create the finite element map (needed because construction differs for CG vs DG)
  static std::unique_ptr<FEM> create_fem(const EntitySet& es)
  {
    if constexpr (UseDG) return std::make_unique<FEM>();
    else return std::make_unique<FEM>(es);
  }
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
  static constexpr int blocksize = dim;                         ///< Vector problem: one block per spatial dimension
  static constexpr bool assembled_matrix_is_consistent = false; ///< We assume that the assembled matrix is additive, i.e., entries that belong to multiple subdomains are not already summed

  /// Entity set: OverlappingEntitySet for CG assembly (for non-overlapping grids this means that ghost elements are not included)
  using EntitySet = Dune::PDELab::OverlappingEntitySet<GridView>;

  /// Finite element map: Pk for simplicial grids, Qk for structured grids
  /// Note: For now we use Pk which works on both simplex and cube elements
  using FEM = Dune::PDELab::PkLocalFiniteElementMap<EntitySet, DF, RF, degree>;

  /// Model problem defining material parameters
  using ModelProblem = ProblemParameters;

  /// Local operator for linear elasticity
  using LocalOperator = Dune::PDELab::LinearElasticity<ModelProblem>;

  /// Constraints: Conforming Dirichlet constraints
  using Constraints = Dune::PDELab::ConformingDirichletConstraints;

  /// Default assembly parameters
  struct AssemblyDefaults {
    static constexpr int nonzeros_per_row = dim * dim * (2 * degree + 1) * (2 * degree + 1);
    static constexpr int quadrature_order = 2 * degree;
  };

  /// A function to create the finite element map
  static std::unique_ptr<FEM> create_fem(const EntitySet& es) { return std::make_unique<FEM>(es); }
};
