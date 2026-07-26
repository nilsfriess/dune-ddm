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
 *
 * Supported PDEs:
 * - Convection-Diffusion (CG and DG)
 * - Linear Elasticity
 *
 */

#include <dune/pdelab/common/partitionviewentityset.hh>
#include <dune/pdelab/constraints/conforming.hh>
#include <dune/pdelab/constraints/noconstraints.hh>
#include <dune/pdelab/finiteelementmap/opbfem.hh>
#include <dune/pdelab/finiteelementmap/pkfem.hh>
#include <dune/pdelab/finiteelementmap/qkdg.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/pdelab/gridfunctionspace/vectorgridfunctionspace.hh>
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
 * The discretization type is controlled by the UseDG template parameter.
 *
 * @tparam GridView DUNE grid view type
 * @tparam ProblemParameters User-provided class defining PDE coefficients and boundary conditions
 *                           Must inherit from Dune::PDELab::ConvectionDiffusionModelProblem
 * @tparam SymmetricProblemParameters User-provided class defining the symmetric part of the operator.
 *                                    Defaults to ProblemParameters. This is used, e.g., in the MsGFEM
 *                                    coarse space to build the local spectral problems based on the
 *                                    symmetric part only.
 * @tparam IsSymmetic If true, the problem is considered symmetric; In this case, the GenericDDMProblem
 *                    will only assemble a problem using the SymmetricLocalOperator and SymmetricModelProblem.
 * @tparam UseDG If true, use DG discretization; if false, use CG (default: false)
 * @tparam degree Polynomial degree of finite elements (default: 1)
 * @tparam QkElements If ture, assumes a quadrilateral grid; if false, assumes a simplicial grid (default: true)
 */
template <class GridView, class ProblemParameters, class SymmetricProblemParameters = ProblemParameters, bool IsSymmetric = true, bool UseDG = false, int degree = 1, bool QkElements = true>
struct ConvectionDiffusionTraits {
  using RF = double; ///< Range field type
  using Grid = typename GridView::Grid;
  using DF = typename Grid::ctype; ///< Domain field type

  static constexpr int dim = GridView::dimension;
  static constexpr bool assembled_matrix_is_consistent = UseDG; ///< For CG, assume assembled matrix is additive, for DG it's consistent by construction because we assemble including ghost elements.

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

  /// Model problem corresponding to the symmetric part of the operator
  using SymmetricModelProblem = SymmetricProblemParameters;

  /// Boundary condition adapter
  // TODO: Do we need a different BCType for the symmetric part?
  using BCType = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<ModelProblem>;

  /// Local operator: ConvectionDiffusionDG for DG, ConvectionDiffusionFEM for CG
  using LocalOperator = std::conditional_t<UseDG, Dune::PDELab::ConvectionDiffusionDG<ModelProblem, FEM>, Dune::PDELab::ConvectionDiffusionFEM<ModelProblem, FEM>>;

  /// Symmetric local operator, same as above but for symmetric part only (if the PDE is already symmetric, this is the same as LocalOperator
  using SymmetricLocalOperator = std::conditional_t<UseDG, Dune::PDELab::ConvectionDiffusionDG<SymmetricModelProblem, FEM>, Dune::PDELab::ConvectionDiffusionFEM<SymmetricModelProblem, FEM>>;

  /// Convenience bool to check if the problem is symmetric
  static constexpr bool is_symmetric = IsSymmetric;

  /// Constraints: None for DG (no DOF coupling across elements), Conforming Dirichlet for CG
  using Constraints = std::conditional_t<UseDG, Dune::PDELab::NoConstraints, Dune::PDELab::ConformingDirichletConstraints>;

  using DirichletExtensionAdapter = Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter<ModelProblem>;

  using VBE = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>;
  using GFS = Dune::PDELab::GridFunctionSpace<EntitySet, FEM, Constraints, VBE>;

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

  static constexpr bool vector_valued = false;

  static constexpr bool is_dg = UseDG;
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

  /// Symmetric versions (for elliptic PDEs, same as non-symmetric)
  using SymmetricModelProblem = ModelProblem;
  using SymmetricLocalOperator = LocalOperator;

  /// Convenience bool to mark this as a symmetric problem
  static constexpr bool is_symmetric = true;

  using BCType = ModelProblem;

  using DirichletExtensionAdapter = Dune::PDELab::LinearElasticityDirichletExtensionAdapter<ModelProblem>;

  /// Constraints: Conforming Dirichlet constraints
  using Constraints = Dune::PDELab::ConformingDirichletConstraints;
  using ComponentVectorBackend = Dune::PDELab::ISTL::VectorBackend<>;
  using Mapper = Dune::PDELab::DefaultLeafOrderingTag;
  using OrderingTag = Dune::PDELab::LexicographicOrderingTag;
  using GFS = Dune::PDELab::VectorGridFunctionSpace<EntitySet, FEM, dim, Dune::PDELab::ISTL::VectorBackend<>, ComponentVectorBackend, Constraints, OrderingTag, Mapper>;
  using CC = typename GFS::template ConstraintsContainer<double>::Type;

  /// Default assembly parameters
  struct AssemblyDefaults {
    static constexpr int nonzeros_per_row = dim * dim * (2 * degree + 1) * (2 * degree + 1);
    static constexpr int quadrature_order = 2 * degree;
  };

  static constexpr bool vector_valued = true;

  /// A function to create the finite element map
  static std::unique_ptr<FEM> create_fem(const EntitySet& es) { return std::make_unique<FEM>(es); }

  static constexpr bool is_dg = false;
};
