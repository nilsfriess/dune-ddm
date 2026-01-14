#pragma once

/**
 * @file generic_ddm_problem.hh
 * @brief Generic PDELab problem wrapper for domain decomposition methods
 *
 * This file provides a template-based wrapper around PDELab that handles
 * the complete setup and assembly workflow for domain decomposition methods.
 */

#include "assemblewrapper.hh"
#include "dune/ddm/datahandles.hh"
#include "dune/ddm/logger.hh"
#include "pdelab_helper.hh"

#include <cstddef>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/pdelab/backend/interface.hh>
#include <dune/pdelab/backend/istl/bcrsmatrixbackend.hh>
#include <dune/pdelab/backend/istl/vector.hh>
#include <dune/pdelab/constraints/common/constraints.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include <dune/pdelab/gridfunctionspace/interpolate.hh>
#include <dune/pdelab/gridoperator/gridoperator.hh>
#include <memory>
#include <vector>

/**
 * @brief Generic problem class for PDELab+DDM workflows
 *
 * This class handles the complete workflow from grid function space setup
 * to matrix assembly for domain decomposition methods. It is parameterized
 * by a Traits class that provides all PDE-specific type information.
 *
 *
 * @tparam GridView DUNE grid view type
 * @tparam Traits Problem traits providing FEM, LocalOperator, Constraints, etc.
 *                Must follow the interface defined in problem_traits.hh
 *
 * Example usage:
 * @code
 * using Traits = ConvectionDiffusionTraits<GridView, MyProblemParams, true>;
 * GenericDDMProblem<GridView, Traits> problem(gv, helper);
 * @endcode
 */
template <class GridView, class TraitsT>
class GenericDDMProblem {
public:
  using Grid = typename GridView::Grid;
  using DF = typename Grid::ctype;

  // Import types from traits
  using Traits = TraitsT;
  using RF = typename Traits::RF;
  using EntitySet = typename Traits::EntitySet;
  using FEM = typename Traits::FEM;
  using ModelProblem = typename Traits::ModelProblem;
  using SymmetricModelProblem = typename Traits::SymmetricModelProblem;
  using BCType = typename Traits::BCType;
  using LocalOperator = typename Traits::LocalOperator;
  using SymmetricLocalOperator = typename Traits::SymmetricLocalOperator;
  using Constraints = typename Traits::Constraints;
  constexpr static bool is_symmetric = Traits::is_symmetric;
  static_assert((is_symmetric && std::is_same_v<LocalOperator, SymmetricLocalOperator>) or !is_symmetric,
                "If the problem is symmetric, the LocalOperator and SymmetricLocalOperator must be the same types");

  // PDELab types
  using VBE = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>;
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using GFS = Dune::PDELab::GridFunctionSpace<EntitySet, FEM, Constraints, VBE>;
  using CC = typename GFS::template ConstraintsContainer<RF>::Type;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, LocalOperator, MBE, RF, RF, RF, CC, CC>;
  using SymmetricGO = Dune::PDELab::GridOperator<GFS, GFS, AssembleWrapper<SymmetricLocalOperator>, MBE, RF, RF, RF, CC, CC>;

  // Vector and matrix types
  using Vec = Dune::PDELab::Backend::Vector<GFS, RF>;
  using Mat = typename GO::Jacobian;
  static_assert(std::is_same_v<Mat, typename SymmetricGO::Jacobian>);
  using NativeMat = Dune::PDELab::Backend::Native<Mat>;
  using NativeVec = Dune::PDELab::Backend::Native<Vec>;

  /**
   * @brief Constructor: sets up complete finite element discretization
   *
   * Performs the following steps:
   * 1. Creates entity set and finite element map
   * 2. Sets up grid function space with constraints
   * 3. Initializes solution vector with boundary conditions
   * 4. Creates Dirichlet constraint mask
   * 5. Assembles grid operator, sparsity pattern, and residual
   * 6. Communicates residual in parallel if needed
   *
   * @note This variant is for symmetric problems and is only chosen if Traits::is_symmetric == true.
   *       It only uses the SymmetricLocalOperator and SymmetricModelProblem.
   *
   * @param gv Grid view defining the computational domain
   * @param helper MPI helper for parallel operations
   * @param model_problem Instance of the problem parameter class (coefficients, BC)
   */
  template <bool symm = is_symmetric, std::enable_if_t<symm, bool> = true>
  GenericDDMProblem(const GridView& gv, const Dune::MPIHelper& helper, std::shared_ptr<SymmetricModelProblem> symm_model_problem = std::make_shared<SymmetricModelProblem>())
      : es(gv)
      , symmetricModelProblem(std::move(symm_model_problem))
      , bc(gv, *this->symmetricModelProblem)
      , fem(Traits::create_fem(es))
      , gfs(std::make_shared<GFS>(es, *fem))
      , symm_lop(*this->symmetricModelProblem)
      , wrapper(&symm_lop)
      , x(std::make_unique<Vec>(*gfs, 0.0))
      , x0(std::make_unique<Vec>(*gfs, 0.0))
      , d(std::make_unique<Vec>(*gfs, 0.0))
      , dirichlet_mask(std::make_unique<Vec>(*gfs, 0))
  {
    using Dune::PDELab::Backend::native;

    gfs->name("Solution");

    // Interpolate Dirichlet boundary conditions
    cc.clear();
    Dune::PDELab::constraints(bc, *gfs, cc);

    // For ConvectionDiffusion problems, use ConvectionDiffusionDirichletExtensionAdapter
    // Note: It expects GridView, not EntitySet
    Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter g(gv, *this->symmetricModelProblem);
    Dune::PDELab::interpolate(g, *gfs, *x);

    // Set Dirichlet mask
    Dune::PDELab::set_constrained_dofs(cc, 1.0, *dirichlet_mask);

    // Create grid operator with appropriate matrix nonzeros estimate
    const int nz = Traits::AssemblyDefaults::nonzeros_per_row;
    symm_go = std::make_unique<SymmetricGO>(*gfs, cc, *gfs, cc, wrapper, MBE(nz));

    logger::debug("Assembling sparsity pattern");
    As = std::make_unique<Mat>(*symm_go);

    logger::debug("Assembling residual");
    symm_go->residual(*x, *d);

    *x0 = *x;

    // Communicate residual in parallel
    if (helper.size() > 1) {
      Dune::PDELab::AddDataHandle adddhd(*gfs, *d);
      gfs->gridView().communicate(adddhd, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);
    }
  }

  /**
   * @brief Constructor: sets up complete finite element discretization
   *
   * Performs the following steps:
   * 1. Creates entity set and finite element map
   * 2. Sets up grid function space with constraints
   * 3. Initializes solution vector with boundary conditions
   * 4. Creates Dirichlet constraint mask
   * 5. Assembles grid operator, sparsity pattern, and residual
   * 6. Communicates residual in parallel if needed
   *
   * @note This variant is for non-symmetric problems and is only chosen if Traits::is_symmetric == false.
   *
   * @param gv Grid view defining the computational domain
   * @param helper MPI helper for parallel operations
   * @param model_problem Instance of the problem parameter class (coefficients, BC)
   * @param symm_model_problem Instance of the problem parameter class corresponding to the elliptic part
   */
  template <bool symm = is_symmetric, std::enable_if_t<!symm, bool> = true>
  GenericDDMProblem(const GridView& gv, const Dune::MPIHelper& helper, std::shared_ptr<ModelProblem> model_problem = std::make_shared<ModelProblem>(),
                    std::shared_ptr<SymmetricModelProblem> symm_model_problem = std::make_shared<SymmetricModelProblem>())
      : es(gv)
      , modelProblem(std::move(model_problem))
      , symmetricModelProblem(std::move(symm_model_problem))
      , bc(gv, *this->modelProblem)
      , fem(Traits::create_fem(es))
      , gfs(std::make_shared<GFS>(es, *fem))
      , lop(std::make_unique<LocalOperator>(*this->modelProblem))
      , symm_lop(*this->symmetricModelProblem)
      , wrapper(&symm_lop)
      , x(std::make_unique<Vec>(*gfs, 0.0))
      , x0(std::make_unique<Vec>(*gfs, 0.0))
      , d(std::make_unique<Vec>(*gfs, 0.0))
      , dirichlet_mask(std::make_unique<Vec>(*gfs, 0))
  {
    using Dune::PDELab::Backend::native;

    gfs->name("Solution");

    // Interpolate Dirichlet boundary conditions
    cc.clear();
    Dune::PDELab::constraints(bc, *gfs, cc);

    // For ConvectionDiffusion problems, use ConvectionDiffusionDirichletExtensionAdapter
    // Note: It expects GridView, not EntitySet
    Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter g(gv, *this->modelProblem);
    Dune::PDELab::interpolate(g, *gfs, *x);

    // Set Dirichlet mask
    Dune::PDELab::set_constrained_dofs(cc, 1.0, *dirichlet_mask);

    // Create grid operator with appropriate matrix nonzeros estimate
    const int nz = Traits::AssemblyDefaults::nonzeros_per_row;
    go = std::make_unique<GO>(*gfs, cc, *gfs, cc, *lop, MBE(nz));
    symm_go = std::make_unique<SymmetricGO>(*gfs, cc, *gfs, cc, wrapper, MBE(nz));

    logger::debug("Assembling sparsity pattern");
    As = std::make_unique<Mat>(*go);

    logger::debug("Assembling residual");
    go->residual(*x, *d);

    *x0 = *x;

    // Communicate residual in parallel
    if (helper.size() > 1) {
      Dune::PDELab::AddDataHandle adddhd(*gfs, *d);
      gfs->gridView().communicate(adddhd, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);
    }
  }

  /**
   * @brief Assemble overlapping Dirichlet and Neumann matrices for DDM coarse spaces
   *
   * This function assembles three matrices needed for domain decomposition:
   * - Dirichlet matrix: with Dirichlet BC at subdomain boundary
   * - First Neumann matrix: with Neumann BC (region controlled by first_neumann_region)
   * - Second Neumann matrix: restricted Neumann matrix (region controlled by second_neumann_region)
   *
   * @tparam Communication Parallel communication type
   * @param ovlp_comm Overlapping communication object
   * @param first_neumann_region Region where first Neumann matrix is defined
   * @param second_neumann_region Region where second Neumann matrix is defined
   * @param overlap Overlap width
   * @param neumann_size_as_dirichlet If true, Neumann matrices have same size as Dirichlet
   * @param novlp_comm Non-overlapping communication (optional, for Neumann corrections)
   */
  template <class Communication>
  void assemble_overlapping_matrices(Communication& ovlp_comm, NeumannRegion first_neumann_region, NeumannRegion second_neumann_region, int overlap, bool neumann_size_as_dirichlet = true,
                                     const Communication* novlp_comm = nullptr)
  {
    if constexpr (is_symmetric) {
      auto [matrices, dirichlet_mask_ovlp_, neumann_region_to_subdomain_] =
          ::assemble_overlapping_matrices(*As, *x, *symm_go, Dune::PDELab::Backend::native(*dirichlet_mask), ovlp_comm, first_neumann_region, second_neumann_region, overlap, neumann_size_as_dirichlet,
                                          Traits::assembled_matrix_is_consistent, novlp_comm);

      A_dir = std::move(matrices.A_dir);
      A_neu = std::move(matrices.A_neu);
      B_neu = std::move(matrices.B_neu);
      dirichlet_mask_ovlp = std::move(*dirichlet_mask_ovlp_);
      neumann_region_to_subdomain = std::move(neumann_region_to_subdomain_);
    }
    else {
      // In the non-symmetric case, we only need the Neumann matrices for the elliptic part of the PDE
      auto [matrices, dirichlet_mask_ovlp_, neumann_region_to_subdomain_] =
          ::assemble_overlapping_matrices(*As, *x, *symm_go, Dune::PDELab::Backend::native(*dirichlet_mask), ovlp_comm, first_neumann_region, second_neumann_region, overlap, neumann_size_as_dirichlet,
                                          Traits::assembled_matrix_is_consistent, novlp_comm);

      A_neu = std::move(matrices.A_neu);
      B_neu = std::move(matrices.B_neu);
      dirichlet_mask_ovlp = std::move(*dirichlet_mask_ovlp_);
      neumann_region_to_subdomain = std::move(neumann_region_to_subdomain_);

      // Now we have the Neumann matrix corresponding to the symmetric part, we still need the Dirichlet matrix corresponding to the actual problem
      assemble_dirichlet_matrix_only(ovlp_comm, novlp_comm);
    }
  }

  /**
   * @brief Assemble only the overlapping Dirichlet matrix
   *
   * Simplified version for coarse spaces that don't need Neumann matrices (e.g., POU).
   *
   * @tparam Communication Parallel communication type
   * @param comm Overlapping communication
   */
  template <class Communication>
  void assemble_dirichlet_matrix_only(const Communication& comm, const Communication* novlp_comm = nullptr)
  {
    using Dune::PDELab::Backend::native;
    logger::debug("Assembling overlapping Dirichlet matrix");
    jacobian(novlp_comm);

    // Create communicator on the overlapping index set
    typename Communication::AllSet allset;
    Dune::Interface interface_ext;
    interface_ext.build(comm.remoteIndices(), allset, allset);
    Dune::VariableSizeCommunicator varcomm(interface_ext);

    // Communicate matrix structure and values
    CreateMatrixDataHandle cmdh(native(*As), comm.indexSet());
    varcomm.forward(cmdh);
    A_dir = std::make_shared<NativeMat>(cmdh.getOverlappingMatrix());

    AddMatrixDataHandle amdh(native(*As), *A_dir, comm.indexSet());
    varcomm.forward(amdh);

    // Set up Dirichlet mask on overlapping subdomain
    dirichlet_mask_ovlp = NativeVec(A_dir->N());
    dirichlet_mask_ovlp = 0;
    for (std::size_t i = 0; i < dirichlet_mask->N(); ++i) dirichlet_mask_ovlp[i] = native(*dirichlet_mask)[i];
    comm.addOwnerCopyToAll(dirichlet_mask_ovlp, dirichlet_mask_ovlp);

    // Eliminate Dirichlet dofs and subdomain boundary dofs symmetrically
    IdentifyBoundaryDataHandle ibdh(*A_dir, comm.indexSet());
    varcomm.forward(ibdh);
    const auto& boundary_mask = ibdh.get_boundary_mask();

    eliminate_dirichlet(*A_dir, dirichlet_mask_ovlp);
    // eliminate_dirichlet(*A_dir, boundary_mask, false); // false = don't eliminate symmetrically
  }

  /**
   * @brief Assemble the Jacobian matrix
   *
   * Assembles the stiffness matrix and eliminates Dirichlet DOFs symmetrically.
   */
  template <class Communication = std::nullptr_t>
  void jacobian(const Communication* novlp_comm = nullptr)
  {
    using Dune::PDELab::Backend::native;
    // Simple assembly without Neumann corrections
    wrapper.reset_masks();
    As = std::make_unique<Mat>(*go);
    go->jacobian(*x, *As);

    if (Traits::assembled_matrix_is_consistent) {
      if (novlp_comm == nullptr) DUNE_THROW(Dune::Exception, "Need non-overlapping communicator for DG assembly");
      make_additive(*As, *novlp_comm);
    }

    eliminate_dirichlet(native(*As), native(*dirichlet_mask));
  }

  ///@{
  /** @name Accessors for vectors and matrices */

  Vec& getXVec() { return *x0; }
  NativeVec& getX() { return Dune::PDELab::Backend::native(*x0); }
  NativeVec& getD() const { return Dune::PDELab::Backend::native(*d); }
  Vec& getDVec() const { return *d; }
  const Vec& getDirichletMask() const { return *dirichlet_mask; }
  const NativeVec& get_overlapping_dirichlet_mask() const { return dirichlet_mask_ovlp; }
  const Mat& getA() const { return *As; }
  ///@}

  ///@{
  /** @name Accessors for PDELab objects */

  std::shared_ptr<GFS> getGFS() const { return gfs; }
  const EntitySet& getEntitySet() const { return es; }
  const ModelProblem& getUnderlyingProblem() const { return *modelProblem; }
  ///@}

  ///@{
  /** @name Accessors for DDM matrices */

  std::shared_ptr<NativeMat> get_dirichlet_matrix() { return A_dir; }
  std::shared_ptr<NativeMat> get_first_neumann_matrix() { return A_neu; }
  std::shared_ptr<NativeMat> get_second_neumann_matrix() { return B_neu; }
  const std::vector<std::size_t>& get_neumann_region_to_subdomain() const { return neumann_region_to_subdomain; }
  ///@}

private:
  ///@{
  /** @name Core DUNE/PDELab objects */
  EntitySet es;
  std::shared_ptr<ModelProblem> modelProblem;
  std::shared_ptr<SymmetricModelProblem> symmetricModelProblem{nullptr};
  BCType bc;
  CC cc;
  std::unique_ptr<FEM> fem;
  std::shared_ptr<GFS> gfs;
  std::unique_ptr<LocalOperator> lop; // Pointer because it does not have a default constructor
  SymmetricLocalOperator symm_lop;
  AssembleWrapper<SymmetricLocalOperator> wrapper; // Some explanations are in order here: The wrapper object only exists for the SymmetricLocalOperator.
                                                   // This is because we only ever need to assemble the Neumann matrix for the elliptic part of the PDE,
                                                   // and for PDEs that are already elliptic, we know that SymmetricLocalOperator == LocalOperator.

  std::unique_ptr<GO> go;
  std::unique_ptr<SymmetricGO> symm_go;
  ///@}

  ///@{
  /** @name Solution vectors and system matrix */
  std::unique_ptr<Vec> x;              ///< Current solution
  std::unique_ptr<Vec> x0;             ///< Initial solution (with BC)
  std::unique_ptr<Vec> d;              ///< Residual vector
  std::unique_ptr<Vec> dirichlet_mask; ///< Dirichlet constraint mask
  std::unique_ptr<Mat> As;             ///< System matrix
  ///@}

  ///@{
  /** @name DDM matrices and masks */
  std::shared_ptr<NativeMat> A_dir;                     ///< Overlapping Dirichlet matrix
  std::shared_ptr<NativeMat> A_neu;                     ///< First Neumann matrix
  std::shared_ptr<NativeMat> B_neu;                     ///< Second/restricted Neumann matrix
  NativeVec dirichlet_mask_ovlp;                        ///< Dirichlet mask on overlapping subdomain
  std::vector<std::size_t> neumann_region_to_subdomain; ///< Index mapping for ring coarse spaces
  ///@}
};
