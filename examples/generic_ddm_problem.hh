#pragma once

/**
 * @file generic_ddm_problem.hh
 * @brief Generic PDELab problem wrapper for domain decomposition methods
 * 
 * This file provides a template-based wrapper around PDELab that handles
 * the complete setup and assembly workflow for domain decomposition methods.
 * It replaces the problem-specific classes (like PoissonProblem) with a
 * generic implementation that works with any PDE defined via traits.
 * 
 * Key features:
 * - Traits-based design: all PDE-specific types provided via template parameter
 * - Supports both CG and DG discretizations
 * - Efficient assembly of overlapping Dirichlet and Neumann matrices
 * - Built-in Dirichlet boundary condition handling and symmetric elimination
 * - MPI-aware parallel assembly with Neumann correction communication
 * 
 * Part of the unified PDELab example framework.
 */

#include <memory>
#include <vector>

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

#include "assemblewrapper.hh"
#include "pdelab_helper.hh"
#include "dune/ddm/logger.hh"

/**
 * @brief Generic problem class for PDELab+DDM workflows
 * 
 * This class handles the complete workflow from grid function space setup
 * to matrix assembly for domain decomposition methods. It is parameterized
 * by a Traits class that provides all PDE-specific type information.
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
template <class GridView, class Traits>
class GenericDDMProblem {
public:
  using RF = typename Traits::RF;
  using Grid = typename GridView::Grid;
  using DF = typename Grid::ctype;

  // Import types from traits
  using EntitySet = typename Traits::EntitySet;
  using FEM = typename Traits::FEM;
  using ModelProblem = typename Traits::ModelProblem;
  using BCType = typename Traits::BCType;
  using LocalOperator = typename Traits::LocalOperator;
  using Constraints = typename Traits::Constraints;
  
  static constexpr int blocksize = Traits::blocksize;
  static constexpr bool is_dg = Traits::is_dg;

  // PDELab types
  using VBE = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>;
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using GFS = Dune::PDELab::GridFunctionSpace<EntitySet, FEM, Constraints, VBE>;
  using CC = typename GFS::template ConstraintsContainer<RF>::Type;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, AssembleWrapper<LocalOperator>, MBE, RF, RF, RF, CC, CC>;

  // Vector and matrix types
  using Vec = Dune::PDELab::Backend::Vector<GFS, RF>;
  using Mat = typename GO::Jacobian;
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
   * @param gv Grid view defining the computational domain
   * @param helper MPI helper for parallel operations
   * @param model_problem Instance of the problem parameter class (coefficients, BC)
   */
  GenericDDMProblem(const GridView& gv, const Dune::MPIHelper& helper, const ModelProblem& model_problem = ModelProblem())
      : es(gv)
      , modelProblem(model_problem)
      , bc(gv, modelProblem)
      , lop(modelProblem)
      , wrapper(std::make_unique<AssembleWrapper<LocalOperator>>(&lop))
  {
    using Dune::PDELab::Backend::native;
    
    // Create finite element map (construction differs for DG vs CG)
    if constexpr (is_dg) {
      fem = std::make_unique<FEM>();
    } else {
      fem = std::make_unique<FEM>(es);
    }
    
    // Create grid function space
    gfs = std::make_shared<GFS>(es, *fem);
    gfs->name("Solution");
    
    // Initialize vectors
    x = std::make_unique<Vec>(*gfs, 0.0);
    x0 = std::make_unique<Vec>(*gfs, 0.0);
    d = std::make_unique<Vec>(*gfs, 0.0);
    dirichlet_mask = std::make_unique<Vec>(*gfs, 0);

    // Interpolate Dirichlet boundary conditions
    // Note: ConvectionDiffusion uses ConvectionDiffusionDirichletExtensionAdapter
    // We need to handle this generically - for now assume the model problem provides g()
    cc.clear();
    Dune::PDELab::constraints(bc, *gfs, cc);
    
    // For problems with Dirichlet BC, interpolate the boundary values
    if constexpr (requires { typename Traits::DirichletExtension; }) {
      typename Traits::DirichletExtension g(es, modelProblem);
      Dune::PDELab::interpolate(g, *gfs, *x);
    } else {
      // Try generic approach - assume model problem can be used directly
      *x = 0.0;
    }

    // Set Dirichlet mask
    Dune::PDELab::set_constrained_dofs(cc, 1.0, *dirichlet_mask);

    // Create grid operator with appropriate matrix nonzeros estimate
    const int nz = Traits::AssemblyDefaults::nonzeros_per_row;
    go = std::make_unique<GO>(*gfs, cc, *gfs, cc, *wrapper, MBE(nz));

    logger::info("Assembling sparsity pattern");
    As = std::make_unique<Mat>(*go);

    logger::info("Assembling residual");
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
   * @param comm Overlapping communication object
   * @param first_neumann_region Region where first Neumann matrix is defined
   * @param second_neumann_region Region where second Neumann matrix is defined
   * @param overlap Overlap width
   * @param neumann_size_as_dirichlet If true, Neumann matrices have same size as Dirichlet
   * @param novlp_comm Non-overlapping communication (optional, for Neumann corrections)
   */
  template <class Communication>
  void assemble_overlapping_matrices(Communication& comm, NeumannRegion first_neumann_region, 
                                     NeumannRegion second_neumann_region, int overlap, 
                                     bool neumann_size_as_dirichlet = true,
                                     const Communication* novlp_comm = nullptr)
  {
    auto [A_dir_, A_neu_, B_neu_, dirichlet_mask_ovlp_, neumann_region_to_subdomain_] = 
        ::assemble_overlapping_matrices(
            *As, *x, *go, Dune::PDELab::Backend::native(*dirichlet_mask), 
            comm, first_neumann_region, second_neumann_region, overlap, 
            neumann_size_as_dirichlet, is_dg, novlp_comm);

    A_dir = std::move(A_dir_);
    A_neu = std::move(A_neu_);
    B_neu = std::move(B_neu_);
    dirichlet_mask_ovlp = std::move(*dirichlet_mask_ovlp_);
    neumann_region_to_subdomain = std::move(neumann_region_to_subdomain_);
  }

  /**
   * @brief Assemble only the overlapping Dirichlet matrix
   * 
   * Simplified version for coarse spaces that don't need Neumann matrices (e.g., POU).
   * 
   * @tparam Communication Parallel communication type
   * @param novlp_comm Non-overlapping communication
   * @param comm Overlapping communication
   */
  template <class Communication>
  void assemble_dirichlet_matrix_only([[maybe_unused]] const Communication& novlp_comm, const Communication& comm)
  {
    using Dune::PDELab::Backend::native;
    logger::info("Assembling overlapping Dirichlet matrix");
    jacobian();

    // Create communicator on the overlapping index set
    typename Communication::AllSet allset;
    Dune::Interface interface_ext;
    interface_ext.build(comm.remoteIndices(), allset, allset);
    Dune::VariableSizeCommunicator varcomm(interface_ext);

    // Communicate matrix structure and values
    #include "dune/ddm/datahandles.hh"
    CreateMatrixDataHandle cmdh(native(*As), comm.indexSet());
    varcomm.forward(cmdh);
    A_dir = std::make_shared<NativeMat>(cmdh.getOverlappingMatrix());

    AddMatrixDataHandle amdh(native(*As), *A_dir, comm.indexSet());
    varcomm.forward(amdh);

    // Set up Dirichlet mask on overlapping subdomain
    dirichlet_mask_ovlp = NativeVec(A_dir->N());
    dirichlet_mask_ovlp = 0;
    for (std::size_t i = 0; i < dirichlet_mask->N(); ++i) 
      dirichlet_mask_ovlp[i] = native(*dirichlet_mask)[i];
    comm.addOwnerCopyToAll(dirichlet_mask_ovlp, dirichlet_mask_ovlp);

    // Eliminate Dirichlet dofs symmetrically
    eliminate_dirichlet(*A_dir, dirichlet_mask_ovlp);

    // No Neumann matrices for this mode
    A_neu = nullptr;
    B_neu = nullptr;
  }

  /**
   * @brief Assemble the Jacobian matrix
   * 
   * Assembles the stiffness matrix and eliminates Dirichlet DOFs symmetrically.
   */
  void jacobian()
  {
    using Dune::PDELab::Backend::native;
    // Simple assembly without Neumann corrections
    std::map<int, std::vector<bool>> empty_boundary_map;
    std::vector<bool> empty_mask(As->N(), false);
    wrapper->set_masks(native(*As), &empty_boundary_map, &empty_boundary_map, 
                      &empty_mask, &empty_mask);
    go->jacobian(*x, *As);
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
  const ModelProblem& getUnderlyingProblem() const { return modelProblem; }
  ///@}

  ///@{
  /** @name Accessors for DDM matrices */
  
  std::shared_ptr<NativeMat> get_dirichlet_matrix() { return A_dir; }
  std::shared_ptr<NativeMat> get_first_neumann_matrix() { return A_neu; }
  std::shared_ptr<NativeMat> get_second_neumann_matrix() { return B_neu; }
  const std::vector<std::size_t>& get_neumann_region_to_subdomain() const { 
    return neumann_region_to_subdomain; 
  }
  ///@}

private:
  ///@{
  /** @name Core DUNE/PDELab objects */
  EntitySet es;
  ModelProblem modelProblem;
  BCType bc;
  CC cc;
  std::unique_ptr<FEM> fem;
  std::shared_ptr<GFS> gfs;
  LocalOperator lop;
  std::unique_ptr<AssembleWrapper<LocalOperator>> wrapper;
  std::unique_ptr<GO> go;
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
  std::shared_ptr<NativeMat> A_dir;    ///< Overlapping Dirichlet matrix
  std::shared_ptr<NativeMat> A_neu;    ///< First Neumann matrix
  std::shared_ptr<NativeMat> B_neu;    ///< Second/restricted Neumann matrix
  NativeVec dirichlet_mask_ovlp;       ///< Dirichlet mask on overlapping subdomain
  std::vector<std::size_t> neumann_region_to_subdomain; ///< Index mapping for ring coarse spaces
  ///@}
};
