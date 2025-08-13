#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <mpi.h>

#include <dune/common/version.hh>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-W#warnings" // Silence annoying warnings about #warnings in PDELab headers
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
#include <dune/pdelab/constraints/p0ghost.hh>
#include <dune/pdelab/finiteelementmap/pkfem.hh>
#include <dune/pdelab/finiteelementmap/qkdg.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/pdelab/gridfunctionspace/genericdatahandle.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include <dune/pdelab/gridfunctionspace/interpolate.hh>
#include <dune/pdelab/gridoperator/gridoperator.hh>
#include <dune/pdelab/localoperator/convectiondiffusiondg.hh>
#include <dune/pdelab/localoperator/convectiondiffusionfem.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>
#pragma pop diagnostic

#include <limits>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#include "assemblewrapper.hh"
#include "datahandles.hh"
#include "helpers.hh"
#include "spdlog/spdlog.h"

/** @brief Determines the region where the "restricted Neumann" matrix is defined (to be used with the function PoissonProblem#assemble_overlapping_matrices()).
 */
enum class NeumannRegion : std::uint8_t {
  Overlap,         ///< Restricted Neumann matrix is only defined in the overlapping region
  ExtendedOverlap, ///< Restricted Neumann matrix is defined in the overlap + one additional layer of finite elements towards the interior
  All              ///< Restricted Neumann matrix is defined on the whole subdomain
};

/** @brief Model problem with heterogeneous coefficients arranged in vertical beams
 *
 *  This class defines a convection-diffusion problem with strongly heterogeneous
 *  diffusion coefficients. The domain features vertical "beams" with high diffusion
 *  coefficient (1e6) arranged in a regular pattern, surrounded by regions with
 *  low diffusion coefficient (1). This creates a challenging problem for domain
 *  decomposition methods due to the strong contrast in material properties.
 *
 *  The problem uses Dirichlet boundary conditions and a unit source term.
 *
 *  @tparam GridView The DUNE grid view type
 *  @tparam RF The range field type (typically double)
 */
template <class GridView, class RF>
class PoissonModelProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  typename Traits::PermTensorType A(const typename Traits::ElementType &e, const typename Traits::DomainType &x) const
  {
    auto xg = e.geometry().global(x);
    const auto width = 0.02;

    const RF small_coeff = 1;
    const RF large_coeff = 1e6;
    RF coeff = small_coeff;

    int num_beams = 8;
    double space_between = 0.1;

    if (xg[1] <= 0.95) { // Beams should not touch the boundary
      for (int i = 1; i <= num_beams; ++i) {
        if (xg[0] >= i * space_between && xg[0] <= i * space_between + width) {
          coeff = large_coeff;
        }
      }

      if (xg[1] >= 0.95 - width) {
        for (int i = 1; i <= num_beams; ++i) {
          if (xg[0] >= i * space_between && xg[0] <= i * space_between + 3 * width) {
            coeff = large_coeff;
          }
        }
      }

      if (xg[1] >= 0.95 - 2 * width) {
        for (int i = 1; i <= num_beams; ++i) {
          if (xg[0] >= i * space_between + 2 * width && xg[0] <= i * space_between + 3 * width) {
            coeff = large_coeff;
          }
        }
      }
    }

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++) {
      for (std::size_t j = 0; j < Traits::dimDomain; j++) {
        I[i][j] = (i == j) ? 1 : 0;
      }
    }
    return I;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType &, const typename Traits::DomainType &) const { return 1.0; }

  typename Traits::RangeFieldType g(const typename Traits::ElementType &e, const typename Traits::DomainType &x) const
  {
    return 0;
    auto xglobal = e.geometry().global(x);
    return xglobal[0];
  }

  BC bctype(const typename Traits::IntersectionType &is, const typename Traits::IntersectionDomainType &x) const
  {
    auto xglobal = is.geometry().global(x);
    if (true or xglobal[0] < 1e-6) {
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    }
    if (xglobal[1] > 1.0 - 1e-6) {
      return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    }
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
  }
};

/** @brief Model problem with complex heterogeneous coefficient patterns (islands)
 *
 *  This class defines a convection-diffusion problem with complex heterogeneous
 *  diffusion coefficients arranged in geometric patterns resembling "islands".
 *  The coefficient distribution includes:
 *  - Diagonal strips with varying coefficients based on position
 *  - Triangular and trapezoidal regions with high coefficients
 *  - Checkerboard pattern in parts of the domain
 *  - Coefficient values ranging from 1.0 to 10^5 * (position-dependent factors)
 *
 *  This creates an even more challenging test case for domain decomposition
 *  methods than the beam problem, with irregular coefficient patterns.
 *
 *  @tparam GridView The DUNE grid view type
 *  @tparam RF The range field type (typically double)
 */
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

/** @brief Check if a grid type is a YASPGrid
 *
 *  Distinguishes YASPGrid from UGGrid using the method "torus" which only exists in YASPGrid.
 *
 *  @tparam Grid The grid type to check
 *  @return true if the grid is a YASPGrid, false otherwise
 */
template <class Grid>
constexpr bool isYASPGrid()
{
  return requires(Grid g) { g.torus(); }; // We distinguish YASPGrid and UGGrid using the method "torus" which only exists in YASPGrid
}

/** @brief A PDELab-based solver for convection-diffusion problems with domain decomposition support
 *
 *  This class implements a finite element solver for convection-diffusion problems using the DUNE
 *  PDELab framework. It is specifically designed for domain decomposition methods and provides
 *  functionality to assemble overlapping Dirichlet and Neumann matrices required for coarse space
 *  construction in methods like GenEO and MsGFEM.
 *
 *  Key features:
 *  - Supports both continuous Galerkin (CG) and discontinuous Galerkin (DG) discretizations
 *  - Automatic finite element selection based on grid type (Qk for structured, Pk for unstructured)
 *  - Efficient assembly of overlapping matrices for domain decomposition coarse spaces
 *  - Built-in Dirichlet boundary condition handling and symmetric elimination
 *  - MPI-aware parallel assembly with communication of Neumann correction terms
 *
 *  The class handles the complete workflow from grid function space setup to matrix assembly,
 *  including proper treatment of constraints, boundary conditions, and parallel communication.
 *
 *  @tparam GridView The DUNE grid view type
 *  @tparam USEDG Whether to use discontinuous Galerkin (true) or continuous Galerkin (false)
 *
 *  @note The class uses OverlappingEntitySet even in non-overlapping settings to prevent
 *        PDELab from attempting automatic data distribution, as all parallel communication
 *        is handled manually for domain decomposition purposes.
 */
template <class GridView, bool USEDG = false>
class PoissonProblem {
public:
  using RF = double;
  using Grid = typename GridView::Grid;
  using DF = typename Grid::ctype;

  using ES = Dune::PDELab::OverlappingEntitySet<GridView>; // Even though we're in a non-overlapping setting, we tell PDELab that we're in an overlapping setting, because
                                                           // we handle everything ourselves; PDELab should just assemble locally and not attempt to distribute any data
                                                           // (which it would do if we put 'NonOverlappingEntitySet' here). Note that this still skips ghost elements,
                                                           // only AllEntitySet would also include them.

  // using ModelProblem = PoissonModelProblem<ES, RF>;
  using ModelProblem = IslandsModelProblem<ES, RF>;
  // using ModelProblem = Dune::PDELab::ConvectionDiffusionModelProblem<ES, RF>;
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<ModelProblem>;

  static constexpr int degree = 1;
  // clang-format off
  using FEM = std::conditional_t<isYASPGrid<Grid>(),                                                                                                              // If YASP grid
                                 std::conditional_t<USEDG,                                                                                                        // and discretisation is DG
                                                    Dune::PDELab::QkDGLocalFiniteElementMap<typename GridView::Grid::ctype, double, degree, GridView::dimension>, // then use Qk DG
                                                    Dune::PDELab::QkLocalFiniteElementMap<ES, DF, RF, degree>>,                                                   // otherwise use Qk CG
                                 Dune::PDELab::PkLocalFiniteElementMap<ES, DF, RF, degree>>;                                                                      // and in case of another grid, just use Pk CG
  // clang-format on
  using LOP = std::conditional_t<USEDG, Dune::PDELab::ConvectionDiffusionDG<ModelProblem, FEM>, Dune::PDELab::ConvectionDiffusionFEM<ModelProblem, FEM>>;

  using CON = std::conditional_t<USEDG, Dune::PDELab::P0ParallelGhostConstraints, Dune::PDELab::ConformingDirichletConstraints>;

  using VBE = std::conditional_t<USEDG, Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none, Dune::QkStuff::QkSize<degree, GridView::dimension>::value>,
                                 Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>>;
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;

  using GFS = Dune::PDELab::GridFunctionSpace<ES, FEM, CON, VBE>;

  using CC = typename GFS::template ConstraintsContainer<RF>::Type;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, AssembleWrapper<LOP>, MBE, RF, RF, RF, CC, CC>;

  using Vec = Dune::PDELab::Backend::Vector<GFS, RF>;
  using Mat = typename GO::Jacobian;
  using NativeMat = Dune::PDELab::Backend::Native<Mat>;
  using NativeVec = Dune::PDELab::Backend::Native<Vec>;

  /** @brief Constructor that sets up the complete finite element discretization
   *
   *  Initializes the grid function space, finite element map, constraints, and assembles
   *  the initial system state including sparsity pattern and residual vector. The constructor
   *  performs the following steps:
   *
   *  1. Sets up finite element map (FEM) based on grid type and discretization choice
   *  2. Creates grid function space (GFS) with appropriate constraints
   *  3. Initializes solution vector with Dirichlet boundary values
   *  4. Sets up Dirichlet constraint mask for boundary condition enforcement
   *  5. Creates grid operator for matrix/vector assembly
   *  6. Assembles initial sparsity pattern and residual vector
   *  7. Performs parallel communication of residual if running with multiple processes
   *
   *  @param gv The grid view defining the computational domain
   *  @param helper MPI helper for parallel communication setup
   *
   *  @note After construction, the object is ready for matrix assembly operations
   *        via assemble_overlapping_matrices() or assemble_dirichlet_matrix_only()
   */
  PoissonProblem(const GridView &gv, const Dune::MPIHelper &helper) : es(gv), bc(es, modelProblem), lop(modelProblem), wrapper(std::make_unique<AssembleWrapper<LOP>>(&lop))
  {
    using Dune::PDELab::Backend::native;
    if constexpr (!USEDG) {
      fem = std::make_unique<FEM>(es);
    }
    else {
      fem = std::make_unique<FEM>();
    }
    gfs = std::make_unique<GFS>(es, *fem);
    gfs->name("Solution");
    x = std::make_unique<Vec>(*gfs, 0.0);
    x0 = std::make_unique<Vec>(*gfs, 0.);
    d = std::make_unique<Vec>(*gfs, 0.);
    dirichlet_mask = std::make_unique<Vec>(*gfs, 0);

    // Create solution vector and initialise with Dirichlet conditions
    Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter g(es, modelProblem);
    cc.clear();
    Dune::PDELab::interpolate(g, *gfs, *x);
    Dune::PDELab::constraints(bc, *gfs, cc);

    // Set Dirichlet mask
    Dune::PDELab::set_constrained_dofs(cc, 1., *dirichlet_mask);

    // Create the grid operator, assemble the residual and setup the nonzero pattern of the matrix
    go = std::make_unique<GO>(*gfs, cc, *gfs, cc, *wrapper, MBE(9));

    spdlog::info("Assembling sparsity pattern");
    As = std::make_unique<Mat>(*go);

    spdlog::info("Assembling residual");
    go->residual(*x, *d);

    *x0 = *x;

    if (helper.size() > 1) {
      Dune::PDELab::AddDataHandle adddhd(*gfs, *d);
      gfs->gridView().communicate(adddhd, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);
    }
  }

  /** @brief Assemble only the overlapping Dirichlet matrix

      This is a simplified version of assemble_overlapping_matrices() that only assembles
      the Dirichlet matrix. This is sufficient for coarse spaces like POU that don't
      require Neumann matrices for eigenproblems.

      The created matrix can be accessed via get_dirichlet_matrix().
   */
  template <class ExtendedRemoteIndices>
  void assemble_dirichlet_matrix_only(const ExtendedRemoteIndices &extids)
  {
    using Dune::PDELab::Backend::native;
    spdlog::info("Assembling overlapping Dirichlet matrix");

    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    Dune::Interface interface_ext;
    interface_ext.build(extids.get_remote_indices(), allAttributes, allAttributes);
    Dune::VariableSizeCommunicator varcomm_ext(interface_ext);

    // Create the (at this point still empty) overlapping subdomain matrix
    A_dir = std::make_shared<NativeMat>(extids.create_overlapping_matrix(native(*As)));

    // Simple assembly without Neumann corrections - just assemble locally
    std::map<int, std::vector<bool>> empty_boundary_map;
    std::vector<bool> empty_mask(As->N(), false);
    wrapper->set_masks(native(*As), &empty_boundary_map, &empty_boundary_map, &empty_mask, &empty_mask);
    go->jacobian(*x, *As);

    // Assemble the overlapping Dirichlet matrix by adding contributions
    AddMatrixDataHandle amdh(native(*As), *A_dir, extids.get_parallel_index_set());
    extids.get_overlapping_communicator().forward(amdh);

    // Set up Dirichlet mask on overlapping subdomain
    dirichlet_mask_ovlp = NativeVec(A_dir->N());
    dirichlet_mask_ovlp = 0;
    for (std::size_t i = 0; i < dirichlet_mask->N(); ++i) {
      dirichlet_mask_ovlp[i] = native(*dirichlet_mask)[i];
    }
    AddVectorDataHandle<NativeVec> advdh;
    advdh.setVec(dirichlet_mask_ovlp);
    varcomm_ext.forward(advdh);

    // Eliminate Dirichlet dofs symmetrically
    eliminate_dirichlet(*A_dir, dirichlet_mask_ovlp);

    // For POU, we don't need Neumann matrices, so set them to nullptr
    A_neu = nullptr;
    B_neu = nullptr;
  }

  /** @brief Assemble the overlapping Dirichlet and Neumann matrices

      This function assembles the stiffness matrix locally and then creates three matrices:
      - The "Dirichlet" matrix which is obtained by extracting a sub-matrix out of the global
        stiffness matrix on the overlapping subdomain that is described by \p extids. It is
        called the Dirichlet matrix because it corresponds to a problem with Dirichlet boundary
        conditions at the overlapping subdomain boundary. This is the matrix that should be used
        on the fine level in a two-level Schwarz method.
      - The first "Neumann" matrix which might also be defined on the whole overlapping subdomain,
        or on a smaller sub-region. The region is determined by the second parameter \p first_neumann_region.
      - The second "Neumann" matrix which might also be defined on the whole overlapping subdomain,
        or on a smaller sub-region. The region is determined by the third parameter \p second_neumann_region.

      The created matrices can be accessed via the methods get_dirichlet_matrix(), get_first_neumann_matrix(),
      and get_second_neumann_matrix(). If \p first_neumann_region and \p second_neumann_region are
      the same, then only two matrices will be assembled and get_first_neumann_matrix() and get_second_neumann_matrix()
      return pointers to the same matrix.

      The last parameter \p neumann_size_as_dirichlet can be used to control the size of the Neumann matrix.
      If it is true, then the Neumann matrices will be of the same size as the Dirichlet matrix. If it is false
      and, e.g., \p first_neumann_region equals NeumannRegion::Overlap, then the Neumann matrix will only have
      as many rows/columns as there are degrees of freedom in the overlap region. This can be used to assemble
      the matrices for the "ring" coarse spaces.

      @see NeumannRegion for the options that can be used for \p first_neumann_region and \p second_neumann_region.
   */
  template <class ExtendedRemoteIndices>
  void assemble_overlapping_matrices(const ExtendedRemoteIndices &extids, NeumannRegion first_neumann_region, NeumannRegion second_neumann_region, bool neumann_size_as_dirichlet = true)
  {
    using Dune::PDELab::Backend::native;
    spdlog::info("Assembling overlapping Dirichlet and Neumann matrices");

    int ownrank{};
    MPI_Comm_rank(extids.get_remote_indices().communicator(), &ownrank);
    const auto &ovlp_paridxs = extids.get_parallel_index_set();

    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    Dune::Interface interface_ext;
    interface_ext.build(extids.get_remote_indices(), allAttributes, allAttributes);
    Dune::VariableSizeCommunicator varcomm_ext(interface_ext);

    // Create the (at this point still empty) overlapping subdomain matrix
    A_dir = std::make_shared<NativeMat>(extids.create_overlapping_matrix(native(*As)));

    // Identify the boundary of the overlapping subdomain
    IdentifyBoundaryDataHandle ibdh(*A_dir, extids.get_parallel_index_set());
    varcomm_ext.forward(ibdh);

    // Now we know *our* subdomain boundary. We'll now create a vector on the overlapping
    // subdomain that contains the value 1 on the subdomain boundary and the value 2 one
    // layer into the subdomain. We'll communicate this vector with our neighbours so
    // that they can find out which elements they have to integrate separately in order
    // to send us the "Neumann correction terms".
    const auto &boundary_mask = ibdh.get_boundary_mask();

    std::vector<int> boundary_dst(boundary_mask.size(), std::numeric_limits<int>::max() - 1);
    for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
      if (boundary_mask[i]) {
        boundary_dst[i] = 0;
      }
    }

    int overlap = extids.get_overlap();
    for (int round = 0; round <= 2 * overlap + 4; ++round) {
      for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
        for (auto cit = (*A_dir)[i].begin(); cit != (*A_dir)[i].end(); ++cit) {
          boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cit.index()] + 1); // Increase distance from boundary by one
        }
      }
    }

    NativeVec boundary_indicator(extids.size());
    boundary_indicator = 0;
    for (std::size_t i = 0; i < extids.size(); ++i) {
      if (boundary_dst[i] == 0) {
        boundary_indicator[i] = 1;
      }
      else if (boundary_dst[i] == 1) {
        boundary_indicator[i] = 2;
      }
    }

    CopyVectorDataHandleWithRank<NativeVec> cvdh(boundary_indicator);
    varcomm_ext.forward(cvdh);

    // Next, we turn this vector into two boolean masks for each rank.
    std::map<int, std::vector<bool>> on_boundary_mask_for_rank;
    std::map<int, std::vector<bool>> inside_boundary_mask_for_rank;
    for (const auto &[rank, copied_vec] : cvdh.copied_vecs) {
      on_boundary_mask_for_rank[rank].resize(As->N(), false);
      inside_boundary_mask_for_rank[rank].resize(As->N(), false);

      for (std::size_t i = 0; i < As->N(); ++i) {
        on_boundary_mask_for_rank[rank][i] = copied_vec[i] == 1;
        inside_boundary_mask_for_rank[rank][i] = copied_vec[i] == 2;
      }
    }

    std::vector<bool> on_boundary_mask(As->N(), false);
    std::vector<bool> outside_boundary_mask(As->N(), false);
    if (first_neumann_region != NeumannRegion::All and first_neumann_region != second_neumann_region) {
      DUNE_THROW(Dune::NotImplemented, "Two different Neumann regions are only supported if the first is NeumannRegion::All");
    }
    if (first_neumann_region == NeumannRegion::Overlap or second_neumann_region == NeumannRegion::Overlap) {
      for (std::size_t i = 0; i < As->N(); ++i) {
        on_boundary_mask[i] = boundary_dst[i] == (2 * overlap);
        outside_boundary_mask[i] = boundary_dst[i] == (2 * overlap + 1);
      }
    }
    else if (first_neumann_region == NeumannRegion::ExtendedOverlap or second_neumann_region == NeumannRegion::ExtendedOverlap) {
      for (std::size_t i = 0; i < As->N(); ++i) {
        on_boundary_mask[i] = boundary_dst[i] == (2 * overlap + 1);
        outside_boundary_mask[i] = boundary_dst[i] == (2 * overlap + 2);
      }
    }

    // No we can assemble while keeping track of the contributions that later need to be sent to other ranks so that they can assemble the
    // Neumann matrices they need.
    wrapper->set_masks(native(*As), &on_boundary_mask_for_rank, &inside_boundary_mask_for_rank, &on_boundary_mask, &outside_boundary_mask);
    go->jacobian(*x, *As);

    Dune::GlobalLookupIndexSet glis(extids.get_parallel_index_set());
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

      MPI_Isend(triples.data(), triples.size(), triple_type, rank, 0, extids.get_remote_indices().communicator(), &requests.emplace_back());
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

      MPI_Probe(rank, 0, extids.get_remote_indices().communicator(), &status);
      MPI_Get_count(&status, triple_type, &count);
      num_triples += count;

      remote_triples[rank].resize(count);
      MPI_Recv(remote_triples[rank].data(), count, triple_type, rank, 0, extids.get_remote_indices().communicator(), MPI_STATUS_IGNORE);
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // Now we can assemble the overlapping Dirichlet matrix
    AddMatrixDataHandle amdh(native(*As), *A_dir, extids.get_parallel_index_set());
    extids.get_overlapping_communicator().forward(amdh);

    // Next, make sure that Dirichlet dofs are eliminated symmetrically
    dirichlet_mask_ovlp = NativeVec(A_dir->N());
    dirichlet_mask_ovlp = 0;
    for (std::size_t i = 0; i < dirichlet_mask->N(); ++i) {
      dirichlet_mask_ovlp[i] = native(*dirichlet_mask)[i];
    }
    AddVectorDataHandle<NativeVec> advdh;
    advdh.setVec(dirichlet_mask_ovlp);
    varcomm_ext.forward(advdh);

    eliminate_dirichlet(*A_dir, dirichlet_mask_ovlp);

    // Next, assemble the first Neumann matrix
    if (first_neumann_region == NeumannRegion::All) {
      A_neu = std::make_shared<NativeMat>(*A_dir);
      for (const auto &[rank, triples] : remote_triples) {
        for (const auto &triple : triples) {
          if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
            auto lrow = ovlp_paridxs[triple.row].local();
            auto lcol = ovlp_paridxs[triple.col].local();

            (*A_neu)[lrow][lcol] -= triple.val;
          }
          else {
            spdlog::debug("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
          }
        }
      }
      // Just to be sure
      eliminate_dirichlet(*A_neu, dirichlet_mask_ovlp);
    }
    else if (first_neumann_region == NeumannRegion::ExtendedOverlap) {
      if (neumann_size_as_dirichlet) {
        // First create a copy of the Dirichlet matrix, but only those entries in the extended overlap region
        auto avg = A_dir->nonzeroes() / A_dir->N() + 2;
        A_neu = std::make_shared<NativeMat>(A_dir->N(), A_dir->N(), avg, 0.2, NativeMat::implicit);
        for (auto ri = A_dir->begin(); ri != A_dir->end(); ++ri) {
          if (boundary_dst[ri.index()] > 2 * overlap + 1) {
            continue;
          }

          for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
            if (boundary_dst[ci.index()] > 2 * overlap + 1) {
              continue;
            }

            A_neu->entry(ri.index(), ci.index()) = *ci;
          }
        }
        A_neu->compress();

        // Then apply the outer and inner Neumann corrections
        for (const auto &[rank, triples] : remote_triples) {
          for (const auto &triple : triples) {
            if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
              auto lrow = ovlp_paridxs[triple.row].local();
              auto lcol = ovlp_paridxs[triple.col].local();

              (*A_neu)[lrow][lcol] -= triple.val;
            }
            else {
              spdlog::debug("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
            }
          }
        }

        for (const auto &triple : triples_for_rank[-1]) {
          (*A_neu)[triple.row][triple.col] -= triple.val;
        }

        eliminate_dirichlet(*A_neu, dirichlet_mask_ovlp);
      }
      else {
        // In this case, we essentially do the same thing as above but now the matrix is only defined on the overlap region
        // (whereas above it is just zero everywhere else). We start by creating an index set for the overlap region, i.e.,
        // a vector with as many entries as there are dofs in the overlap regions that maps 'overlap region index' -> 'subdomain index'.
        auto n = std::count_if(boundary_dst.begin(), boundary_dst.end(), [overlap](auto &&x) { return x <= 2 * overlap + 1; });
        neumann_region_to_subdomain.resize(n);
        std::size_t cnt = 0;
        for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
          if (boundary_dst[i] <= 2 * overlap + 1) {
            neumann_region_to_subdomain[cnt++] = i;
          }
        }

        // Also create the inverse mapping.
        // TODO: Check if it's faster to create a vector that just has bogus values outside the interesting region
        std::unordered_map<std::size_t, std::size_t> subdomain_to_neumann_region;
        for (std::size_t i = 0; i < neumann_region_to_subdomain.size(); ++i) {
          subdomain_to_neumann_region[neumann_region_to_subdomain[i]] = i;
        }

        // Additionally, create a mapping from the interior, overlapping one layer with the extended overlap, to the subdomain
        // TODO: I'm not sure if this is the correct place to build this. This is only needed in the ring coarse spaces.
        auto N = std::count_if(boundary_dst.begin(), boundary_dst.end(), [&](auto &&x) { return x >= 2 * overlap; });
        interior_to_subdomain.resize(N);
        cnt = 0;
        for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
          if (boundary_dst[i] >= 2 * overlap) {
            interior_to_subdomain[cnt++] = i;
          }
        }

        // Then, build the matrix
        auto avg = A_dir->nonzeroes() / A_dir->N() + 2;
        A_neu = std::make_shared<NativeMat>(n, n, avg, 0.2, NativeMat::implicit);

        for (std::size_t i = 0; i < neumann_region_to_subdomain.size(); ++i) {
          auto ri = neumann_region_to_subdomain[i];
          for (auto ci = (*A_dir)[ri].begin(); ci != (*A_dir)[ri].end(); ++ci) {
            if (boundary_dst[ci.index()] > 2 * overlap + 1) {
              continue;
            }

            A_neu->entry(i, subdomain_to_neumann_region[ci.index()]) = *ci;
          }
        }
        A_neu->compress();

        // Then apply the outer and inner Neumann corrections
        for (const auto &[rank, triples] : remote_triples) {
          for (const auto &triple : triples) {
            if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
              auto lrow = ovlp_paridxs[triple.row].local();
              auto lcol = ovlp_paridxs[triple.col].local();

              assert(subdomain_to_neumann_region.contains(lrow) && subdomain_to_neumann_region.contains(lcol) && "Remote corrections should be applicable to ring matrix");

              (*A_neu)[subdomain_to_neumann_region[lrow]][subdomain_to_neumann_region[lcol]] -= triple.val;
            }
            else {
              spdlog::debug("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
            }
          }
        }

        for (const auto &triple : triples_for_rank[-1]) {
          assert(subdomain_to_neumann_region.contains(triple.row) && subdomain_to_neumann_region.contains(triple.col) && "Own corrections should be applicable to ring matrix");

          (*A_neu)[subdomain_to_neumann_region[triple.row]][subdomain_to_neumann_region[triple.col]] -= triple.val;
        }
      }
    }
    else {
      DUNE_THROW(Dune::NotImplemented, "Only NeumannRegion::All and NeumannRegion::ExtendedOverlap implemented for first Neumann matrix currently");
    }

    // Lastly, assemble then second Neumann matrix
    if (second_neumann_region == first_neumann_region) {
      B_neu = A_neu;
    }
    else if (second_neumann_region == NeumannRegion::Overlap) {
      // First create a copy of the Neumann matrix, but only those entries in the overlap region
      auto avg = A_dir->nonzeroes() / A_dir->N() + 2;
      B_neu = std::make_shared<NativeMat>(A_dir->N(), A_dir->N(), avg, 0.2, NativeMat::implicit);
      for (auto ri = A_neu->begin(); ri != A_neu->end(); ++ri) {
        if (boundary_dst[ri.index()] > 2 * overlap) {
          continue;
        }

        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if (boundary_dst[ci.index()] > 2 * overlap) {
            continue;
          }

          B_neu->entry(ri.index(), ci.index()) = *ci;
        }
      }
      B_neu->compress();

      // Finally, apply the "inner" Neumann corrections
      for (const auto &triple : triples_for_rank[-1]) {
        (*B_neu)[triple.row][triple.col] -= triple.val;
      }

      eliminate_dirichlet(*B_neu, dirichlet_mask_ovlp);
    }
    else {
      DUNE_THROW(Dune::NotImplemented, "Unknown neumann_region type");
    }
  }

  ///@{
  /** @name Vector and matrix accessors
   *  Methods to access the various vectors and matrices used in the problem setup.
   */

  /** @brief Get the solution vector (PDELab backend)
   *  @return Reference to the initial solution vector with Dirichlet boundary values
   */
  Vec &getXVec() { return *x0; }

  /** @brief Get the solution vector (native ISTL)
   *  @return Reference to the native ISTL solution vector
   */
  NativeVec &getX() { return Dune::PDELab::Backend::native(*x0); }

  /** @brief Get the residual vector (native ISTL)
   *  @return Reference to the native ISTL residual vector
   */
  NativeVec &getD() const { return Dune::PDELab::Backend::native(*d); }

  /** @brief Get the residual vector (PDELab backend)
   *  @return Reference to the PDELab residual vector
   */
  Vec &getDVec() const { return *d; }

  /** @brief Get the Dirichlet constraint mask
   *  @return Reference to vector with 1.0 at Dirichlet DOFs, 0.0 elsewhere
   */
  const Vec &getDirichletMask() const { return *dirichlet_mask; }

  /** @brief Get the overlapping Dirichlet constraint mask
   *  @return Reference to Dirichlet mask on the overlapping subdomain (valid after matrix assembly)
   */
  const NativeVec &get_overlapping_dirichlet_mask() const { return dirichlet_mask_ovlp; }

  /** @brief Get the system matrix (native ISTL)
   *  @return Reference to the native ISTL system matrix
   */
  NativeMat &getA() { return Dune::PDELab::Backend::native(*As); }
  ///@}

  ///@{
  /** @name Grid function space and entity set accessors
   *  Methods to access the underlying DUNE/PDELab objects.
   */

  /** @brief Get the grid function space
   *  @return Reference to the PDELab grid function space
   */
  const GFS &getGFS() const { return *gfs; }

  /** @brief Get the entity set
   *  @return Reference to the overlapping entity set used for assembly
   */
  const ES &getEntitySet() const { return es; }

  /** @brief Get the underlying model problem
   *  @return Reference to the PDE parameter object defining coefficients and boundary conditions
   */
  const ModelProblem &getUnderlyingProblem() const { return modelProblem; }
  ///@}

  ///@{
  /** @name Overlapping matrix accessors
   *  Methods to access matrices assembled for domain decomposition coarse spaces.
   *  These are only valid after calling assemble_overlapping_matrices() or assemble_dirichlet_matrix_only().
   */

  /** @brief Get the overlapping Dirichlet matrix
   *
   *  Returns the matrix corresponding to the PDE with Dirichlet boundary conditions
   *  at the overlapping subdomain boundary. This matrix is used on the fine level
   *  in two-level Schwarz methods.
   *
   *  @return Shared pointer to the Dirichlet matrix (nullptr if not yet assembled)
   */
  std::shared_ptr<NativeMat> get_dirichlet_matrix() { return A_dir; }

  /** @brief Get the first overlapping Neumann matrix
   *
   *  Returns the matrix corresponding to the PDE with Neumann boundary conditions
   *  at the overlapping subdomain boundary. The region where this matrix is defined
   *  depends on the NeumannRegion parameter used in assemble_overlapping_matrices().
   *
   *  @return Shared pointer to the first Neumann matrix (nullptr if not yet assembled)
   */
  std::shared_ptr<NativeMat> get_first_neumann_matrix() { return A_neu; }

  /** @brief Get the second overlapping Neumann matrix
   *
   *  Returns the restricted Neumann matrix, which may be defined on a different region
   *  than the first Neumann matrix. If both Neumann regions are identical, this returns
   *  the same pointer as get_first_neumann_matrix().
   *
   *  @return Shared pointer to the second Neumann matrix (nullptr if not yet assembled)
   *  @see assemble_overlapping_matrices() for details on when matrices are shared
   */
  std::shared_ptr<NativeMat> get_second_neumann_matrix() { return B_neu; }
  ///@}

  ///@{
  /** @name Index mapping accessors
   *  Methods to access index mappings between different regions (used for ring coarse spaces).
   */

  /** @brief Get mapping from Neumann region to subdomain indices
   *  @return Vector mapping Neumann region DOF indices to subdomain DOF indices
   *  @note Only populated when neumann_size_as_dirichlet=false in assemble_overlapping_matrices()
   */
  const std::vector<std::size_t> &get_neumann_region_to_subdomain() const { return neumann_region_to_subdomain; }

  /** @brief Get mapping from overlapping interior to subdomain indices
   *  @return Vector mapping interior overlap DOF indices to subdomain DOF indices
   *  @note Used for ring coarse space construction
   */
  const std::vector<std::size_t> &get_overlapping_interior_to_subdomain() const { return interior_to_subdomain; }

private:
  /** @brief Symmetrically eliminate Dirichlet degrees of freedom from a matrix
   *
   *  For Dirichlet DOFs (where dirichlet_mask > 0), sets the diagonal entry to 1.0
   *  and all off-diagonal entries in that row to 0.0. Also zeros out the column
   *  entries corresponding to Dirichlet DOFs in other rows. This maintains the
   *  symmetric structure of the matrix while enforcing Dirichlet constraints.
   *
   *  @param A Matrix to modify (modified in place)
   *  @param dirichlet_mask Vector with non-zero values at Dirichlet DOF indices
   */
  void eliminate_dirichlet(NativeMat &A, const NativeVec &dirichlet_mask)
  {
    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      if (dirichlet_mask[ri.index()] > 0) {
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          *ci = (ci.index() == ri.index()) ? 1.0 : 0.0;
        }
      }
      else {
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if (dirichlet_mask[ci.index()] > 0) {
            *ci = 0.0;
          }
        }
      }
    }
  }

  /** @brief Symmetrically eliminate Dirichlet degrees of freedom using index mapping
   *
   *  Similar to eliminate_dirichlet() but uses an index mapping to translate between
   *  matrix indices and mask indices. This is used when the matrix has different
   *  dimensions than the mask (e.g., for restricted Neumann matrices).
   *
   *  @param A Matrix to modify (modified in place)
   *  @param dirichlet_mask Vector with non-zero values at Dirichlet DOF indices
   *  @param map Index mapping from matrix indices to mask indices
   */
  void eliminate_dirichlet(NativeMat &A, const NativeVec &dirichlet_mask, const std::vector<std::size_t> &map)
  {
    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      if (dirichlet_mask[map[ri.index()]] > 0) {
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          *ci = (ci.index() == ri.index()) ? 1.0 : 0.0;
        }
      }
      else {
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if (dirichlet_mask[map[ci.index()]] > 0) {
            *ci = 0.0;
          }
        }
      }
    }
  }

  ///@{
  /** @name Core DUNE/PDELab objects */
  ES es;                    ///< Overlapping entity set for local assembly
  std::unique_ptr<FEM> fem; ///< Finite element map
  std::unique_ptr<GFS> gfs; ///< Grid function space

  BC bc;                     ///< Boundary condition adapter
  CC cc;                     ///< Constraints container
  ModelProblem modelProblem; ///< PDE parameter object (coefficients, boundary conditions, source term)

  LOP lop;                                       ///< Local operator for PDE assembly
  std::unique_ptr<AssembleWrapper<LOP>> wrapper; ///< Assembly wrapper for mask handling
  std::unique_ptr<GO> go;                        ///< Grid operator for matrix/vector assembly
  ///@}

  ///@{
  /** @name Solution and system vectors/matrices */
  std::unique_ptr<Vec> x;              ///< Current solution vector
  std::unique_ptr<Vec> x0;             ///< Initial solution vector (with Dirichlet boundary values)
  std::unique_ptr<Vec> d;              ///< Residual vector
  std::unique_ptr<Vec> dirichlet_mask; ///< Mask vector (1.0 at Dirichlet DOFs, 0.0 elsewhere)
  std::unique_ptr<Mat> As;             ///< System matrix (PDELab backend)
  ///@}

  ///@{
  /** @name Domain decomposition matrices */
  std::shared_ptr<NativeMat> A_dir; ///< Overlapping Dirichlet matrix (assembled after matrix assembly call)
  std::shared_ptr<NativeMat> A_neu; ///< First overlapping Neumann matrix
  std::shared_ptr<NativeMat> B_neu; ///< Second/restricted Neumann matrix (may share memory with A_neu)
  ///@}

  ///@{
  /** @name Index mappings for ring coarse spaces */
  std::vector<std::size_t> neumann_region_to_subdomain; ///< Mapping from Neumann region DOF index to subdomain DOF index
  std::vector<std::size_t> interior_to_subdomain;       ///< Mapping from interior overlap DOF index to subdomain DOF index
  ///@}

  NativeVec dirichlet_mask_ovlp; ///< Dirichlet mask on overlapping subdomain
};
