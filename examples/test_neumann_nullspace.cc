/**
 * @file test_neumann_nullspace.cc
 * @brief Test that A * 1 = 0 for a pure Neumann problem
 *
 * For the Laplacian with pure Neumann boundary conditions, constant functions
 * should be in the null space of the stiffness matrix. This test verifies that
 * the parallel assembly and communication are correct by checking ||A * 1||.
 *
 * The test uses the NonOverlappingOperator which:
 * 1. Applies the local matrix A_local to the input vector
 * 2. Communicates contributions via addOwnerCopyToOwnerCopy
 *
 * If the assembly or communication is incorrect, A * 1 will not be zero
 * (particularly at processor boundaries).
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define USE_UGGRID 1  // Set to 1 for UGGrid, 0 for YaspGrid
#define USE_DG 1      // Set to 1 for DG discretization, 0 for conforming FEM

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <dune/common/fmatrix.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/pdelab/backend/istl.hh>
#include <dune/pdelab/common/partitionviewentityset.hh>
#include <dune/pdelab/constraints/conforming.hh>
#include <dune/pdelab/constraints/p0ghost.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/pdelab/finiteelementmap/pkfem.hh>
#include <dune/pdelab/finiteelementmap/qkdg.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include <dune/pdelab/gridoperator/gridoperator.hh>
#include <dune/pdelab/localoperator/convectiondiffusionfem.hh>
#include <dune/pdelab/localoperator/convectiondiffusiondg.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>

#if USE_UGGRID
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#else
#include <dune/grid/yaspgrid.hh>
#endif

#include <dune/ddm/nonoverlapping_operator.hh>
#include <dune/ddm/pdelab_helper.hh>

/**
 * @brief Heterogeneous problem with pure Neumann boundary conditions (like IslandsModelProblem)
 *
 * This problem defines -∇·(κ∇u) = f with heterogeneous coefficients and homogeneous Neumann BCs.
 * The stiffness matrix for this problem should STILL have constant functions in its null space.
 */
template <class GridView, class RF>
class PureNeumannProblem : public Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF> {
  using BC = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;

public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GridView, RF>::Traits;

  /// Diffusion tensor: heterogeneous coefficients like IslandsModelProblem
  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& /*xloc*/) const
  {
    auto xg = e.geometry().center();
    int ix = std::floor(15.0 * xg[0]);
    int iy = std::floor(15.0 * xg[1]);
    auto x = xg[0];
    auto y = xg[1];

    double kappa = 1.0;

    if (x > 0.3 && x < 0.9 && y > 0.6 - (x - 0.3) / 6 && y < 0.8 - (x - 0.3) / 6) 
      kappa = std::pow(10.0, 5.0) * (x + y) * 10.0;

    if (x > 0.1 && x < 0.5 && y > 0.1 + x && y < 0.25 + x) 
      kappa = std::pow(10.0, 5.0) * (1.0 + 7.0 * y);

    if (x > 0.5 && x < 0.9 && y > 0.15 - (x - 0.5) * 0.25 && y < 0.35 - (x - 0.5) * 0.25) 
      kappa = std::pow(10.0, 5.0) * 2.5;

    if (ix % 2 == 0 && iy % 2 == 0) 
      kappa = std::pow(10.0, 5.0) * (1.0 + ix + iy);

    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++)
      for (std::size_t j = 0; j < Traits::dimDomain; j++)
        I[i][j] = (i == j) ? kappa : 0.0;
    return I;
  }

  /// Source term: zero (doesn't affect matrix)
  typename Traits::RangeFieldType f(const typename Traits::ElementType& /*e*/, const typename Traits::DomainType& /*x*/) const
  {
    return 0.0;
  }

  /// Dirichlet boundary data: not used since we have pure Neumann BCs
  typename Traits::RangeFieldType g(const typename Traits::ElementType& /*e*/, const typename Traits::DomainType& /*x*/) const
  {
    return 0.0;
  }

  /// Boundary condition type: pure Neumann everywhere
  BC bctype(const typename Traits::IntersectionType& /*is*/, const typename Traits::IntersectionDomainType& /*x*/) const
  {
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
  }

  /// Neumann boundary data: zero (homogeneous Neumann)
  typename Traits::RangeFieldType j(const typename Traits::IntersectionType& /*is*/, const typename Traits::IntersectionDomainType& /*x*/) const
  {
    return 0.0;
  }
};

int main(int argc, char** argv)
{
  try {
    using Dune::PDELab::Backend::native;

    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    const int rank = helper.rank();
    const int size = helper.size();

    std::cout << "Rank " << rank << "/" << size << " starting test_neumann_nullspace (USE_DG=" << USE_DG << ", USE_UGGRID=" << USE_UGGRID << ")" << std::endl;

    // ========== Grid Setup ==========
    constexpr int dim = 2;
#if USE_UGGRID
    using Grid = Dune::UGGrid<dim>;
    const int gridsize = 16;
    // Use createCubeGrid to get quads (needed for QkDG)
    auto grid = Dune::StructuredGridFactory<Grid>::createCubeGrid({0, 0}, {1, 1}, {static_cast<unsigned>(gridsize), static_cast<unsigned>(gridsize)});
    
    // Load balance the grid using ParMETIS
    auto gv_temp = grid->leafGridView();
    auto part = Dune::ParMetisGridPartitioner<decltype(gv_temp)>::partition(gv_temp, helper);
    grid->loadBalance(part, 0);
#else
    using Grid = Dune::YaspGrid<dim>;
    const int gridsize = 16;
    const int grid_overlap = 0;
    Dune::FieldVector<double, dim> L = {1.0, 1.0};
    std::array<int, dim> N = {gridsize, gridsize};
    auto grid = std::make_unique<Grid>(L, N, std::bitset<dim>(0ULL), grid_overlap);
    grid->loadBalance();
#endif
    
    auto gv = grid->leafGridView();
    using GridView = decltype(gv);

    // ========== Grid Function Space Setup ==========
    using DF = Grid::ctype;
    using RF = double;

#if USE_DG
    // For DG, use AllEntitySet (includes ghosts for proper skeleton assembly)
    using ES = Dune::PDELab::AllEntitySet<GridView>;
    ES es(gv);
    
    // DG finite element map
    static constexpr int degree = 1;
    using FEM = Dune::PDELab::QkDGLocalFiniteElementMap<DF, RF, degree, dim>;
    FEM fem;
    
    // DG constraints (handle ghost elements)
    using CON = Dune::PDELab::P0ParallelGhostConstraints;
#else
    // For conforming FEM, use OverlappingEntitySet
    using ES = Dune::PDELab::OverlappingEntitySet<GridView>;
    ES es(gv);
    
    // Conforming finite element map
    using FEM = Dune::PDELab::QkLocalFiniteElementMap<ES, DF, RF, 1>;
    FEM fem(es);
    
    // Conforming constraints
    using CON = Dune::PDELab::ConformingDirichletConstraints;
#endif

    using VBE = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>;
    using GFS = Dune::PDELab::GridFunctionSpace<ES, FEM, CON, VBE>;
    GFS gfs(es, fem);
    gfs.name("u");

    // No constraints (pure Neumann problem)
    using CC = typename GFS::template ConstraintsContainer<RF>::Type;
    CC cc;
    cc.clear();

    // ========== Local Operator and Grid Operator ==========
    using Problem = PureNeumannProblem<ES, RF>;
    Problem problem;

#if USE_DG
    using LOP = Dune::PDELab::ConvectionDiffusionDG<Problem, FEM>;
#else
    using LOP = Dune::PDELab::ConvectionDiffusionFEM<Problem, FEM>;
#endif
    LOP lop(problem);

    using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
    using GO = Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, RF, RF, RF, CC, CC>;
    GO go(gfs, cc, gfs, cc, lop, MBE(27)); // 27 nonzeros for DG with degree 1

    // ========== Assemble Matrix ==========
    using Vec = Dune::PDELab::Backend::Vector<GFS, RF>;
    using Mat = typename GO::Jacobian;
    using NativeMat = Dune::PDELab::Backend::Native<Mat>;
    using NativeVec = Dune::PDELab::Backend::Native<Vec>;

    Vec x(gfs, 0.0);
    Mat A(go);
    go.jacobian(x, A);

    std::cout << "Rank " << rank << ": Matrix size = " << native(A).N() << " x " << native(A).M() << std::endl;

    // ========== Test 1: Local A * 1 (before make_additive) ==========
    // First test the raw assembled matrix - each row should sum to zero
    {
      NativeVec local_ones(native(x).N());
      NativeVec local_result(native(x).N());
      local_ones = 1.0;
      local_result = 0.0;
      
      native(A).mv(local_ones, local_result);
      
      double local_max_before = 0.0;
      int bad_rows = 0;
      for (std::size_t i = 0; i < local_result.N(); ++i) {
        local_max_before = std::max(local_max_before, std::abs(local_result[i][0]));
        if (std::abs(local_result[i][0]) > 1.0) bad_rows++;
      }
      
      std::cout << "Rank " << rank << ": BEFORE make_additive, max row sum = " << local_max_before << ", " << bad_rows << " bad rows" << std::endl;
      
      double global_max_before = 0.0;
      MPI_Allreduce(&local_max_before, &global_max_before, 1, MPI_DOUBLE, MPI_MAX, helper.getCommunicator());
      
      if (rank == 0) {
        std::cout << "\n===== BEFORE make_additive =====\n";
        std::cout << "||A_local * 1||_inf = " << global_max_before << "\n";
      }
    }

    // ========== Create Communication ==========
    auto novlp_comm = make_communication(gfs);
    
    // Count owner vs copy DOFs
    int num_owner = 0, num_copy = 0;
    const auto& pis = novlp_comm->indexSet();
    for (auto it = pis.begin(); it != pis.end(); ++it) {
      if (it->local().attribute() == Dune::OwnerOverlapCopyAttributeSet::owner) num_owner++;
      else if (it->local().attribute() == Dune::OwnerOverlapCopyAttributeSet::copy) num_copy++;
    }
    std::cout << "Rank " << rank << ": IndexSet size = " << novlp_comm->indexSet().size() 
              << ", owners = " << num_owner << ", copies = " << num_copy << std::endl;

    // ========== Test with NonOverlappingOperator BEFORE make_additive ==========
    // This tests whether the "consistent" matrix gives correct results
    {
      using Communication = std::decay_t<decltype(*novlp_comm)>;
      using Op = NonOverlappingOperator<NativeMat, NativeVec, NativeVec, Communication>;
      
      auto A_consistent = std::make_shared<NativeMat>(native(A));
      auto op_consistent = std::make_shared<Op>(A_consistent, novlp_comm);
      
      NativeVec ones_test(native(x).N());
      NativeVec result_test(native(x).N());
      ones_test = 1.0;
      result_test = 0.0;
      
      op_consistent->apply(ones_test, result_test);
      
      double raw_norm_before = result_test.two_norm();
      std::cout << "Rank " << rank << ": BEFORE make_additive, op->apply(1) raw two_norm = " << raw_norm_before << std::endl;
    }

    // ========== Make Matrix Additive (required for DG) ==========
    // The assembled matrix is "consistent" (duplicate entries on different ranks are identical).
    // We need to convert it to "additive" form where each entry is contributed by exactly one process.
#if USE_DG
    make_additive(A, *novlp_comm);
#endif

    // ========== Test 2: Local A * 1 (after make_additive) ==========
    {
      NativeVec local_ones(native(x).N());
      NativeVec local_result(native(x).N());
      local_ones = 1.0;
      local_result = 0.0;
      
      native(A).mv(local_ones, local_result);
      
      double local_max_after = 0.0;
      for (std::size_t i = 0; i < local_result.N(); ++i) {
        local_max_after = std::max(local_max_after, std::abs(local_result[i][0]));
      }
      
      double global_max_after = 0.0;
      MPI_Allreduce(&local_max_after, &global_max_after, 1, MPI_DOUBLE, MPI_MAX, helper.getCommunicator());
      
      if (rank == 0) {
        std::cout << "\n===== AFTER make_additive =====\n";
        std::cout << "||A_local * 1||_inf = " << global_max_after << "\n";
      }
    }

    // ========== Create NonOverlappingOperator ==========
    using Communication = std::decay_t<decltype(*novlp_comm)>;
    using Op = NonOverlappingOperator<NativeMat, NativeVec, NativeVec, Communication>;

    // We need to share ownership of the matrix with the operator
    auto A_ptr = std::make_shared<NativeMat>(native(A));
    auto op = std::make_shared<Op>(A_ptr, novlp_comm);

    // ========== Test: A * 1 should be 0 ==========
    NativeVec ones(native(x).N());
    NativeVec result(native(x).N());
    ones = 1.0;
    result = 0.0;

    op->apply(ones, result);

    // Report the RAW two_norm like poisson.cc does (this is what you're seeing)
    // This does NOT properly account for parallel DOF ownership
    double raw_two_norm = result.two_norm();
    std::cout << "Rank " << rank << ": RAW ||A*1||_2 = " << raw_two_norm << " (like poisson.cc reports)" << std::endl;

    // Compute local infinity norm (max absolute value)
    double local_max = 0.0;
    for (std::size_t i = 0; i < result.N(); ++i) {
      local_max = std::max(local_max, std::abs(result[i][0]));
    }

    // Compute global infinity norm
    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, helper.getCommunicator());

    // Compute local L2 norm squared (only for owned DOFs)
    double local_l2_sq = 0.0;
    novlp_comm->dot(result, result, local_l2_sq);

    // Global L2 norm is already computed by the dot product
    double global_l2 = std::sqrt(local_l2_sq);

    if (rank == 0) {
      std::cout << "\n========== TEST RESULTS ==========\n";
      std::cout << "Grid size: " << gridsize << " x " << gridsize << "\n";
      std::cout << "Number of MPI ranks: " << size << "\n";
      std::cout << "||A * 1||_inf = " << global_max << "\n";
      std::cout << "||A * 1||_2   = " << global_l2 << "\n";
    }

    // Find DOFs where the result is non-zero (for debugging)
    const double tol = 1e-10;
    std::vector<std::size_t> offending_dofs;
    for (std::size_t i = 0; i < result.N(); ++i) {
      if (std::abs(result[i][0]) > tol) {
        offending_dofs.push_back(i);
      }
    }

    // Report from each rank
    std::cout << "Rank " << rank << ": " << offending_dofs.size() << " DOFs with |A*1| > " << tol;
    if (!offending_dofs.empty() && offending_dofs.size() <= 20) {
      std::cout << " at indices: ";
      for (auto idx : offending_dofs) {
        std::cout << idx << " (val=" << result[idx][0] << ") ";
      }
    }
    std::cout << std::endl;

    // Test passes if the norm is below tolerance
    const double pass_tol = 1e-8;
    bool passed = global_max < pass_tol;

    if (rank == 0) {
      std::cout << "\n========== " << (passed ? "PASS" : "FAIL") << " ==========\n";
      if (!passed) {
        std::cout << "Expected ||A * 1||_inf < " << pass_tol << " but got " << global_max << "\n";
        std::cout << "This indicates a problem with matrix assembly or parallel communication.\n";
      }
    }

    return passed ? 0 : 1;
  }
  catch (const Dune::Exception& e) {
    std::cerr << "Caught Dune exception: " << e << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  catch (const std::exception& e) {
    std::cerr << "Caught std exception: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 1;
  }
}
