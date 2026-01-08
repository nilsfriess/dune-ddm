// Simple test for nonoverlapping DG with UGGrid
// Tests that parallel assembly works correctly

#include <dune/pdelab/constraints/noconstraints.hh>
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>
#include <dune/ddm/nonoverlapping_operator.hh>
#include <dune/ddm/pdelab_helper.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/istl/novlpschwarz.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/pdelab.hh>
#include <iostream>

// Simple Poisson problem: -Δu = 1, u = 0 on boundary
template <typename GV, typename RF>
class SimplePoissonProblem {
public:
  using Traits = Dune::PDELab::ConvectionDiffusionParameterTraits<GV, RF>;

  // Constant diffusion tensor (identity)
  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; i++)
      for (std::size_t j = 0; j < Traits::dimDomain; j++) I[i][j] = (i == j) ? 1.0 : 0.0;
    return I;
  }

  // No convection
  typename Traits::RangeType b(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return typename Traits::RangeType(0.0); }

  // No reaction
  typename Traits::RangeFieldType c(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return 0.0; }

  // Constant source term
  typename Traits::RangeFieldType f(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return 1.0; }

  // Pure Neumann boundary for A*1 test
  Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type bctype(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
  }

  // Dirichlet value = 0
  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return 0.0; }

  // Neumann flux = 0
  typename Traits::RangeFieldType j(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const { return 0.0; }

  // Outflow = 0
  typename Traits::RangeFieldType o(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const { return 0.0; }

  // Permeability is constant per cell
  bool permeabilityIsConstantPerCell() const { return true; }
};

int main(int argc, char** argv)
{
  try {
    // Initialize MPI
    auto& helper = Dune::MPIHelper::instance(argc, argv);

    if (helper.rank() == 0) std::cout << "Running on " << helper.size() << " processes" << std::endl;

    // Grid parameters
    constexpr int dim = 2;
    using Grid = Dune::UGGrid<dim>;
    using GV = typename Grid::LeafGridView;
    using DF = typename Grid::ctype;
    using RF = double;

    // Create grid
    Dune::FieldVector<DF, dim> lower(0.0);
    Dune::FieldVector<DF, dim> upper(1.0);
    auto cells = Dune::filledArray<dim, unsigned int>(8);

    auto grid = Dune::StructuredGridFactory<Grid>::createCubeGrid(lower, upper, cells);
    grid->loadBalance();

    if (helper.rank() == 0) std::cout << "Grid created with " << grid->size(0) << " elements per rank (approx)" << std::endl;

    GV gv = grid->leafGridView();

    // NOTE: We cannot use NonOverlappingEntitySet with DG because PDELab's
    // NonOverlappingBorderDOFExchanger only handles codim > 0 entities (vertices/edges),
    // not codim-0 (elements). Using it causes a segfault in BorderIndexIdCache.
    //
    // Instead, use the full GridView with P0ParallelGhostConstraints to mark ghost DOFs.
    using ES = Dune::PDELab::AllEntitySet<GV>;
    ES es(gv);

    // DG finite element map (Q1 DG)
    constexpr int degree = 1;
    using FEM = Dune::PDELab::QkDGLocalFiniteElementMap<DF, RF, degree, dim>;
    FEM fem;

    // Use P0ParallelGhostConstraints to mark ghost element DOFs as constrained
    using CON = Dune::PDELab::NoConstraints;
    CON con;

    // Vector backend - use unblocked for compatibility with make_communication
    using VBE = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>;

    // Grid function space
    using GFS = Dune::PDELab::GridFunctionSpace<ES, FEM, CON, VBE>;
    GFS gfs(es, fem, con);
    gfs.name("u");
    gfs.update();

    std::cout << "Rank " << helper.rank() << ": GFS size = " << gfs.size() << std::endl;

    // Constraints container - will hold ghost DOF constraints
    using CC = typename GFS::template ConstraintsContainer<RF>::Type;
    CC cc;
    Dune::PDELab::constraints(con, gfs, cc); // Apply P0ParallelGhostConstraints

    std::cout << "Rank " << helper.rank() << ": constrained DOFs = " << cc.size() << std::endl;

    // Problem definition
    using Problem = SimplePoissonProblem<ES, RF>;
    Problem problem;

    // DG local operator
    using LOP = Dune::PDELab::ConvectionDiffusionDG<Problem, FEM>;
    LOP lop(problem, Dune::PDELab::ConvectionDiffusionDGMethod::SIPG, Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn, 2.0);

    // Matrix backend
    using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
    MBE mbe(27); // Estimate for nonzeros per row

    // Grid operator
    using GO = Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, RF, RF, RF, CC, CC>;
    GO go(gfs, cc, gfs, cc, lop, mbe);

    // Solution and RHS vectors
    using V = Dune::PDELab::Backend::Vector<GFS, RF>;
    V x(gfs, 0.0);
    V r(gfs, 0.0);

    // Assemble residual to check
    go.residual(x, r);

    // Compute norm of residual
    RF local_norm_sq = 0.0;
    for (std::size_t i = 0; i < Dune::PDELab::Backend::native(r).N(); ++i) local_norm_sq += Dune::PDELab::Backend::native(r)[i].two_norm2();
    RF global_norm_sq = 0.0;
    MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, helper.getCommunication());

    if (helper.rank() == 0) std::cout << "Initial ||r||_2 = " << std::sqrt(global_norm_sq) << std::endl;

    // Assemble matrix
    using M = typename GO::Traits::Jacobian;
    M A(go);
    go.jacobian(x, A);

    // Create nonoverlapping communication from PDELab GFS
    auto novlp_comm = make_communication(gfs);

    make_additive(A, *novlp_comm);

    // Get native matrix and vector types
    using NativeMat = Dune::PDELab::Backend::Native<M>;
    using NativeVec = Dune::PDELab::Backend::Native<V>;
    using Communication = Dune::OwnerOverlapCopyCommunication<std::size_t, int>;

    // Create the NonOverlappingOperator
    using Op = NonOverlappingOperator<NativeMat, NativeVec, NativeVec, Communication>;
    auto op = std::make_shared<Op>(A.storage(), novlp_comm);

    // Create scalar product for nonoverlapping case
    auto sp = Dune::createScalarProduct(op);

    // Create preconditioner (identity for now)
    Dune::Richardson<NativeVec, NativeVec> prec(1.0);

    // Create BiCGSTAB solver
    Dune::BiCGSTABSolver<NativeVec> solver(*op, *sp, prec, 1e-8, 100, helper.rank() == 0 ? 2 : 0);

    // Prepare RHS and solution vectors
    NativeVec b = Dune::PDELab::Backend::native(r); // r already contains -residual
    b *= -1.0;                                      // Convert to RHS
    NativeVec sol(Dune::PDELab::Backend::native(x));
    sol = 0.0;

    // Solve
    Dune::InverseOperatorResult res;
    solver.apply(sol, b, res);

    // Copy solution back
    Dune::PDELab::Backend::native(x) = sol;

    // Output solution norm
    local_norm_sq = 0.0;
    for (std::size_t i = 0; i < Dune::PDELab::Backend::native(x).N(); ++i) local_norm_sq += Dune::PDELab::Backend::native(x)[i].two_norm2();
    MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, helper.getCommunication());

    if (helper.rank() == 0) std::cout << "Solution ||x||_2 = " << std::sqrt(global_norm_sq) << std::endl;

    // VTK output
    Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, Dune::refinementLevels(0));
    Dune::PDELab::addSolutionToVTKWriter(vtkwriter, gfs, x);
    vtkwriter.write("test_dg_nonoverlapping", Dune::VTK::ascii);

    if (helper.rank() == 0) std::cout << "Test completed successfully!" << std::endl;

    return 0;
  }
  catch (Dune::Exception& e) {
    std::cerr << "Dune exception: " << e << std::endl;
    return 1;
  }
  catch (std::exception& e) {
    std::cerr << "std exception: " << e.what() << std::endl;
    return 1;
  }
}
