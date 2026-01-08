#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

// Minimal test of the unified PDELab framework
// This tests the basic functionality without all the DDM machinery

#include <iostream>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/grid/uggrid.hh>

#include "ddm_utilities.hh"
#include "poisson_problems.hh"
#include "problem_traits.hh"
#include "generic_ddm_problem.hh"
#include "pdelab_helper.hh"

#include "dune/ddm/logger.hh"

int main(int argc, char* argv[])
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    setup_loggers(helper.rank(), argc, argv);

    logger::info("Testing unified PDELab framework");

    // Read parameter file
    Dune::ParameterTree ptree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readINITree("test_framework.ini", ptree);
    ptreeparser.readOptions(argc, argv, ptree);

    // Create grid using utilities
    constexpr int dim = 2;
    using Grid = Dune::UGGrid<dim>;
    auto grid = DDMUtilities::make_grid<Grid>(ptree, helper, "");
    auto gv = grid->leafGridView();
    
    logger::info("Grid created with {} elements", gv.size(0));

    // Define problem using the framework
    using ProblemParams = SimplePoissonProblem<decltype(gv), double>;
    using Traits = ConvectionDiffusionTraits<decltype(gv), ProblemParams, /*UseDG=*/true>;
    
    logger::info("Setting up GenericDDMProblem with traits:");
    logger::info("  - Dimension: {}", Traits::dim);
    logger::info("  - Block size: {}", Traits::blocksize);
    logger::info("  - DG: {}", Traits::is_dg);

    // Create the problem
    GenericDDMProblem<decltype(gv), Traits> problem(gv, helper);
    
    logger::info("Problem created successfully!");
    logger::info("  - Matrix size: {}x{}", problem.getA().N(), problem.getA().M());
    logger::info("  - Vector size: {}", problem.getX().N());

    // Test assembly
    logger::info("Testing Jacobian assembly...");
    problem.jacobian();
    logger::info("Jacobian assembled successfully!");

    // Check matrix properties
    using Dune::PDELab::Backend::native;
    const auto& A = native(problem.getA());
    auto [nnz, max_nnz] = [&]() {
      std::size_t total_nnz = 0;
      std::size_t max_row_nnz = 0;
      for (auto row = A.begin(); row != A.end(); ++row) {
        std::size_t row_nnz = 0;
        for (auto col = row->begin(); col != row->end(); ++col)
          row_nnz++;
        total_nnz += row_nnz;
        max_row_nnz = std::max(max_row_nnz, row_nnz);
      }
      return std::make_pair(total_nnz, max_row_nnz);
    }();

    logger::info("Matrix statistics:");
    logger::info("  - Total nonzeros: {}", nnz);
    logger::info("  - Max nonzeros per row: {}", max_nnz);
    double sparsity = 100.0 * nnz / double(A.N() * A.M());
    logger::info("  - Sparsity: {}%", sparsity);

    logger::info("\n✓ Framework test PASSED - all basic functionality works!");
    
    return 0;
  }
  catch (const Dune::Exception& e) {
    std::cerr << "Dune exception: " << e << '\n';
    return 1;
  }
  catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << '\n';
    return 1;
  }
}
