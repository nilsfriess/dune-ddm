#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <algorithm>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include "problem.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/timer.hh>
#include <dune/ddm/logger.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/solvers.hh>
#pragma GCC diagnostic pop

#include <dune/ddm/eigensolvers/blockmultivector.hh>
#include <dune/ddm/eigensolvers/eigensolvers.hh>
#include <dune/ddm/eigensolvers/subspace_iteration.hh>
#include <experimental/simd>

template <typename Real, typename Matrix, typename UMFSolver, typename BlockSolver>
bool run_correctness_test(const Matrix& A, UMFSolver& umfsolver, BlockSolver& block_solver)
{
  using VReal = std::experimental::native_simd<Real>;
  constexpr std::size_t simd_width = VReal::size();
  constexpr std::size_t num_vectors = 4 * simd_width; // Use native SIMD width
  constexpr std::size_t num_tests = 5;

  std::cout << "\n=== Correctness Test (SIMD width = " << simd_width << ") ===\n";

  // Block vectors
  BlockMultiVector<Real, simd_width> B(A.N(), num_vectors);
  BlockMultiVector<Real, simd_width> X_block(A.N(), num_vectors);

  // Standard vectors
  using Vec = Dune::BlockVector<Dune::FieldVector<Real, 1>>;
  std::vector<Vec> bs(num_vectors, Vec(A.N()));
  std::vector<Vec> xs_standard(num_vectors, Vec(A.N()));

  double total_error = 0.0;
  double max_error = 0.0;

  for (std::size_t test = 0; test < num_tests; ++test) {
    // Initialize with random data
    B.set_random();
    for (std::size_t j = 0; j < B.blocks(); ++j) {
      auto b_block = B.block_view(j);
      for (std::size_t i = 0; i < A.N(); ++i)
        for (std::size_t vec = j * b_block.cols(); vec < (j + 1) * b_block.cols(); ++vec) bs[vec][i] = b_block(i, vec - j * b_block.cols());
    }

    // Solve with standard solver
    for (std::size_t vec = 0; vec < num_vectors; ++vec) {
      xs_standard[vec] = 0;
      Dune::InverseOperatorResult res;
      umfsolver.apply(xs_standard[vec], bs[vec], res);
    }

    // Solve with block solver
    X_block.set_zero();
    block_solver.solve(X_block, B);

    // Compare results
    auto x_block = X_block.block_view(0);
    double test_error = 0.0;
    for (std::size_t block = 0; block < X_block.blocks(); ++block) {
      auto x_block = X_block.block_view(block);
      for (std::size_t vec = 0; vec < x_block.cols(); ++vec) {
        std::size_t global_vec = block * x_block.cols() + vec;
        for (std::size_t i = 0; i < A.N(); ++i) {
          double diff = x_block(i, vec) - xs_standard[global_vec][i];
          test_error += diff * diff;
          max_error = std::max(max_error, std::abs(diff));
        }
      }
    }
    total_error += std::sqrt(test_error);
  }

  std::cout << "Average error: " << (total_error / num_tests) << "\n";
  std::cout << "Maximum error: " << max_error << "\n";

  return max_error < 1e-12;
}

template <typename Real, typename Matrix, typename UMFSolver, typename BlockSolver>
void run_performance_test(const Matrix& A, UMFSolver& umfsolver, BlockSolver& block_solver, std::size_t num_solves = 1000)
{
  using VReal = std::experimental::native_simd<Real>;
  constexpr std::size_t simd_width = 2 * VReal::size();
  constexpr std::size_t num_vectors = 2 * simd_width; // Use exactly the SIMD width

  std::cout << "\n=== Performance Test ===\n";
  std::cout << "SIMD width: " << simd_width << ", Block vectors: " << num_vectors << "\n";
  std::cout << "Number of solves: " << num_solves << "\n";

  // Block vectors
  BlockMultiVector<Real, simd_width> B(A.N(), num_vectors);
  BlockMultiVector<Real, simd_width> X_block(A.N(), num_vectors);

  // Standard vectors
  using Vec = Dune::BlockVector<Dune::FieldVector<Real, 1>>;
  std::vector<Vec> bs(num_vectors, Vec(A.N()));
  std::vector<Vec> xs_standard(num_vectors, Vec(A.N()));

  // Initialize with random data
  B.set_random();
  for (std::size_t j = 0; j < B.blocks(); ++j) {
    auto b_block = B.block_view(j);
    for (std::size_t i = 0; i < A.N(); ++i)
      for (std::size_t vec = j * b_block.cols(); vec < (j + 1) * b_block.cols(); ++vec) bs[vec][i] = b_block(i, vec - j * b_block.cols());
  }

  // Warm up
  X_block.set_zero();
  block_solver.solve(X_block, B);
  for (std::size_t vec = 0; vec < num_vectors; ++vec) {
    xs_standard[vec] = 0;
    Dune::InverseOperatorResult res;
    umfsolver.apply(xs_standard[vec], bs[vec], res);
  }

  // Time standard solver
  Dune::Timer timer;
  timer.start();
  for (std::size_t solve = 0; solve < num_solves; ++solve) {
    for (std::size_t vec = 0; vec < num_vectors; ++vec) {
      xs_standard[vec] = 0;
      umfsolver.apply(&xs_standard[vec][0][0], &bs[vec][0][0]);
    }
  }
  double time_standard = timer.elapsed();

  // Time block solver
  timer.reset();
  for (std::size_t solve = 0; solve < num_solves; ++solve) {
    X_block.set_zero();
    block_solver.solve(X_block, B);
  }
  double time_block = timer.elapsed();

  std::cout << "Standard solver: " << time_standard << " s (" << (time_standard * 1000.0 / (num_solves * num_vectors)) << " ms per solve)\n";
  std::cout << "Block solver:    " << time_block << " s (" << (time_block * 1000.0 / num_solves) << " ms per " << num_vectors << " solves)\n";
  std::cout << "Speedup: " << (time_standard / time_block) << "x\n";
}

int main(int argc, char* argv[])
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    if (helper.size() != 1) {
      logger::error("This test must be ran sequentially");
      return 1;
    }

    Dune::ParameterTree ptree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readOptions(argc, argv, ptree);

    // Create grid - use larger size for performance testing
    constexpr int dim = 2;
    using Grid = Dune::YaspGrid<dim>;
    unsigned int grid_size = ptree.get("grid_size", 50); // Much larger for performance test

    std::cout << "Creating " << grid_size << "x" << grid_size << " grid...\n";
    auto grid = Dune::StructuredGridFactory<Grid>::createCubeGrid({0, 0}, {1, 1}, {grid_size, grid_size});
    auto gv = grid->leafGridView();

    // Define the Real type - can be changed here to test different precisions
    using Real = double; // Change this to float, long double, etc.

    std::cout << "Assembling matrix...\n";
    auto A = *assemble_problem(gv);

    std::cout << "Factoring matrix...\n";
    Dune::UMFPack umfsolver(A);
    UMFPackMultivecSolver block_solver(A);

    std::cout << "Matrix size: " << A.N() << "x" << A.M() << " (" << A.nonzeroes() << " nonzeros)\n";

    using VReal = std::experimental::native_simd<Real>;
    std::cout << "Native SIMD width: " << VReal::size() << " (Real = " << typeid(Real).name() << ")\n";

    // Run correctness test
    auto success = run_correctness_test<Real>(A, umfsolver, block_solver);

    // Run performance test
    if (ptree.get("run_perf_test", false)) {
      std::size_t num_solves = ptree.get("num_solves", grid_size > 20 ? 100 : 1000);
      run_performance_test<Real>(A, umfsolver, block_solver, num_solves);
    }

    return success ? 0 : 1;
  }
  catch (Dune::Exception& e) {
    std::cout << "Error in DUNE: " << e.what() << "\n";
    return 1;
  }
  catch (std::exception& e) {
    std::cout << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
