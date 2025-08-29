#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dune-pdelab-config.hh>

#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/istl/cholmod.hh>
#include <dune/istl/superlu.hh>
#include <dune/istl/umfpack.hh>

#include <iostream>
#include <type_traits>

#include "logger.hh"
#include "poisson.hh"
#include "strumpack.hh"

int main(int argc, char **argv)
{
  const auto &helper = Dune::MPIHelper::instance(argc, argv);
  if (helper.size() != 1) {
    std::cout << "Must be run sequentially\n";
    return 1;
  }
  setup_loggers(helper.rank(), argc, argv);

  using Grid = Dune::UGGrid<2>;
  auto grid = Dune::StructuredGridFactory<Grid>::createSimplexGrid({0, 0}, {1, 1}, {511, 511});
  auto gv = grid->leafGridView();
  spdlog::info("Created grid with {} elements", gv.size(0));

  PoissonProblem poisson(gv, helper);
  poisson.jacobian();
  const auto &A = poisson.getA();
  spdlog::info("Assembled matrix of size {}x{}", A.N(), A.M());

  auto b = poisson.getD();
  auto &x = poisson.getX();
  auto v = b;

  Dune::Timer timer;
  double factor = 0;
  double solve = 0;
  int verbose = 0;
  {
    spdlog::info("=== UMFPACK === ");
    v = 0;

    timer.start();
    Dune::UMFPack solver(A, verbose);
    factor = timer.elapsed();

    timer.reset();
    Dune::InverseOperatorResult res;
    solver.apply(v, b, res);
    solve = timer.elapsed();

    spdlog::info("Factorisation took: {:.5f}s", factor);
    spdlog::info("Solving took:       {:.5f}s", solve);
  }

  {
    spdlog::info("=== CHOLMOD === ");
    v = 0;

    timer.start();
    Dune::Cholmod<std::remove_cvref_t<decltype(x)>> solver;
    solver.setMatrix(A);
    factor = timer.elapsed();

    timer.reset();
    Dune::InverseOperatorResult res;
    solver.apply(v, b, res);
    solve = timer.elapsed();

    spdlog::info("Factorisation took: {:.5f}s", factor);
    spdlog::info("Solving took:       {:.5f}s", solve);
  }

  {
    spdlog::info("=== SuperLU === ");
    v = 0;

    timer.start();
    Dune::SuperLU solver(A, verbose > 0);
    factor = timer.elapsed();

    timer.reset();
    Dune::InverseOperatorResult res;
    solver.apply(v, b, res);
    solve = timer.elapsed();

    spdlog::info("Factorisation took: {:.5f}s", factor);
    spdlog::info("Solving took:       {:.5f}s", solve);
  }

  {
    spdlog::info("=== STRUMPACK [CPU] === ");
    v = 0;

    timer.start();
    Dune::STRUMPACK<std::remove_cvref_t<decltype(A)>> solver(verbose > 0);
    solver.setMatrix(A);
    solver.get_options().disable_gpu();
    factor = timer.elapsed();

    timer.reset();
    Dune::InverseOperatorResult res;
    solver.apply(v, b, res);
    solve = timer.elapsed();

    spdlog::info("Factorisation took: {:.5f}s", factor);
    spdlog::info("Solving took:       {:.5f}s", solve);
  }

  {
    spdlog::info("=== STRUMPACK [GPU] === ");
#if defined(STRUMPACK_USE_GPU)
    v = 0;

    timer.start();
    Dune::STRUMPACK<std::remove_cvref_t<decltype(A)>> solver(verbose > 0);
    // solver.get_options().enable_gpu();
    solver.setMatrix(A);
    factor = timer.elapsed();

    timer.reset();
    Dune::InverseOperatorResult res;
    solver.apply(v, b, res);
    solve = timer.elapsed();

    spdlog::info("Factorisation took: {:.5f}s", factor);
    spdlog::info("Solving took:       {:.5f}s", solve);
#else
    spdlog::warn("STRUMPACK compiled without GPU support");
#endif
  }
}
