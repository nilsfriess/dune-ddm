#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <algorithm>
#include <limits>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
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

    // Create grid
    constexpr int dim = 2;
    using Grid = Dune::YaspGrid<dim>;
    unsigned int size = ptree.get("size", 64);
    auto grid = Dune::StructuredGridFactory<Grid>::createCubeGrid({0, 0}, {1, 1}, {size, size});
    auto gv = grid->leafGridView();

    Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> A;
    Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>> B;
    Dune::loadMatrixMarket(A, "A.mtx");
    Dune::loadMatrixMarket(B, "B.mtx");

    std::cout << "Matrices loaded: A has size " << A.N() << " x " << A.M() << " and " << A.nonzeroes() << " nonzeros\n";
    std::cout << "                 B has size " << B.N() << " x " << B.M() << " and " << B.nonzeroes() << " nonzeros\n";

    const auto normalise = [](auto& v) { v *= (1. / v.two_norm()); };

    const auto error = [](const auto& x, const auto& y) {
      if (x.size() != y.size()) return std::numeric_limits<double>::max();

      double max_err = 0;
      for (std::size_t i = 0; i < x.size(); ++i) {
        auto err = 1 - std::abs(x[i] * y[i]);
        if (err > max_err) max_err = err;
      }
      return max_err;
    };

    logger::set_level(logger::Level::trace);
    Dune::Timer timer;

    auto& subtree = ptree.sub("eigensolver");
    subtree["tolerance"] = "1e-12";
    subtree["nev"] = "32";

    subtree["type"] = "Spectra";
    timer.reset();
    auto spectra_evecs = solve_gevp(A, B, subtree);
    std::for_each(spectra_evecs.begin(), spectra_evecs.end(), normalise);
    std::cout << "Spectra took " << timer.elapsed() << "s\n";

    subtree["type"] = "SubspaceIteration";
    timer.reset();
    auto subspace_it_evecs = solve_gevp(A, B, subtree);
    std::for_each(subspace_it_evecs.begin(), subspace_it_evecs.end(), normalise);
    std::cout << "Subspace iteration took " << timer.elapsed() << "s\n";

    auto err = error(spectra_evecs, subspace_it_evecs);
    if (err < 1e-8) logger::info("Results of Spectra and SubspaceIteration eigensolvers are sufficiently close, error is: {}", err);
    else {
      logger::error("Results of Spectra and SubspaceIteration eigensolvers are not close, error is: {}", err);
      return 1;
    }

  // RAES experimental solver intentionally not part of the repo; skip RAES run

    Logger::get().report(MPI_COMM_SELF);
  }
  catch (Dune::Exception& e) {
    std::cout << "Error in DUNE: " << e.what() << "\n";
    return 1;
  }
  catch (std::exception& e) {
    std::cout << "Error: " << e.what() << "\n";
    return 2;
  }
  return 0;
}
