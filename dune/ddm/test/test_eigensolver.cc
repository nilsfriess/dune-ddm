#include "dune/ddm/eigensolvers/inner_products.hh"

#include <type_traits>
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

#include "matrix_symmetry_helper.hh"
#include "mmloader/mmloader.h"

#include <dune/ddm/eigensolvers/blockmultivector.hh>
#include <dune/ddm/eigensolvers/eigensolvers.hh>
#include <dune/ddm/eigensolvers/orthogonalisation.hh>
#include <dune/ddm/eigensolvers/subspace_iteration.hh>
#include <experimental/simd>
#include <slepceps.h>

int main(int argc, char* argv[])
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    if (helper.size() != 1) {
      logger::error("This test must be ran sequentially");
      return 1;
    }

    PetscCall(SlepcInitialize(&argc, &argv, nullptr, nullptr));

    Dune::ParameterTree ptree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readOptions(argc, argv, ptree);

    // Create grid
    constexpr int dim = 2;
    using Grid = Dune::YaspGrid<dim>;
    unsigned int size = ptree.get("size", 64);
    auto grid = Dune::StructuredGridFactory<Grid>::createCubeGrid({0, 0}, {1, 1}, {size, size});
    auto gv = grid->leafGridView();

    auto Afile = ptree.get("A_filename", "A.mtx");
    auto Bfile = ptree.get("B_filename", "B.mtx");

    auto A = std::make_shared<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>>();
    auto B = std::make_shared<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>>();
    Dune::loadMatrixMarket(*A, Afile);
    Dune::loadMatrixMarket(*B, Bfile);

    // Check and ensure matrix symmetry for proper eigenvalue computation
    matrix_symmetry_helper::ensure_matrix_symmetry(A, B, 1e-12, false);

    if (false) {
      // Solve problem in SLEPc

      Mat A;
      Mat B;
      EPS eps;

      PetscCall(MatCreateFromMTX(&A, Afile.c_str(), PETSC_TRUE));
      PetscCall(MatCreateFromMTX(&B, Bfile.c_str(), PETSC_TRUE));

      PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
      PetscCall(EPSSetOperators(eps, A, B));
      PetscCall(EPSSetProblemType(eps, EPS_GHEP));
      PetscCall(EPSSetFromOptions(eps));
      PetscCall(EPSSetUp(eps));
      PetscCall(EPSSolve(eps));

      PetscCall(EPSDestroy(&eps));
      PetscCall(MatDestroy(&A));
      PetscCall(MatDestroy(&B));
    }

    if (true) {
      orthogonalisation::MatrixInnerProduct<std::remove_cvref_t<decltype(*B)>, BlockMultiVector<double, 8>> ip(B);

      std::cout << "Matrices loaded: A has size " << A->N() << " x " << A->M() << " and " << A->nonzeroes() << " nonzeros\n";
      std::cout << "                 B has size " << B->N() << " x " << B->M() << " and " << B->nonzeroes() << " nonzeros\n";

      const auto normalise = [&ip](auto& v) {
        auto norm = ip.dot(v, v);
        v /= norm;
      };

      const auto error = [&](const auto& x, const auto& y) {
        if (x.size() != y.size()) return std::numeric_limits<double>::max();

        double max_err = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
          auto err = 1 - std::abs(ip.dot(x[i], y[i]));

          std::cout << "  Eigenvector " << i << " error: " << err << "\n";

          if (err > max_err) max_err = err;
        }
        return max_err;
      };

      logger::set_level(logger::Level::trace);
      Dune::Timer timer;

      auto& subtree = ptree.sub("eigensolver");
      // subtree["tolerance"] = "1e-10";  // Commented out to allow command-line override

      subtree["type"] = "Spectra";
      timer.reset();
      auto spectra_evecs = solve_gevp(A, B, subtree);
      std::for_each(spectra_evecs.begin(), spectra_evecs.end(), normalise);
      std::cout << "Spectra took " << timer.elapsed() << "s" << std::endl;

      // {
      //   subtree["type"] = "SubspaceIteration";
      //   timer.reset();
      //   auto subspace_it_evecs = solve_gevp(A, B, subtree);
      //   std::for_each(subspace_it_evecs.begin(), subspace_it_evecs.end(), normalise);
      //   std::cout << "Subspace iteration took " << timer.elapsed() << "s\n";

      //   auto err = error(spectra_evecs, subspace_it_evecs);
      //   if (err < 1e-8) logger::info("Results of Spectra and standard subspace iteration eigensolvers are sufficiently close, error is: {}", err);
      //   else logger::error("Results of Spectra and standard subspace iteration eigensolvers are not close, error is: {}", err);
      // }

      /*{
        subtree["type"] = "SRRIT";
        timer.reset();
        auto srrit_evecs = solve_gevp(A, B, subtree);
        std::for_each(srrit_evecs.begin(), srrit_evecs.end(), normalise);
        std::cout << "SRRIT subspace iteration took " << timer.elapsed() << "s\n";

        // auto err = error(spectra_evecs, srrit_evecs);
        // if (err < 1e-8) logger::info("Results of Spectra and SRRIT subspace iteration eigensolvers are sufficiently close, error is: {}", err);
        // else logger::error("Results of Spectra and SRRIT subspace iteration eigensolvers are not close, error is: {}", err);
      }*/

      if (false) {
        using Real = double;
        constexpr std::size_t blocksize = 8;
        using Mat = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
        const std::size_t m = 8; // number of blocks after the extension

        ShiftInvertEigenproblem<Mat, blocksize> evp(*A, B, subtree.get("shift", 1e-4));
        using BMV = typename ShiftInvertEigenproblem<Mat, blocksize>::BlockMultiVec;
        using DenseMatView = typename BMV::BlockMatrixBlockView;

        BMV Q(A->N(), m * blocksize);
        Q.set_random();

        evp.apply(Q, Q);

        std::vector<std::vector<Real>> alpha_data(m);
        std::vector<std::vector<Real>> beta_data(m);
        std::vector<DenseMatView> alpha_coeffs;
        std::vector<DenseMatView> beta_coeffs;
        for (std::size_t i = 0; i < m; ++i) {
          alpha_data[i].resize(blocksize * blocksize);
          beta_data[i].resize(blocksize * blocksize);
          alpha_coeffs.emplace_back(alpha_data[i].data());
          beta_coeffs.emplace_back(beta_data[i].data());
        }

        // Parse muscle algorithm from command line
        std::string muscle_str = ptree.get("muscle", "CholQR");
        WithinBlocks muscle = WithinBlocks::ShiftedCholQR3; // Default value
        if (muscle_str == "CholQR") {
          muscle = WithinBlocks::CholQR;
          std::cout << "Using CholQR muscle algorithm\n";
        }
        else if (muscle_str == "CholQR2") {
          muscle = WithinBlocks::CholQR2;
          std::cout << "Using CholQR2 muscle algorithm\n";
        }
        else if (muscle_str == "PreCholQR") {
          muscle = WithinBlocks::PreCholQR;
          std::cout << "Using PreCholQR muscle algorithm\n";
        }
        else if (muscle_str == "ShiftedCholQR3") {
          muscle = WithinBlocks::ShiftedCholQR3;
          std::cout << "Using ShiftedCholQR3 muscle algorithm\n";
        }
        else {
          std::cout << "Unknown muscle algorithm: " << muscle_str << ", defaulting to CholQR\n";
          muscle = WithinBlocks::CholQR;
        }

        orthogonalisation::BlockOrthogonalisation orth(BetweenBlocks::ModifiedGramSchmidt, muscle, evp.get_inner_product());
        orth.orthonormalise_block_against_previous(Q, 0);

        if (!lanczos_extend_decomposition(evp, Q, 0, Q.blocks() - 1, orth, alpha_coeffs, beta_coeffs)) std::cout << "ERROR: Lanczos extension step failed\n";

        // For a random start, build the tridiagonal matrix and compute its eigenvalues
        std::vector<Real> T_data(Q.cols() * Q.cols(), 0); // Dense column-major matrix
        build_tridiagonal_matrix(alpha_coeffs, beta_coeffs, T_data);

        const std::size_t T_size = m * blocksize;
        const int T_size_int = static_cast<int>(T_size);

        // Compute eigenvalues of T using LAPACK
        std::vector<Real> eigenvalues(T_size);
        int info = lapacke::syev(LAPACK_COL_MAJOR, 'N', 'U', T_size_int, T_data.data(), T_size_int, eigenvalues.data());

        if (info != 0) { std::cout << "FAILED: LAPACK syev failed with info = " << info << "\n"; }
        else {
          // Sort eigenvalues (LAPACK returns them in ascending order, but let's be sure)
          auto lambdas = evp.transform_eigenvalues(eigenvalues);
          std::sort(lambdas.begin(), lambdas.end(), [](auto&& x, auto&& y) { return std::abs(x) < std::abs(y); });

          for (std::size_t i = 0; i < std::min(T_size, std::size_t(32)); ++i) std::cout << lambdas[i] << " ";
          std::cout << std::endl;
        }
      }

      if (1 + 1 == 2) {
        subtree["type"] = "KrylovSchur";
        timer.reset();
        auto krylov_schur_evecs = solve_gevp(A, B, subtree);
        std::for_each(krylov_schur_evecs.begin(), krylov_schur_evecs.end(), normalise);
        std::cout << "KRYLOVSCHUR subspace iteration took " << timer.elapsed() << "s\n";

        auto err = error(spectra_evecs, krylov_schur_evecs);
        if (err < 1e-8) logger::info("Results of Spectra and KRYLOVSCHUR subspace iteration eigensolvers are sufficiently close, error is: {}", err);
        else logger::error("Results of Spectra and KRYLOVSCHUR subspace iteration eigensolvers are not close, error is: {}", err);
      }

      // Logger::get().report(MPI_COMM_SELF);
    }
  }
  catch (Dune::Exception& e) {
    std::cout << "Error in DUNE: " << e.what() << "\n";
    return 1;
  }
  catch (std::exception& e) {
    std::cout << "Error: " << e.what() << "\n";
    return 2;
  }

  PetscCall(SlepcFinalize());
  return 0;
}
