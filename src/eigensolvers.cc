#include <dune-istl-config.hh>

#include <mpi.h>

#include <dune/common/parametertree.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/istl/foreach.hh>
#include <vector>

#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include "coarsespaces/eigensolvers.hh"

#include <spdlog/spdlog.h>

#include <lobpcg.h>

#include "coarsespaces/fakemultivec.hh"
#include "coarsespaces/multivec.hh"

extern "C" { // Tell the linker that LAPACK functions are mangled as C functions
BlopexInt dsygv_(BlopexInt *itype, char *jobz, char *uplo, BlopexInt *n, double *a, BlopexInt *lda, double *b, BlopexInt *ldb, double *w, double *work, BlopexInt *lwork, BlopexInt *info);
BlopexInt dpotrf_(char *uplo, BlopexInt *n, double *a, BlopexInt *lda, BlopexInt *info);
}

std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> solveGEVP(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A, const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &B,
                                                                       Eigensolver eigensolver, const Dune::ParameterTree &ptree)
{
  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;
  using Mat = Dune::BCRSMatrix<double>;
  std::vector<Vec> eigenvectors;

  const double tolerance = ptree.get("eigensolver_tolerance", 1e-5);

  if (eigensolver == Eigensolver::Spectra) {
    using Triplet = Eigen::Triplet<double>;
    auto N = static_cast<Eigen::Index>(A.N());
    Eigen::SparseMatrix<double> A_(N, N);
    Eigen::SparseMatrix<double> B_(N, N);

    std::vector<Triplet> triplets;
    triplets.reserve(A.nonzeroes());
    Dune::flatMatrixForEach(A, [&](auto &&entry, std::size_t i, std::size_t j) {
      if (std::abs(entry) > 0) {
        triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry});
      }
    });
    A_.setFromTriplets(triplets.begin(), triplets.end());

    triplets.clear();
    triplets.reserve(A.nonzeroes());
    Dune::flatMatrixForEach(B, [&](auto &&entry, std::size_t i, std::size_t j) {
      if (std::abs(entry) > 0) {
        triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry});
      }
    });
    B_.setFromTriplets(triplets.begin(), triplets.end());

    A_.makeCompressed();
    B_.makeCompressed();

    spdlog::get("all_ranks")->debug("Solving eigenproblem Ax=lBx of size {} with nnz(A) = {}, nnz(B) = {}", A_.rows(), A_.nonZeros(), B_.nonZeros());

    using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = Spectra::SparseSymMatProd<double>;
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    OpType op(A_, B_);
    BOpType Bop(B_);

    auto nev = ptree.get("eigensolver_nev", 10);
    auto nev_max = nev;
    double threshold = -1;
    if (ptree.hasKey("eigensolver_nev_target") and not ptree.hasKey("eigensolver_nev_threshold")) {
      spdlog::warn("Parameter 'eigensolver_nev_target' is set but no 'eigensolver_nev_threshold' is provided, using a default threshold of 0.5");
      threshold = 0.5;
      nev = ptree.get<int>("eigensolver_nev_target");
    }
    else if (ptree.hasKey("eigensolver_nev_target") and ptree.hasKey("eigensolver_nev_threshold")) {
      threshold = ptree.get<double>("eigensolver_nev_threshold");
      nev = ptree.get<int>("eigensolver_nev_target");
    }

    if (ptree.hasKey("eigensolver_nev_target") and not ptree.hasKey("eigensolver_nev_max")) {
      nev_max = 2 * nev;
      spdlog::warn("Parameter 'eigensolver_nev_target' is set but no 'eigensolver_nev_max' is provided, using a maximum number of computed eigenvectors of 2*nev = {}", nev_max);
    }
    else if (ptree.hasKey("eigensolver_nev_target") and ptree.hasKey("eigensolver_nev_max")) {
      nev_max = ptree.get<int>("eigensolver_nev_max");
    }

    bool done = threshold < 0; // In user has not provided a treshold, then we're done after one iteration

    do {
      Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, nev, 2.5 * nev, 0.00001);
      geigs.init();

      // Find largest eigenvalue of the shifted problem (which corresponds to the smallest of the original problem)
      // with max. 1000 iterations and sort the resulting eigenvalues from small to large.
      Eigen::Index nconv{};
      try {
        nconv = geigs.compute(Spectra::SortRule::LargestMagn, 1000, tolerance, Spectra::SortRule::SmallestAlge);
      }
      catch (const std::runtime_error &e) {
        spdlog::get("all_ranks")->error("ERROR: Computation of eigenvalues failed (reason: {})", e.what());
        MPI_Abort(MPI_COMM_WORLD, 5);
      }

      if (geigs.info() == Spectra::CompInfo::Successful) {
        const auto evalues = geigs.eigenvalues();
        const auto evecs = geigs.eigenvectors();

        if (evalues[nconv - 1] >= threshold or nev >= nev_max) {
          if (ptree.get("eigensolver_keep_strict", false)) {
            // If this parameter is set, we only keep those eigenvectors that correspond to eigenvalues below the threshold.
            // This is mainly useful for testing.
            std::size_t cnt = 0;
            while (cnt < nconv - 1 and evalues[cnt] < threshold) {
              ++cnt;
            }
            nconv = cnt;
          }

          if (spdlog::get("all_ranks")->level() <= spdlog::level::debug) {
            auto eigstring =
                std::accumulate(evalues.begin() + 1, evalues.begin() + nconv, to_string_with_precision(evalues[0]), [](const std::string &a, double b) { return a + ", " + std::to_string(b); });

            spdlog::get("all_ranks")->debug("Computed {} eigenvalues: {}", nconv, eigstring);
          }
          else {
            spdlog::get("all_ranks")->info("Computed {} eigenvalues: smallest {}, largest {}", nconv, evalues[0], evalues[nconv]);
          }

          if (std::any_of(evalues.begin(), evalues.end(), [](auto x) { return x < -1e-5; })) {
            spdlog::get("all_ranks")->warn("Computed a negative eigenvalue");
          }

          Vec vec(evecs.rows());
          eigenvectors.resize(nconv);
          std::fill(eigenvectors.begin(), eigenvectors.end(), vec);

          for (int i = 0; i < nconv; ++i) {
            for (int j = 0; j < evecs.rows(); ++j) {
              eigenvectors[i][j] = evecs(j, i);
            }
          }

          done = true;
        }
      }
      else {
        std::cerr << "ERROR: Eigensolver did not converge";
      }

      nev *= 1.6;
    } while (not done);
  }
  else if (eigensolver == Eigensolver::BLOPEX) {
    lobpcg_BLASLAPACKFunctions blap_fn;
    blap_fn.dsygv = dsygv_;
    blap_fn.dpotrf = dpotrf_;

    lobpcg_Tolerance lobpcg_tol;
    lobpcg_tol.absolute = tolerance;
    lobpcg_tol.relative = tolerance;

    int maxit = 10000;
    int its = 0; // Iterations that were actually required

    Vector sample(A.N());

    auto nev = ptree.get("eigensolver_nev", 10);
    std::vector<double> eigenvalues(nev);
    std::vector<double> residuals(nev);

    mv_InterfaceInterpreter ii;
    SetUpMultiVectorInterfaceInterpreter(&ii);
    // auto *x = mv_TempMultiVectorCreateFromSampleVector((void *)&ii, nev, (void *)&sample);
    // mv_TempMultiVectorSetRandom(x, 1);
    Multivec x(A.N(), nev);
    MultiVectorSetRandomValues(&x, 1);
    auto *xx = mv_MultiVectorWrap(&ii, &x, 0);

    int err = 0;
    // using Prec = void;
    // Prec *prec = nullptr;
    // if constexpr (not std::is_same_v<Prec, void>) {
    //   err = lobpcg_solve_double(xx,                        // The initial eigenvectors + pointers to interface routines
    //                             (void *)&A,                // The lhs matrix
    //                             MatMultiVec<Mat>,          // A function implementing matvec for the lhs matrix
    //                             (void *)&B,                // The rhs matrix
    //                             MatMultiVec<Mat>,          // A function implementing matvec for the rhs matrix
    //                             (void *)prec,              // The preconditioner for the lhs
    //                             applyPreconditioner<Prec>, // A function implementing the preconditioner
    //                             nullptr,                   // Input matrix Y (no idea what this is)
    //                             blap_fn,                   // The LAPACK functions that BLOPEX should use for the small gevp
    //                             lobpcg_tol,                // Tolerance that should be achieved
    //                             maxit,                     // Maximum number of iterations
    //                             0,                         // Verbosity level
    //                             &its,                      // Iterations that were required
    //                             eigenvalues.data(),        // The computed eigenvalues
    //                             nullptr,                   // History of the compute eigenvalues
    //                             0,                         // Size of the history
    //                             residuals.data(),          // residual norms
    //                             nullptr,                   // residual norms history
    //                             0                          // Size of the norms history
    //   );
    // }
    // else {
    err = lobpcg_solve_double(xx,                 // The initial eigenvectors + pointers to interface routines
                              (void *)&A,         // The lhs matrix
                              MatMultiVec<Mat>,   // A function implementing matvec for the lhs matrix
                              (void *)&B,         // The rhs matrix
                              MatMultiVec<Mat>,   // A function implementing matvec for the rhs matrix
                              nullptr,            // The preconditioner for the lhs
                              nullptr,            // A function implementing the preconditioner
                              nullptr,            // Input matrix Y (no idea what this is)
                              blap_fn,            // The LAPACK functions that BLOPEX should use for the small gevp
                              lobpcg_tol,         // Tolerance that should be achieved
                              maxit,              // Maximum number of iterations
                              1,                  // Verbosity level
                              &its,               // Iterations that were required
                              eigenvalues.data(), // The computed eigenvalues
                              nullptr,            // History of the compute eigenvalues
                              0,                  // Size of the history
                              residuals.data(),   // residual norms
                              nullptr,            // residual norms history
                              0                   // Size of the norms history
    );
    // }

    if (err != 0) {
      std::cerr << "Eigensolver failed" << std::endl;
    }

    // // Eigensolver succeded, copy computed vectors
    // std::cout << "Eigenvalues after " << its << " iterations: \n";
    // for (auto ev : eigenvalues) {
    //   std::cout << "  " << ev << "\n";
    // }
    // std::cout << std::endl;

    Vec vec(A.N());
    eigenvectors.resize(nev);
    std::fill(eigenvectors.begin(), eigenvectors.end(), vec);
    for (std::size_t i = 0; i < nev; ++i) {
      for (std::size_t j = 0; j < A.N(); ++j) {
        eigenvectors[i][j] = x.entries[i * A.N() + j];
      }
    }

    // Vec vec(A.N());
    // auto *xvec = static_cast<mv_TempMultiVector *>(x);
    // eigenvectors.resize(xvec->numVectors);
    // std::fill(eigenvectors.begin(), eigenvectors.end(), vec);
    // for (std::size_t i = 0; i < xvec->numVectors; ++i) {
    //   auto *v = static_cast<Vector *>(xvec->vector[i]);
    //   for (std::size_t j = 0; j < A.N(); ++j) {
    //     eigenvectors[i][j] = (*v)[j];
    //   }
    // }
  }
  else {
    assert(false && "Unreachable");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  // for (auto &eigenvector : eigenvectors) {
  //   eigenvector *= 1. / eigenvector.two_norm();
  // }

  return eigenvectors;
}
