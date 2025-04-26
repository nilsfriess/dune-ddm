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

class SymShiftInvert {
public:
  using Scalar = double;

  SymShiftInvert(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A, const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &B) : A(A), B(B), b(A.N()) {}

  void set_shift(double sigma)
  {
    auto A_minus_sigma_B = A;
    // TODO: This assumes that the sparsity pattern of B is a subset of that of A
    for (auto ri = B.begin(); ri != B.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        A_minus_sigma_B[ri.index()][ci.index()] -= sigma * B[ri.index()][ci.index()];
      }
    }

    solver = std::make_unique<Dune::UMFPack<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>>>(A_minus_sigma_B, 0);
    solver->setOption(UMFPACK_IRSTEP, 0);
  }

  void perform_op(const Scalar *x_in, Scalar *y_out) const
  {
    auto *x_in_mut = const_cast<Scalar *>(x_in); // TODO: Here we just hope that either (i) the solver doesn't modify the rhs or (ii) it's not a problem if it does.
    solver->apply(y_out, x_in_mut);
  }

  Eigen::Index rows() const { return static_cast<Eigen::Index>(A.N()); }
  Eigen::Index cols() const { return static_cast<Eigen::Index>(A.M()); }

private:
  const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A;
  const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &B;

  mutable std::vector<double> b;

  std::unique_ptr<Dune::UMFPack<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>>> solver{nullptr};
};

class MatOp {
public:
  using Scalar = double;

  explicit MatOp(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A) : A(A), x(A.N()), y(A.M()) {}

  void perform_op(const Scalar *x_in, Scalar *y_out) const
  {
    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      y_out[ri.index()] = 0;
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        y_out[ri.index()] += *ci * x_in[ci.index()];
      }
    }
  }

private:
  const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A;

  mutable Dune::BlockVector<Dune::FieldVector<double, 1>> x;
  mutable Dune::BlockVector<Dune::FieldVector<double, 1>> y;
};
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> solveGEVP(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A, const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &B,
                                                                       Eigensolver eigensolver, const Dune::ParameterTree &ptree)
{
  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;
  using Mat = Dune::BCRSMatrix<double>;
  std::vector<Vec> eigenvectors;

  const double tolerance = ptree.get("eigensolver_tolerance", 1e-5);

  if (eigensolver == Eigensolver::Spectra) {
    spdlog::get("all_ranks")->debug("Solving eigenproblem Ax=lBx of size {} with nnz(A) = {}, nnz(B) = {}", A.N(), A.nonzeroes(), B.nonzeroes());

    using OpType = SymShiftInvert;
    using BOpType = MatOp;

    OpType op(A, B);
    BOpType Bop(B);

    auto nev = ptree.get("eigensolver_nev", 10);
    auto nev_max = nev;
    double threshold = -1;
    if (ptree.hasKey("eigensolver_nev_target") and not ptree.hasKey("eigensolver_threshold")) {
      spdlog::warn("Parameter 'eigensolver_nev_target' is set but no 'eigensolver_threshold' is provided, using a default threshold of 0.5");
      threshold = 0.5;
      nev = ptree.get<int>("eigensolver_nev_target");
    }
    else if (ptree.hasKey("eigensolver_nev_target") and ptree.hasKey("eigensolver_threshold")) {
      threshold = ptree.get<double>("eigensolver_threshold");
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

      spdlog::get("all_ranks")->trace("Eigensolver computed {} eigenvalues", nconv);

      if (geigs.info() == Spectra::CompInfo::Successful) {
        const auto evalues = geigs.eigenvalues();
        const auto evecs = geigs.eigenvectors();

        if (evalues[nconv - 1] >= threshold or nev >= nev_max) {
          if (threshold > 0 and ptree.get("eigensolver_keep_strict", false)) {
            // If this parameter is set, we only keep those eigenvectors that correspond to eigenvalues below the threshold.
            // This is mainly useful for testing.
            long cnt = 0;
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
            spdlog::get("all_ranks")->info("Computed {} eigenvalues: smallest {}, largest {}", nconv, evalues[0], evalues[nconv - 1]);
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
