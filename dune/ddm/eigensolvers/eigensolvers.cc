#include <dune-istl-config.hh>

#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>
#include <dune/istl/cholmod.hh>

#include <mpi.h>

#include <dune/common/parametertree.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/foreach.hh>

#include <algorithm>
#include <umfpack.h>
#include <vector>

#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/Util/CompInfo.h>

#include "eigensolvers.hh"
#include "../logger.hh"
#include "../strumpack.hh"

#include <spdlog/spdlog.h>

#if DUNE_DDM_HAVE_BLOPEX
#include <lobpcg.h>

#include "coarsespaces/fakemultivec.hh"
#include "coarsespaces/multivec.hh"

extern "C" { // Tell the linker that LAPACK functions are mangled as C functions
BlopexInt dsygv_(BlopexInt *itype, char *jobz, char *uplo, BlopexInt *n, double *a, BlopexInt *lda, double *b, BlopexInt *ldb, double *w, double *work, BlopexInt *lwork, BlopexInt *info);
BlopexInt dpotrf_(char *uplo, BlopexInt *n, double *a, BlopexInt *lda, BlopexInt *info);
}

void applyPreconditioner(void *T_, void *r_, void *Tr_)
{
  auto *T = static_cast<Dune::Preconditioner<Dune::BlockVector<Dune::FieldVector<double, 1>>, Dune::BlockVector<Dune::FieldVector<double, 1>>> *>(T_);
  auto *r = static_cast<Multivec *>(r_);
  auto *Tr = static_cast<Multivec *>(Tr_);

  Dune::BlockVector<Dune::FieldVector<double, 1>> v(r->N);
  Dune::BlockVector<Dune::FieldVector<double, 1>> d(r->N);

  for (std::size_t i = 0; i < r->active_indices.size(); ++i) {
    auto *rstart = r->entries.data() + r->active_indices[i] * r->N;
    auto *Trstart = Tr->entries.data() + Tr->active_indices[i] * Tr->N;

    std::copy(rstart, rstart + r->N, d.begin());
    v = 0;

    T->apply(v, d);

    std::copy(v.begin(), v.end(), Trstart);
  }
}
#endif

class SymShiftInvert {
public:
  using Scalar = double;

  using Solver = Dune::UMFPack<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>>;

  SymShiftInvert(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A, const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &B) : A(A), B(B), b(A.N())
  {
    solve_event = Logger::get().registerOrGetEvent("Eigensolver", "solve A-sB");
  }

  void set_shift(double sigma)
  {
    if (!solver or sigma != last_sigma) {
      Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Eigensolver", "factorise A-sB")};

      auto A_minus_sigma_B = A;
      A_minus_sigma_B.axpy(-sigma, B);

#ifndef DUNE_DDM_HAVE_STRUMPACK
      solver = std::make_unique<Solver>();
      // solver->setOption(UMFPACK_STRATEGY, UMFPACK_STRATEGY_SYMMETRIC);
      solver->setOption(UMFPACK_ORDERING, UMFPACK_ORDERING_METIS);
      solver->setOption(UMFPACK_IRSTEP, 0);
      solver->setMatrix(A_minus_sigma_B);
#else
      solver = std::make_unique<Dune::STRUMPACK<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>>>(A_minus_sigma_B);
#endif

      last_sigma = sigma;
    }
  }

  void perform_op(const Scalar *x_in, Scalar *y_out) const
  {
    Logger::ScopedLog sl{solve_event};
    std::copy_n(x_in, b.size(), b.begin());
    solver->apply(y_out, b.data());
  }

  Eigen::Index rows() const { return static_cast<Eigen::Index>(A.N()); }
  Eigen::Index cols() const { return static_cast<Eigen::Index>(A.M()); }

private:
  const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A;
  const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &B;

  mutable std::vector<double> b;

#ifndef DUNE_DDM_HAVE_STRUMPACK
  std::unique_ptr<Solver> solver{nullptr};
#else
  std::unique_ptr<Dune::STRUMPACK<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>>> solver{nullptr};
#endif

  Logger::Event *solve_event;

  double last_sigma;
};

class MatOp {
public:
  using Scalar = double;

  explicit MatOp(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A) : A(A) {}

  void perform_op(const Scalar *x_in, Scalar *y_out) const
  {
    std::fill_n(y_out, A.N(), 0);
    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        y_out[ri.index()] += *ci * x_in[ci.index()];
      }
    }
  }

private:
  const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A;
};

std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> solveGEVP(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A, const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &B,
                                                                       Eigensolver eigensolver, const Dune::ParameterTree &ptree,
                                                                       Dune::Preconditioner<Dune::BlockVector<Dune::FieldVector<double, 1>>, Dune::BlockVector<Dune::FieldVector<double, 1>>> *prec)
{
  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;
  std::vector<Vec> eigenvectors;

  const double tolerance = ptree.get("eigensolver_tolerance", 1e-5);

  if (eigensolver == Eigensolver::Spectra) {
    spdlog::get("all_ranks")->debug("Solving eigenproblem Ax=lBx of size {} with nnz(A) = {}, nnz(B) = {}", A.N(), A.nonzeroes(), B.nonzeroes());

#if 1
    using OpType = SymShiftInvert;
    using BOpType = MatOp;

    OpType op(A, B);
    BOpType Bop(B);

#else
    Eigen::SparseMatrix<double> A_(A.N(), A.M());
    Eigen::SparseMatrix<double> B_(B.N(), B.M());

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;
    triplets.reserve(A.N() * 5);
    Dune::flatMatrixForEach(A, [&](auto &&entry, std::size_t i, std::size_t j) { triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry}); });
    A_.setFromTriplets(triplets.begin(), triplets.end());

    triplets.clear();
    Dune::flatMatrixForEach(B, [&](auto &&entry, std::size_t i, std::size_t j) { triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry}); });
    B_.setFromTriplets(triplets.begin(), triplets.end());

    A_.makeCompressed();
    B_.makeCompressed();

    using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = Spectra::SparseSymMatProd<double>;

    OpType op(A_, B_);
    BOpType Bop(B_);
#endif

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

    auto ncv = 2 * nev;
    int tries = 3;
    int it = 0;
    do {
      if (ncv <= nev) {
        ncv = 2 * nev;
      }

      spdlog::get("all_ranks")->trace("Computing eigenvalues using Spectra, iteration {}", it++);
      Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, nev, ncv, ptree.get("eigensolver_shift", 0.0001));
      geigs.init();

      // Find largest eigenvalue of the shifted problem (which corresponds to the smallest of the original problem)
      // with max. 1000 iterations and sort the resulting eigenvalues from small to large.
      Eigen::Index nconv{};
      try {
        const auto maxit = 100;
        nconv = geigs.compute(Spectra::SortRule::LargestMagn, maxit, tolerance, Spectra::SortRule::SmallestAlge);
      }
      catch (const std::runtime_error &e) {
        spdlog::get("all_ranks")->error("ERROR: Computation of eigenvalues failed, reason: {}, tries left {}", e.what(), tries);
        if (tries != 0) {
          tries--;
          ncv *= 2;
          done = false;
          continue;
        }
        else {
          MPI_Abort(MPI_COMM_WORLD, 5);
        }
      }

      if (geigs.info() == Spectra::CompInfo::Successful) {
        const auto evalues = geigs.eigenvalues();
        const auto evecs = geigs.eigenvectors();

        if (evalues[nconv - 1] >= threshold or nev >= nev_max) {
          if (threshold > 0 and ptree.get("eigensolver_keep_strict", false)) {
            // If this parameter is set, we only keep those eigenvectors that correspond to eigenvalues below the threshold.
            long cnt = 0;
            while (cnt < nconv - 1 and evalues[cnt] < threshold) {
              ++cnt;
            }
            nconv = std::max(cnt, 1L); // Make sure that nconv is at least 1 (might be zero if the first eigenvalue is already larger than the threshold)
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

          break;
        }
        if (!done) {
          nev *= 2;
          spdlog::get("all_ranks")->debug("Eigensolver did not compute enough eigenvalues, largest is {}, now trying with nev = {}", evalues[nconv - 1], nev);
        }
      }
      else {
        if (geigs.info() == Spectra::CompInfo::NotConverging) {
          if (tries != 0) {
            spdlog::get("all_ranks")->warn("Computation of eigenvalues failed, not yet converged, trying again with more Lanzcos vectors. Tries left {}", tries);
            tries--;
            ncv *= 2;
            done = false;
            continue;
          }
          else {
            spdlog::get("all_ranks")->warn("Computation of eigenvalues failed, not yet converged, no more tries left");
            MPI_Abort(MPI_COMM_WORLD, 12);
          }
        }
        else if (geigs.info() == Spectra::CompInfo::NumericalIssue) {
          spdlog::get("all_ranks")->error("Computation of eigenvalues failed, reason 'NumericalIssue'");
          MPI_Abort(MPI_COMM_WORLD, 13);
        }
        else {
          spdlog::get("all_ranks")->error("Computation of eigenvalues failed, unknown reason");
          MPI_Abort(MPI_COMM_WORLD, 14);
        }
      }
    } while (not done);
  }
#if DUNE_DDM_HAVE_BLOPEX
  else if (eigensolver == Eigensolver::BLOPEX) {
    lobpcg_BLASLAPACKFunctions blap_fn;
    blap_fn.dsygv = dsygv_;
    blap_fn.dpotrf = dpotrf_;

    lobpcg_Tolerance lobpcg_tol;
    lobpcg_tol.absolute = tolerance;
    lobpcg_tol.relative = tolerance;

    int maxit = ptree.get("eigensolver_maxit", 1000);
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

    using Mat = Dune::BCRSMatrix<double>;
    int err = 0;
    if (prec != nullptr) {
      err = lobpcg_solve_double(xx,                                    // The initial eigenvectors + pointers to interface routines
                                (void *)&A,                            // The lhs matrix
                                MatMultiVec<Mat>,                      // A function implementing matvec for the lhs matrix
                                (void *)&B,                            // The rhs matrix
                                MatMultiVec<Mat>,                      // A function implementing matvec for the rhs matrix
                                (void *)prec,                          // The preconditioner for the lhs
                                applyPreconditioner,                   // A function implementing the preconditioner
                                nullptr,                               // Input matrix Y (no idea what this is)
                                blap_fn,                               // The LAPACK functions that BLOPEX should use for the small gevp
                                lobpcg_tol,                            // Tolerance that should be achieved
                                maxit,                                 // Maximum number of iterations
                                ptree.get("eigensolver_verbosity", 0), // Verbosity level
                                &its,                                  // Iterations that were required
                                eigenvalues.data(),                    // The computed eigenvalues
                                nullptr,                               // History of the compute eigenvalues
                                0,                                     // Size of the history
                                residuals.data(),                      // residual norms
                                nullptr,                               // residual norms history
                                0                                      // Size of the norms history
      );
    }
    else {
      err = lobpcg_solve_double(xx,                                    // The initial eigenvectors + pointers to interface routines
                                (void *)&A,                            // The lhs matrix
                                MatMultiVec<Mat>,                      // A function implementing matvec for the lhs matrix
                                (void *)&B,                            // The rhs matrix
                                MatMultiVec<Mat>,                      // A function implementing matvec for the rhs matrix
                                nullptr,                               // The preconditioner for the lhs
                                nullptr,                               // A function implementing the preconditioner
                                nullptr,                               // Input matrix Y (no idea what this is)
                                blap_fn,                               // The LAPACK functions that BLOPEX should use for the small gevp
                                lobpcg_tol,                            // Tolerance that should be achieved
                                maxit,                                 // Maximum number of iterations
                                ptree.get("eigensolver_verbosity", 0), // Verbosity level
                                &its,                                  // Iterations that were required
                                eigenvalues.data(),                    // The computed eigenvalues
                                nullptr,                               // History of the compute eigenvalues
                                0,                                     // Size of the history
                                residuals.data(),                      // residual norms
                                nullptr,                               // residual norms history
                                0                                      // Size of the norms history
      );
    }

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
    for (int i = 0; i < nev; ++i) {
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
#endif
  else {
    DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  return eigenvectors;
}
