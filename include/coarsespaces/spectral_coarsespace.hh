#pragma once

#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/foreach.hh>

#include <dune/istl/solver.hh>
#include <mpi.h>
#include <temp_multivector.h>
#include <vector>

#include <lobpcg.h>

#include "fakemultivec.hh"
#include "multivec.hh"
#include "spdlog/spdlog.h"

// struct WrappedVec {
//   using field_type = double;
//   using size_type = std::size_t;
//   using block_type = double;

//   Dune::FieldVector<double, 1> *data;
//   std::size_t N;

//   void axpy(double w, const WrappedVec &v)
//   {
//     for (std::size_t i = 0; i < N; ++i) {
//       data[i] += w * v.data[i];
//     }
//   }

//   void operator*=(double a)
//   {
//     for (std::size_t i = 0; i < N; ++i) {
//       data[i] *= a;
//     }
//   }
//   double &operator[](std::size_t i) { return data[i]; }
//   const double &operator[](std::size_t i) const { return data[i]; }
// };

extern "C" { // Tell the linker that LAPACK functions are mangled as C functions
BlopexInt dsygv_(BlopexInt *itype, char *jobz, char *uplo, BlopexInt *n, double *a, BlopexInt *lda, double *b, BlopexInt *ldb, double *w, double *work, BlopexInt *lwork, BlopexInt *info);
BlopexInt dpotrf_(char *uplo, BlopexInt *n, double *a, BlopexInt *lda, BlopexInt *info);
}

enum class Eigensolver { Spectra, BLOPEX };

template <class Prec>
void applyPreconditionerFake(void *T_, void *r_, void *Tr_)
{
  auto *T = static_cast<Prec *>(T_);
  auto *r = static_cast<mv_TempMultiVector *>(r_);
  auto *Tr = static_cast<mv_TempMultiVector *>(Tr_);

  for (std::size_t i = 0; i < r->numVectors; ++i) {
    if (r->mask && r->mask[i] == 0) {
      continue;
    }

    auto *v = static_cast<Vector *>(r->vector[i]);
    auto *res = static_cast<Vector *>(Tr->vector[i]);

    Dune::InverseOperatorResult ior;
    T->apply(*res, *v, ior);
  }
}

template <class Prec>
void applyPreconditioner(void *T_, void *r_, void *Tr_)
{
  auto *T = static_cast<Prec *>(T_);
  auto *r = static_cast<Multivec *>(r_);
  auto *Tr = static_cast<Multivec *>(Tr_);

  Dune::BlockVector<Dune::FieldVector<double, 1>> v(r->N);
  Dune::BlockVector<Dune::FieldVector<double, 1>> d(r->N);

  Dune::InverseOperatorResult ior;
  for (std::size_t i = 0; i < r->active_indices.size(); ++i) {
    auto *rstart = r->entries.data() + r->active_indices[i] * r->N;
    auto *Trstart = Tr->entries.data() + Tr->active_indices[i] * Tr->N;

    std::copy(rstart, rstart + r->N, d.begin());
    v = 0;

    Dune::InverseOperatorResult res;
    T->apply(v, d, res);

    std::copy(v.begin(), v.end(), Trstart);
  }
}

template <class Vec, class Mat, class Eigensolver, class Prec = void>
std::vector<Vec> solveGEVP(const Mat &A, const Mat &B, Eigensolver eigensolver, long nev, const Dune::ParameterTree &ptree, Prec *prec = nullptr)
{
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
      if (std::abs(entry) > 1e-16) {
        triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry});
      }
    });
    A_.setFromTriplets(triplets.begin(), triplets.end());

    triplets.clear();
    triplets.reserve(A.nonzeroes());
    Dune::flatMatrixForEach(B, [&](auto &&entry, std::size_t i, std::size_t j) {
      if (std::abs(entry) > 1e-16) {
        triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry});
      }
    });
    B_.setFromTriplets(triplets.begin(), triplets.end());

    A_.makeCompressed();
    B_.makeCompressed();

    spdlog::info("nnz(A) = {}, nnz(B) = {}", A_.nonZeros(), B_.nonZeros());

    using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
    using BOpType = Spectra::SparseSymMatProd<double>;
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    OpType op(A_, B_);
    BOpType Bop(B_);

    Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, nev, 3 * nev, 0.001);
    geigs.init();

    // Find largest eigenvalue of the shifted problem (which corresponds to the smallest of the original problem)
    // with max. 1000 iterations and sort the resulting eigenvalues from small to large.
    auto nconv = geigs.compute(Spectra::SortRule::LargestMagn, 1000, tolerance, Spectra::SortRule::SmallestAlge);

    if (geigs.info() == Spectra::CompInfo::Successful) {
      const auto evalues = geigs.eigenvalues();
      const auto evecs = geigs.eigenvectors();

      spdlog::get("all_ranks")->info("Computed eigenvalues, smallest {}, largest {}", evalues[0], evalues[evalues.size() - 1]);

      Vec vec(evecs.rows());
      eigenvectors.resize(nconv);
      std::fill(eigenvectors.begin(), eigenvectors.end(), vec);

      for (int i = 0; i < nconv; ++i) {
        for (int j = 0; j < evecs.rows(); ++j) {
          eigenvectors[i][j] = evecs(j, i);
        }
      }
    }
    else {
      std::cerr << "ERROR: Eigensolver did not converge";
    }
  }
  else if (eigensolver == Eigensolver::BLOPEX) {
    lobpcg_BLASLAPACKFunctions blap_fn;
    blap_fn.dsygv = dsygv_;
    blap_fn.dpotrf = dpotrf_;

    lobpcg_Tolerance lobpcg_tol;
    lobpcg_tol.absolute = tolerance;
    lobpcg_tol.relative = tolerance;

    int maxit = 100;
    int its = 0; // Iterations that were actually required

    Vector sample(A.N());

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
    if constexpr (not std::is_same_v<Prec, void>) {
      err = lobpcg_solve_double(xx,                        // The initial eigenvectors + pointers to interface routines
                                (void *)&A,                // The lhs matrix
                                MatMultiVec<Mat>,          // A function implementing matvec for the lhs matrix
                                (void *)&B,                // The rhs matrix
                                MatMultiVec<Mat>,          // A function implementing matvec for the rhs matrix
                                (void *)prec,              // The preconditioner for the lhs
                                applyPreconditioner<Prec>, // A function implementing the preconditioner
                                nullptr,                   // Input matrix Y (no idea what this is)
                                blap_fn,                   // The LAPACK functions that BLOPEX should use for the small gevp
                                lobpcg_tol,                // Tolerance that should be achieved
                                maxit,                     // Maximum number of iterations
                                0,                         // Verbosity level
                                &its,                      // Iterations that were required
                                eigenvalues.data(),        // The computed eigenvalues
                                nullptr,                   // History of the compute eigenvalues
                                0,                         // Size of the history
                                residuals.data(),          // residual norms
                                nullptr,                   // residual norms history
                                0                          // Size of the norms history
      );
    }
    else {
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

  return eigenvectors;
}
