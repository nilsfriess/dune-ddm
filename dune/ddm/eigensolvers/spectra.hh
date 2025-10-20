#pragma once

#include "../logger.hh"
#include "../strumpack.hh"
#include "eigensolver_params.hh"

#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/Util/CompInfo.h>
#include <algorithm>
#include <dune/common/parametertree.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/cholmod.hh>
#include <dune/istl/foreach.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>
#include <mpi.h>
#include <umfpack.h>
#include <vector>

// Helper to convert floats to string because std::to_string outputs with low precision
template <typename T>
std::string to_string_with_precision(const T a_value)
{
  std::ostringstream out;
  out << a_value;
  return std::move(out).str();
}

class SymShiftInvert {
public:
  using Scalar = double;

  using Solver = Dune::UMFPack<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>>;

  SymShiftInvert(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>& A, const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>& B)
      : A(A)
      , B(B)
      , b(A.N())
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
      // solver->setOption(UMFPACK_IRSTEP, 0);
      solver->setMatrix(A_minus_sigma_B);
#else
      solver = std::make_unique<Dune::STRUMPACK<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>>>(A_minus_sigma_B);
#endif

      last_sigma = sigma;
    }
  }

  void perform_op(const Scalar* x_in, Scalar* y_out) const
  {
    Logger::ScopedLog sl{solve_event};
    std::copy_n(x_in, b.size(), b.begin());
    solver->apply(y_out, b.data());
  }

  Eigen::Index rows() const { return static_cast<Eigen::Index>(A.N()); }
  Eigen::Index cols() const { return static_cast<Eigen::Index>(A.M()); }

private:
  const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>& A;
  const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>& B;

  mutable std::vector<double> b;

#ifndef DUNE_DDM_HAVE_STRUMPACK
  std::unique_ptr<Solver> solver{nullptr};
#else
  std::unique_ptr<Dune::STRUMPACK<Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>>> solver{nullptr};
#endif

  Logger::Event* solve_event;

  double last_sigma;
};

class MatOp {
public:
  using Scalar = double;

  explicit MatOp(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>& A)
      : A(A)
  {
  }

  void perform_op(const Scalar* x_in, Scalar* y_out) const
  {
    std::fill_n(y_out, A.N(), 0);
    for (auto ri = A.begin(); ri != A.end(); ++ri)
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) y_out[ri.index()] += *ci * x_in[ci.index()];
  }

private:
  const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>>& A;
};

template <class Mat>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> spectra_gevp(const Mat& A, const Mat& B, const EigensolverParams& params)
{
  logger::debug_all("Solving eigenproblem Ax=lBx of size {} with nnz(A) = {}, nnz(B) = {}", A.N(), A.nonzeroes(), B.nonzeroes());

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
  Dune::flatMatrixForEach(A, [&](auto&& entry, std::size_t i, std::size_t j) { triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry}); });
  A_.setFromTriplets(triplets.begin(), triplets.end());

  triplets.clear();
  Dune::flatMatrixForEach(B, [&](auto&& entry, std::size_t i, std::size_t j) { triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry}); });
  B_.setFromTriplets(triplets.begin(), triplets.end());

  A_.makeCompressed();
  B_.makeCompressed();

  using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
  using BOpType = Spectra::SparseSymMatProd<double>;

  OpType op(A_, B_);
  BOpType Bop(B_);
#endif

  auto nev = params.nev;
  auto nev_max = 2 * params.nev; // FIXME
  auto shift = params.shift;
  auto tolerance = params.tolerance;
  double threshold = -1;     // FIXME
  bool done = threshold < 0; // In user has not provided a treshold, then we're done after one iteration

  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;
  std::vector<Vec> eigenvectors;
  auto ncv = params.ncv;
  int tries = 3;
  int it = 0;
  do {
    if (ncv <= nev) ncv = 2 * nev;

    logger::trace_all("Computing eigenvalues using Spectra, iteration {}", it++);
    Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, nev, ncv, shift);
    geigs.init();

    // Find largest eigenvalue of the shifted problem (which corresponds to the smallest of the original problem)
    // with max. 1000 iterations and sort the resulting eigenvalues from small to large.
    Eigen::Index nconv{};
    try {
      const auto maxit = 100;
      nconv = geigs.compute(Spectra::SortRule::LargestMagn, maxit, tolerance, Spectra::SortRule::SmallestAlge);
    }
    catch (const std::runtime_error& e) {
      logger::error_all("ERROR: Computation of eigenvalues failed, reason: {}, tries left {}", e.what(), tries);
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
        if (threshold > 0) {
          // If this parameter is set, we only keep those eigenvectors that correspond to eigenvalues below the threshold.
          long cnt = 0;
          while (cnt < nconv - 1 and evalues[cnt] < threshold) ++cnt;
          nconv = std::max(cnt, 1L); // Make sure that nconv is at least 1 (might be zero if the first eigenvalue is already larger than the threshold)
        }

        if (logger::get_level() <= logger::Level::debug) {
          auto eigstring =
              std::accumulate(evalues.begin() + 1, evalues.begin() + nconv, to_string_with_precision(evalues[0]), [](const std::string& a, double b) { return a + ", " + std::to_string(b); });

          logger::info_all("Computed {} eigenvalues: {}", nconv, eigstring);
        }
        else {
          logger::info_all("Computed {} eigenvalues: smallest {}, largest {}", nconv, evalues[0], evalues[nconv - 1]);
        }

        if (std::any_of(evalues.begin(), evalues.end(), [](auto x) { return x < -1e-5; })) logger::warn_all("Computed a negative eigenvalue");

        Vec vec(evecs.rows());
        eigenvectors.resize(nconv);
        std::fill(eigenvectors.begin(), eigenvectors.end(), vec);

        for (int i = 0; i < nconv; ++i)
          for (int j = 0; j < evecs.rows(); ++j) eigenvectors[i][j] = evecs(j, i);

        break;
      }
      if (!done) {
        nev *= 2;
        logger::debug_all("Eigensolver did not compute enough eigenvalues, largest is {}, now trying with nev = {}", evalues[nconv - 1], nev);
      }
    }
    else if (geigs.info() == Spectra::CompInfo::NotConverging) {
      if (tries != 0) {
        logger::warn_all("Computation of eigenvalues failed, not yet converged, trying again with more Lanzcos vectors. Tries left {}", tries);
        tries--;
        ncv *= 2;
        done = false;
        continue;
      }
      else {
        logger::warn_all("Computation of eigenvalues failed, not yet converged, no more tries left");
        MPI_Abort(MPI_COMM_WORLD, 12);
      }
    }
    else if (geigs.info() == Spectra::CompInfo::NumericalIssue) {
      logger::error_all("Computation of eigenvalues failed, reason 'NumericalIssue'");
      MPI_Abort(MPI_COMM_WORLD, 13);
    }
    else {
      logger::error_all("Computation of eigenvalues failed, unknown reason");
      MPI_Abort(MPI_COMM_WORLD, 14);
    }
  } while (not done);

  return eigenvectors;
}
