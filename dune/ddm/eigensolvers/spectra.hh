#pragma once

#ifdef DUNE_DDM_HAVE_SPECTRA

#include "../logger.hh"
#include "eigensolver_params.hh"
#include "eigensolver_result.hh"
#include "helpers.hh"

#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/Util/CompInfo.h>
#include <algorithm>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/cholmod.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>
#include <memory>
#include <umfpack.h>
#include <vector>

namespace ddm {

namespace spectra {
// Shift-invert operator (A - sigma B)^{-1} used as the OpType of Spectra's
// generalised symmetric shift-invert eigensolver. The factorisation is computed
// lazily on the first call to set_shift() and reused for repeated shifts.
template <class Scalar_ = double>
class SymShiftInvert {
public:
  using Scalar = Scalar_;
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;
  using DefaultSolver = Dune::UMFPack<Matrix>;

  SymShiftInvert(std::shared_ptr<const Matrix> A, std::shared_ptr<const Matrix> B, Dune::ParameterTree solver_subtree)
      : A(std::move(A))
      , B(std::move(B))
      , solver_subtree(std::move(solver_subtree))
  {
    solve_event = Logger::get().registerOrGetEvent("Eigensolver", "solve A-sB");
  }

  void set_shift(Scalar sigma)
  {
    if (!solver or sigma != last_sigma) {
      Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Eigensolver", "factorise A-sB")};

      A_shifted = std::make_shared<Matrix>(*A);
      A_shifted->axpy(-sigma, *B);

      if (solver_subtree.get("type", std::string{"umfpack"}) == "umfpack") {
        auto umfpack = std::make_shared<DefaultSolver>();
        umfpack->setOption(UMFPACK_ORDERING, UMFPACK_ORDERING_METIS);
        umfpack->setMatrix(*A_shifted);
        solver = umfpack;
      }
      else {
        using Op = Dune::MatrixAdapter<Matrix, Vector, Vector>;
        Dune::initSolverFactories<Op>();
        auto op = std::make_shared<Op>(std::const_pointer_cast<const Matrix>(A_shifted));
        solver = Dune::getSolverFromFactory(op, solver_subtree);
      }

      last_sigma = sigma;
    }
  }

  void perform_op(const Scalar* x_in, Scalar* y_out) const
  {
    Logger::ScopedLog sl{solve_event};

    Vector y(A->N());
    Vector b(A->N());
    std::copy_n(x_in, b.size(), b.begin());

    Dune::InverseOperatorResult res;
    solver->apply(y, b, res);

    std::copy_n(y.begin(), y.size(), y_out);
  }

  Eigen::Index rows() const { return static_cast<Eigen::Index>(A->N()); }
  Eigen::Index cols() const { return static_cast<Eigen::Index>(A->M()); }

private:
  std::shared_ptr<const Matrix> A;
  std::shared_ptr<const Matrix> B;
  std::shared_ptr<Matrix> A_shifted;

  Dune::ParameterTree solver_subtree;
  std::shared_ptr<Dune::InverseOperator<Vector, Vector>> solver{nullptr};

  Logger::Event* solve_event;

  Scalar last_sigma{};
};

// Plain sparse mat-vec y = A x used as the BOpType of Spectra's generalised solver.
template <class Scalar_ = double>
class MatOp {
public:
  using Scalar = Scalar_;
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1>>;

  explicit MatOp(std::shared_ptr<const Matrix> A)
      : A(std::move(A))
  {
  }

  void perform_op(const Scalar* x_in, Scalar* y_out) const
  {
    std::fill_n(y_out, A->N(), 0);
    for (auto ri = A->begin(); ri != A->end(); ++ri)
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) y_out[ri.index()] += *ci * x_in[ci.index()];
  }

private:
  std::shared_ptr<const Matrix> A;
};
} // namespace spectra

// Generalised symmetric eigensolver A x = lambda B x based on Spectra's shift-invert
// Lanczos. Computes the smallest eigenpairs; supports adaptive growth of the number
// of computed eigenvalues until the largest exceeds a threshold (see EigensolverParams).
template <class Scalar = double>
class SpectraEigensolver {
public:
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;

  // The solver shares ownership of A and B (the internal operators keep them alive),
  // so the matrices need not be kept alive by the caller.
  SpectraEigensolver(std::shared_ptr<const Matrix> A, std::shared_ptr<const Matrix> B, EigensolverParams params, const Dune::ParameterTree& ptree)
      : op_(A, B, ptree.sub("solver"))
      , Bop_(B)
      , params_(params)
  {
    logger::debug_all("Setting up eigenproblem Ax=lBx of size {} with nnz(A) = {}, nnz(B) = {}", A->N(), A->nonzeroes(), B->nonzeroes());
  }

  EigensolverResult solve()
  {
    eigenvalues_.clear();
    eigenvectors_.clear();

    auto nev = params_.nev;
    const auto nev_max = params_.nev_max;
    const auto shift = params_.shift;
    const auto tolerance = params_.tolerance;
    const double threshold = params_.threshold;
    bool done = threshold <= 0; // If threshold is not positive, no adaptive nev doubling — just compute once

    auto ncv = params_.ncv;
    int tries = 3;
    int it = 0;

    unsigned int iterations = 0;
    unsigned int operator_applications = 0;

    do {
      if (ncv <= nev) ncv = 2 * nev;

      logger::trace_all("Computing eigenvalues using Spectra, iteration {}", it++);
      Spectra::SymGEigsShiftSolver<spectra::SymShiftInvert<Scalar>, spectra::MatOp<Scalar>, Spectra::GEigsMode::ShiftInvert> geigs(op_, Bop_, nev, ncv, shift);
      geigs.init();

      // Find largest eigenvalue of the shifted problem (which corresponds to the smallest of the original problem)
      // with max. 100 iterations and sort the resulting eigenvalues from small to large.
      Eigen::Index nconv{};
      try {
        const auto maxit = params_.maxit;
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
          logger::error_all("Computation of eigenvalues failed, no more tries left. Returning empty basis.");
          return {.converged = false, .iterations = iterations, .operator_applications = operator_applications};
        }
      }

      iterations += geigs.num_iterations();
      operator_applications += geigs.num_operations();

      if (geigs.info() == Spectra::CompInfo::Successful) {
        const auto evalues = geigs.eigenvalues();
        const auto evecs = geigs.eigenvectors();

        // If threshold is positive, we adaptively compute more eigenvalues until the largest
        // exceeds the threshold (or nev_max is reached). If threshold <= 0, it is ignored
        // and we simply take all nev computed eigenvalues.
        bool have_enough = (threshold <= 0) or (evalues[nconv - 1] >= threshold) or (nev >= nev_max);
        if (have_enough) {
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

          eigenvalues_.resize(nconv);
          for (Eigen::Index i = 0; i < nconv; ++i) eigenvalues_[i] = static_cast<Scalar>(evalues[i]);

          Vector vec(evecs.rows());
          eigenvectors_.assign(nconv, vec);
          for (Eigen::Index i = 0; i < nconv; ++i)
            for (Eigen::Index j = 0; j < evecs.rows(); ++j) eigenvectors_[i][j] = static_cast<Scalar>(evecs(j, i));

          return {.converged = true, .iterations = iterations, .operator_applications = operator_applications};
        }
        if (!done) {
          nev *= 2;
          logger::debug_all("Eigensolver did not compute enough eigenvalues (largest: {}, threshold: {}), now trying with nev = {}", evalues[nconv - 1], threshold, nev);
        }
        else {
          logger::warn_all("Eigensolver computed {} eigenvalues but the largest ({}) did not reach the threshold ({}), "
                           "and nev ({}) reached nev_max ({}). Returning empty basis. Consider adjusting 'shift', 'threshold', or 'nev_max'.",
                           nconv, evalues[nconv - 1], threshold, nev, nev_max);
          return {.converged = false, .iterations = iterations, .operator_applications = operator_applications};
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
          logger::warn_all("Computation of eigenvalues failed, not yet converged, no more tries left. Returning empty basis.");
          return {.converged = false, .iterations = iterations, .operator_applications = operator_applications};
        }
      }
      else if (geigs.info() == Spectra::CompInfo::NumericalIssue) {
        logger::error_all("Computation of eigenvalues failed, reason 'NumericalIssue'. Returning empty basis.");
        return {.converged = false, .iterations = iterations, .operator_applications = operator_applications};
      }
      else {
        logger::error_all("Computation of eigenvalues failed, unknown reason. Returning empty basis.");
        return {.converged = false, .iterations = iterations, .operator_applications = operator_applications};
      }
    } while (not done);

    // Only reached when threshold <= 0 and no branch above returned; treat as non-converged.
    logger::warn_all("Eigensolver returned an empty set of eigenvectors. The coarse space will not be set up.");
    return {.converged = false, .iterations = iterations, .operator_applications = operator_applications};
  }

  const std::vector<Vector>& eigenvectors() const { return eigenvectors_; }
  const std::vector<Scalar>& eigenvalues() const { return eigenvalues_; }

private:
  spectra::SymShiftInvert<Scalar> op_;
  spectra::MatOp<Scalar> Bop_;
  EigensolverParams params_;

  std::vector<Vector> eigenvectors_;
  std::vector<Scalar> eigenvalues_;
};
} // namespace ddm

#endif
