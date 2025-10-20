#pragma once

#include "../helpers.hh"
#include "blockmatrix.hh"
#include "concepts.hh"
#include "dune/ddm/eigensolvers/concepts.hh"
#include "dune/ddm/eigensolvers/orthogonalisation.hh"
#include "dune/ddm/logger.hh"
#include "eigensolver_params.hh"
#include "lapacke.hh"
#include "spectra.hh" // for to_string_with_precision

#include <algorithm>
#include <dune/istl/bvector.hh>
#include <experimental/simd>
#include <lapacke.h>
#include <memory>
#include <numeric>
#include <sstream>

template <Eigenproblem EVP>
class SRRITSubspaceIteration {
  using Real = typename EVP::Real;
  using BlockMultiVec = typename EVP::BlockMultiVec;
  using BlockMatrix = typename BlockMultiVec::BlockMatrix;
  static constexpr std::size_t blocksize = EVP::blocksize;

public:
  SRRITSubspaceIteration(std::shared_ptr<EVP> evp, const EigensolverParams& params)
      : evp(std::move(evp))
      , Q(this->evp->mat_size(), params.ncv)
      , AQ(this->evp->mat_size(), params.ncv)
      , W(this->evp->mat_size(), params.ncv)
      , T(params.ncv / blocksize)
      , S(params.ncv / blocksize)
      , orth(BetweenBlocks::ModifiedGramSchmidt, WithinBlocks::CholQR, this->evp->get_inner_product())
      , eigenvalues(params.ncv)
      , block_resids(blocksize)
      , nev{params.nev}
      , ncv{params.ncv}
      , maxit{params.maxit}
      , tolerance{params.tolerance}
  {
  }

  void solve()
  {
    std::size_t it = 0;
    std::size_t l = 0; // the first block of Q that has not yet converged
    Q.set_random();
    orth.orthonormalise(Q);

    while (true) {
      prev_eigenvalues = eigenvalues; // Store eigenvalues from previous iteration
      rayleigh_ritz();
      l = first_block_not_converged(l);

      logger::debug("SRRIT iter {}: {} of {} blocks converged (checking first {} of {} in subspace)", it, l, nev / blocksize, nev / blocksize, ncv / blocksize);

      if (l >= nev / blocksize or it >= maxit) break;

      auto [nxtssr, idort, nxtort] = calc_nxtssr_idort_nxtort(it);
      Q = AQ;
      it++;

      // Run a subspace iteration until we need a RR-step again
      while (it != nxtssr) {
        while (it != nxtort) {
          evp->apply(Q, AQ);
          std::swap(Q, AQ);
          it++;
        }

        orth.orthonormalise(Q);

        nxtort = std::min(nxtssr, it + idort);
      }

      // Always orthonormalize before the next Rayleigh-Ritz step
      orth.orthonormalise(Q);
    }
  }

  std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> extract_eigenvectors() const
  {
    using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;

    // Transform eigenvalues from θ to λ using the EVP's transformation
    // Note: we only extract the first nev eigenvalues, even though we computed nsubsp
    auto lambda = evp->transform_eigenvalues(eigenvalues);

    // Create index array for sorting eigenvalues in ascending order
    std::vector<std::size_t> indices(nev);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by transformed eigenvalue (smallest first)
    std::sort(indices.begin(), indices.end(), [&lambda](std::size_t i, std::size_t j) { return lambda[i] < lambda[j]; });
    std::sort(lambda.begin(), lambda.end(), [&lambda](const auto& a, const auto& b) { return a < b; });

    // Extract and reorder eigenvectors based on sorted eigenvalues
    std::vector<Vec> eigenvectors(nev, Vec(evp->mat_size()));

    for (std::size_t i = 0; i < nev; ++i) {
      // Get source index from sorted orderg
      std::size_t src_idx = indices[i];
      std::size_t src_block_idx = src_idx / blocksize;
      std::size_t src_local_idx = src_idx % blocksize;

      auto src_block = Q.block_view(src_block_idx);

      // Copy the vector data from the block to the output format
      for (std::size_t row = 0; row < evp->mat_size(); ++row) eigenvectors[i][row][0] = src_block(row, src_local_idx);
    }

    // Log the transformed eigenvalues
    if (logger::get_level() <= logger::Level::debug) {
      auto eigstring = std::accumulate(lambda.begin() + 1, lambda.begin() + nev, to_string_with_precision(static_cast<double>(lambda[0])),
                                       [](const std::string& a, Real b) { return a + ", " + std::to_string(b); });
      logger::info_all("Computed {} eigenvalues: {}", nev, eigstring);
    }
    else {
      // Sort lambda for display
      auto sorted_lambda = lambda;
      std::sort(sorted_lambda.begin(), sorted_lambda.end());
      logger::info_all("Computed {} eigenvalues: smallest {}, largest {}", nev, sorted_lambda[0], sorted_lambda[nev - 1]);
    }

    return eigenvectors;
  }

private:
  void rayleigh_ritz()
  {
    evp->apply(Q, AQ);
    evp->get_inner_product()->dot(Q, AQ, T);

    // DIAGNOSTIC: Check orthonormality of Q before Rayleigh-Ritz
    if (logger::get_level() <= logger::Level::debug) {
      BlockMultiVec temp(Q.rows(), Q.cols());
      BlockMatrix G(ncv / blocksize);
      evp->get_inner_product()->dot(Q, Q, G);
      Real orthog_error = 0.0;
      for (std::size_t i = 0; i < ncv; ++i) {
        for (std::size_t j = 0; j < ncv; ++j) {
          std::size_t bi = i / blocksize;
          std::size_t bj = j / blocksize;
          std::size_t li = i % blocksize;
          std::size_t lj = j % blocksize;
          Real expected = (i == j) ? 1.0 : 0.0;
          Real actual = G.block_view(bi, bj).data()[li * blocksize + lj];
          orthog_error = std::max(orthog_error, std::abs(actual - expected));
        }
      }
      logger::debug_all("  Orthonormality error ||Q^T*B*Q - I||_∞ = {}", orthog_error);
    }

    auto T_flat = T.to_flat_column_major(); // TODO: This allocates memory, we should just pass the storage vector instead and overwrite it every time
    int info = lapacke::syev(LAPACK_COL_MAJOR, 'V', 'L', ncv, T_flat.data(), ncv, eigenvalues.data());
    if (info != 0) DUNE_THROW(Dune::MathError, "LAPACKE_dsyevd failed with info = " << info);

    // DIAGNOSTIC: Check condition number of projected matrix T
    if (logger::get_level() <= logger::Level::debug) {
      auto max_eigenvalue = *std::max_element(eigenvalues.begin(), eigenvalues.end(), [](Real a, Real b) { return std::abs(a) < std::abs(b); });
      auto min_eigenvalue = *std::min_element(eigenvalues.begin(), eigenvalues.end(), [](Real a, Real b) { return std::abs(a) < std::abs(b); });
      Real cond_T = std::abs(max_eigenvalue) / std::abs(min_eigenvalue);

      logger::debug_all("  Condition number of T: κ(T) = {} (λ_max={}, λ_min={})", cond_T, max_eigenvalue, min_eigenvalue);
      if (cond_T > 1e10) logger::warn("  WARNING: T is very ill-conditioned (κ > 1e10), expect loss of accuracy");
    }

    // LAPACK gives us the eigenvalues in ascending order, we want descending since we want the eigenvalues
    // that converge the fastest first (which are the largest in a subspace iteration). So we reverse the order here.
    // TODO: Get rid of the memory allocations here.
    // TODO: Wrap all of this functionality in a helper class for small dense eigenvalue problems
    std::vector<std::size_t> indices(ncv);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [this](std::size_t i, std::size_t j) { return std::abs(eigenvalues[i]) > std::abs(eigenvalues[j]); });
    std::vector<Real> T_flat_sorted(T_flat.size());
    std::vector<Real> eigenvalues_sorted(ncv);
    for (std::size_t i = 0; i < ncv; ++i) {
      eigenvalues_sorted[i] = eigenvalues[indices[i]];
      for (std::size_t j = 0; j < ncv; ++j) T_flat_sorted[i * ncv + j] = T_flat[indices[i] * ncv + j];
    }
    eigenvalues = std::move(eigenvalues_sorted);
    T_flat = std::move(T_flat_sorted);

    // T.from_flat_column_major(T_flat_sorted);
    T.from_flat_column_major(T_flat);
    Q.mult(T, W);
    std::swap(Q, W);
    AQ.mult(T, W);
    // evp->apply(Q, AQ);
    std::swap(AQ, W);
  }

  std::size_t first_block_not_converged(std::size_t l = 0)
  {
    W = Q;
    W.scale_columns(eigenvalues);
    W -= AQ; // W = Q*diag(evals) - AQ

    // Only check convergence for the first nev eigenpairs (not the full subspace)
    std::size_t nev_blocks = nev / blocksize;

    // DIAGNOSTIC: Track eigenvalue spacing to detect clustering
    std::vector<Real> eigenvalue_gaps(nev > 1 ? nev - 1 : 0);
    for (std::size_t i = 0; i + 1 < nev; ++i) eigenvalue_gaps[i] = std::abs(eigenvalues[i] - eigenvalues[i + 1]) / std::max(std::abs(eigenvalues[i]), 1e-14);

    for (std::size_t i = l; i < nev_blocks; ++i) {
      evp->block_column_norms(W, i, block_resids);

      for (std::size_t j = 0; j < blocksize; ++j) {
        std::size_t global_idx = i * blocksize + j;

        if (std::abs(eigenvalues[global_idx]) != 0.0) block_resids[j] /= std::abs(eigenvalues[global_idx]);

        // Log eigenvalue changes and residuals for monitoring
        if (global_idx < 5 || block_resids[j] > tolerance) { // Show first 5 or unconverged
          Real rel_change = 0.0;
          if (!prev_eigenvalues.empty()) rel_change = std::abs(eigenvalues[global_idx] - prev_eigenvalues[global_idx]) / std::max(std::abs(eigenvalues[global_idx]), 1e-14);

          // Show gap to next eigenvalue for clustering detection
          std::stringstream gap_info;
          if (global_idx + 1 < nev) gap_info << ", gap_to_next=" << eigenvalue_gaps[global_idx];

          logger::trace_all("  λ[{}] = {}, residual = {}, rel_change = {}{}{}", global_idx, eigenvalues[global_idx], block_resids[j], rel_change, gap_info.str(),
                            block_resids[j] <= tolerance ? " ✓" : "");
        }
      }

      // Check if this block has converged
      if (std::any_of(block_resids.begin(), block_resids.end(), [this](Real r) { return r > tolerance; })) return i;
    }

    return nev_blocks; // If we didn't return in the loop, then all blocks have converged
  }

  std::tuple<std::size_t, std::size_t, std::size_t> calc_nxtssr_idort_nxtort(std::size_t it) const noexcept
  {
    // TODO: Implement that proper formula as in the paper
    std::size_t nxtssr = std::min(maxit, std::max((std::size_t)std::floor(stpfac * it), init));

    // Find largest and smallest eigenvalue in magnitude
    auto max_eigenvalue = *std::max_element(eigenvalues.begin(), eigenvalues.end(), [](Real a, Real b) { return std::abs(a) < std::abs(b); });
    auto min_eigenvalue = *std::min_element(eigenvalues.begin(), eigenvalues.end(), [](Real a, Real b) { return std::abs(a) < std::abs(b); });
    auto condT = std::abs(max_eigenvalue) / std::abs(min_eigenvalue);
    auto idort = (std::size_t)std::max(1, (int)std::floor(orttol / std::log10(condT)));
    auto nxtort = std::min(it + idort, nxtssr);

    logger::info_all("SRRIT: Updated iteration numbers: it = {}, nxtssr = {}, nxtort = {}, idort = {}", it, nxtssr, nxtort, idort);

    return std::make_tuple(nxtssr, idort, nxtort);
  }

  // Members
  std::shared_ptr<EVP> evp;

  BlockMultiVec Q;
  BlockMultiVec AQ;
  BlockMultiVec W; // work vector
  typename BlockMultiVec::BlockMatrix T;
  typename BlockMultiVec::BlockMatrix S;

  orthogonalisation::BlockOrthogonalisation<typename EVP::InnerProduct> orth;

  std::vector<Real> eigenvalues;
  std::vector<Real> prev_eigenvalues;
  std::vector<Real> block_resids;
  std::size_t nconv = 0;

  // SRRIT specific parameters
  std::size_t init = 5;    // The number of initial iterations to be performed
  Real stpfac = 1.2;       // A constant used to determine the maximum number of iterations before the next RR step
  Real alpha = 1.0;        // A paramter used in predicting when the next residual will converge
  Real beta = 1.0;         // Another paramter used in predicting when the next residual will converge
  Real grptol = 1e-3;      // A tolerance for grouping equimodular eigenvalues // TODO: Is this useful in our case? Should it be blocks of eigenvalues?
  Real cnvtol = 1e-4;      // A convergence criterion for the average value of a cluster of equimodular eigenvalues
  std::size_t orttol = 10; // The number of decimal digits whose loss can be tolerated in orthonormalisation steps (INCREASED from 2 for stability)

  // Parameters read from the config
  std::size_t nev; // Number of requested eigenvalues
  std::size_t ncv; // Subspace size (>= nev, by default 2*nev)
  std::size_t maxit;
  double tolerance;
};

template <Eigenproblem EVP>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> srrit(std::shared_ptr<EVP> evp, const EigensolverParams& params)
{
  SRRITSubspaceIteration solver(evp, params);
  solver.solve();
  return solver.extract_eigenvectors();
}
