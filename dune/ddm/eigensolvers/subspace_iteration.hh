#pragma once

#include "../logger.hh"
#include "blockmatrix.hh"
#include "concepts.hh"
#include "dune/ddm/eigensolvers/orthogonalisation.hh"
#include "eigensolver_params.hh"
#include "lapacke.hh"

#include <algorithm>
#include <dune/common/exceptions.hh>
#include <dune/istl/bvector.hh>
#include <experimental/simd>
#include <numeric>
#include <vector>

/** @brief Compute a few of the smallest eigenpairs of the generalized eigenvalue problem Ax = \lambda Bx.
 */
template <Eigenproblem EVP>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> subspace_iteration(EVP& evp, const EigensolverParams& params)
{
  auto nev = params.nev;
  auto tolerance = params.tolerance;
  auto maxit = params.maxit;

  // Log events
  auto* apply_event = Logger::get().registerEvent("SubspaceIteration", "apply");
  auto* ortho_event = Logger::get().registerEvent("SubspaceIteration", "orthonormalise");
  auto* reorder_event = Logger::get().registerEvent("SubspaceIteration", "extract evecs");
  auto* rayleigh_ritz_event = Logger::get().registerEvent("SubspaceIteration", "Rayleigh Ritz");
  auto* residual_event = Logger::get().registerEvent("SubspaceIteration", "Residual computation");
  auto* setup_event = Logger::get().registerEvent("SubspaceIteration", "Setup");
  Logger::get().startEvent(setup_event);

  const bool own_methods = true;
  const bool with_locking = true;

  using BlockMultiVec = typename EVP::BlockMultiVec;
  using Real = typename EVP::Real;
  auto blocksize = BlockMultiVec::blocksize;
  auto nev_adjusted = nev < blocksize ? blocksize : nev;

  orthogonalisation::BlockOrthogonalisation orth(BetweenBlocks::ModifiedGramSchmidt, WithinBlocks::CholQR, evp.get_inner_product());

  BlockMultiVec Q1(evp.mat_size(), nev_adjusted);
  BlockMultiVec Q2(evp.mat_size(), nev_adjusted);
  BlockMultiVec Qtmp(evp.mat_size(), nev_adjusted);
  BlockMultiVec Qtmp2(evp.mat_size(), nev_adjusted);

  // Initialise the initial set of vectors randomly
  Q1.set_random(params.seed);
  orth.orthonormalise(Q1);

  typename BlockMultiVec::BlockMatrix T(nev_adjusted / blocksize);

  // Storage for eigenvalues
  std::vector<Real> eigenvalues(nev_adjusted);
  std::vector<Real> prev_eigenvalues(nev_adjusted, 0.0);

  // Precompute A and B norm for residual computation
  // auto Anorm = A.frobenius_norm();
  // auto Bnorm = B.frobenius_norm();

  std::size_t it = 0;
  std::size_t rayleigh_ritz_every = 3;
  double error = 0;
  Logger::get().endEvent(setup_event);
  for (; it < maxit; ++it) {
    logger::trace("SubspaceIteration: Iteration {} (Rayleigh-Ritz: {})", it, (it % rayleigh_ritz_every == 0 ? "yes" : "no"));

    /* In each iteration we do:
       1. Apply the operator
       2. Orthogonalise
       3. Maybe do a Rayleigh-Ritz step
       4. If a Rayleigh-Ritz step was performed, check for convergence:
          First compute a cheap residual based on the difference between previously computed
          Rayleigh quotients and the Rayleigh quotients of the current iteration. If this
          indicates convergence, compute a more accurate residual.
     */

    // Step 1: Apply the operator (e.g., for a shift-inverted GEVP Q2 <- (A - σB)^-1 B Q1)
    Logger::get().startEvent(apply_event);
    evp.apply(Q1, Q2);
    Logger::get().endEvent(apply_event);

    // Step 2: Orthonormalize
    Logger::get().startEvent(ortho_event);
    orth.orthonormalise(Q2);
    Logger::get().endEvent(ortho_event);

    // Step 3: Rayleigh-Ritz procedure
    if (it % rayleigh_ritz_every == 0) {
      Logger::ScopedLog rrlog(rayleigh_ritz_event);

      // Compute T = Q1^T B (A - σB)^{-1}B Q1
      evp.apply(Q2, Q1);                      // Q2 <- (A - σB)^{-1}B Q1
      Qtmp = Q1;                              // Save the result since we might need it below
      evp.get_inner_product()->dot(Q2, Q1, T);                     // T = Q1^T B Q2 (using B-inner product)
      auto T_flat = T.to_flat_column_major(); // TODO: This allocates memory, we should just pass the storage vector instead and overwrite it every time

      // Solve the small eigenvalue problem
      int info = lapacke::syevd(LAPACK_COL_MAJOR, 'V', 'L', nev_adjusted, T_flat.data(), nev_adjusted, eigenvalues.data());
      if (info != 0) DUNE_THROW(Dune::MathError, "LAPACKE_dsyevd failed with info = " << info);

      T.from_flat_column_major(T_flat);
      Q2.mult(T, Q1);

      // Check convergence using eigenvalues from Rayleigh-Ritz
      {
        Logger::ScopedLog reslog(residual_event);

        // Transform eigenvalues from θ to λ using λ = σ + 1/θ
        for (auto& e : eigenvalues) e = 1 / e + params.shift;

        // Check relative change in eigenvalues (using eigenvalues from Rayleigh-Ritz)
        if (it > 0) {
          Real max_rel_change = 0.0;
          Real max_eigenvalue = *std::max_element(eigenvalues.begin(), eigenvalues.end());

          // Here we use nev instead of nev_adjusted, because we're only interested in nev
          for (std::size_t i = 0; i < nev; ++i) {
            Real rel_change = std::abs(eigenvalues[i] - prev_eigenvalues[i]) / max_eigenvalue;
            max_rel_change = std::max(max_rel_change, rel_change);
          }

          // // Here we check if we are somewhat close; if yes, then we compute a more accurate residual
          if (max_rel_change < params.tolerance) break;
          //   // Now compute the full residual to make sure we have actually converged;
          //   // if not, iterate again, but immediatly check again, because it's likely that
          //   // we're close to convergence
          //   // TODO: Check if this residual makes sense.
          //   rayleigh_ritz_every = 1;
          //   Q2.apply_to_mat(B, Qtmp2); // Qtmp2 <- B*Q2, Qtmp already contains A*Q2
          //   Qtmp2.scale_columns(eigenvalues);
          //   Qtmp -= Qtmp2;
          //   auto err = Qtmp.norm() / ((Anorm + max_eigenvalue * Bnorm) * Q1.norm()); //  / Q2.norm();
          //   if (err < tolerance) {
          //     // If we're done, orthonormalise and put the result in Q2
          //     block_classical_gram_schmidt(B, Q1, Q2, own_methods, own_methods);
          //     break;
          //   }
          // }
          error = max_rel_change;
        }

        // Store current eigenvalues for next iteration
        prev_eigenvalues = eigenvalues;
      }
    }
  }

  // Sort eigenvalues and create index mapping
  Logger::get().startEvent(reorder_event);
  {
    // Create index array for sorting
    std::vector<std::size_t> indices(nev_adjusted);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by eigenvalue (smallest first)
    std::sort(indices.begin(), indices.end(), [&eigenvalues](std::size_t i, std::size_t j) { return eigenvalues[i] < eigenvalues[j]; });

    // Reorder eigenvalues
    std::vector<Real> sorted_eigenvalues(nev_adjusted);
    for (std::size_t i = 0; i < nev_adjusted; ++i) sorted_eigenvalues[i] = eigenvalues[indices[i]];
    eigenvalues = std::move(sorted_eigenvalues);

    // Extract and reorder eigenvectors in one step
    using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;
    std::vector<Vec> eigenvectors(nev, Vec(evp.mat_size()));

    for (std::size_t i = 0; i < nev; ++i) {
      // Get source index from sorted order
      std::size_t src_idx = indices[i];
      std::size_t src_block_idx = src_idx / blocksize;
      std::size_t src_local_idx = src_idx % blocksize;

      auto src_block = Q1.block_view(src_block_idx);

      // Copy the vector data from the block to the output format
      for (std::size_t row = 0; row < evp.mat_size(); ++row) eigenvectors[i][row][0] = src_block(row, src_local_idx);
    }

    Logger::get().endEvent(reorder_event);

    if (logger::get_level() <= logger::Level::debug) {
      auto eigstring =
          std::accumulate(eigenvalues.begin() + 1, eigenvalues.begin() + nev, to_string_with_precision(eigenvalues[0]), [](const std::string& a, double b) { return a + ", " + std::to_string(b); });

      logger::info_all("Computed {} eigenvalues: {}", nev, eigstring);
    }
    else {
      logger::info_all("Computed {} eigenvalues: smallest {}, largest {}", nev, eigenvalues[0], eigenvalues[nev - 1]);
    }

    return eigenvectors;
  }
}
