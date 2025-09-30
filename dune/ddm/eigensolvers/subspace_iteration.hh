#pragma once

#include "../logger.hh"
#include "blockmultivector.hh"
#include "eigensolver_params.hh"
#include "lapacke.hh"
#include "orthogonalisation.hh"
#include "umfpack.hh"

#include <dune/common/exceptions.hh>
#include <dune/istl/bvector.hh>
#include <experimental/simd>
#include <vector>

/** @brief Compute a few of the smallest eigenpairs of the generalized eigenvalue problem Ax = \lambda Bx.
 */
template <class Mat, class Real = double, std::size_t blocksize = 2 * std::experimental::native_simd<Real>::size()>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> subspace_iteration(const Mat& A, const Mat& B, const EigensolverParams& params)
{
  auto nev = params.nev;
  auto shift = params.shift;
  auto tolerance = params.tolerance;
  auto maxit = params.maxit;

  // Log events
  auto* apply_event = Logger::get().registerEvent("SubspaceIteration", "applyB");
  auto* solve_event = Logger::get().registerEvent("SubspaceIteration", "solve");
  auto* ortho_event = Logger::get().registerEvent("SubspaceIteration", "orthogonalise");
  auto* reorder_event = Logger::get().registerEvent("SubspaceIteration", "extract evecs");
  auto* rayleigh_ritz_event = Logger::get().registerEvent("SubspaceIteration", "Rayleigh Ritz");
  auto* residual_event = Logger::get().registerEvent("SubspaceIteration", "Residual computation");
  auto* setup_event = Logger::get().registerEvent("SubspaceIteration", "Setup");
  Logger::get().startEvent(setup_event);

  const bool own_methods = true;
  const bool with_locking = true;

  using BMV = BlockMultiVector<Real, blocksize>;
  auto nev_adjusted = nev < blocksize ? blocksize : nev;

  BMV Q1(A.N(), nev_adjusted);
  BMV Q2(A.N(), nev_adjusted);
  BMV Qtmp(A.N(), nev_adjusted);
  BMV Qtmp2(A.N(), nev_adjusted);

  // Initialise the initial set of vectors randomly
  Q2.set_random(params.seed);
  block_classical_gram_schmidt(B, Q2, Q1, own_methods, own_methods);

  // Set up the solver for the iteration
  // First, create matrix A - sigma B (shift-and-invert approach)
  auto A_minus_sigma_B = A;
  A_minus_sigma_B.axpy(-shift, B);
  UMFPackMultivecSolver solver(A_minus_sigma_B);

  DenseSquareBlockMatrix<Real, BMV::alignment> T(nev_adjusted / blocksize, blocksize);

  // Storage for eigenvalues
  std::vector<Real> eigenvalues(nev_adjusted);
  std::vector<Real> prev_eigenvalues(nev_adjusted, 0.0);

  // Precompute A and B norm for residual computation
  auto Anorm = A.frobenius_norm();
  auto Bnorm = B.frobenius_norm();

  std::size_t it = 0;
  std::size_t rayleigh_ritz_every = 3;
  double error = 0;
  Logger::get().endEvent(setup_event);
  for (; it < maxit; ++it) {
    /* In each iteration we do:
       1. Solve             Q1 <- (A - σB)^-1 B Q1
       2. Orthogonalise     Q2 <- orth_B(Q1)
       3. Maybe do a Rayleigh-Ritz step
       4. If a Rayleigh-Ritz step was performed, check for convergence:
          First compute a cheap residual based on the difference between previously computed
          Rayleigh quotients and the Rayleigh quotients of the current iteration. If this
          indicates convergence, compute a more accurate residual.
     */

    // Step 1: Solve Q1 <- (A - σB)^-1 B Q1
    Logger::get().startEvent(apply_event);
    Q1.apply_to_mat(B, Q2); // Q2 = B * Q1
    Logger::get().endEvent(apply_event);

    Logger::get().startEvent(solve_event);
    solver.solve(Q1, Q2); // Q1 = (A-σB)^-1 * Q2 = (A-σB)^-1 * B * Q1
    Logger::get().endEvent(solve_event);

    // Step 2: Orthogonalize Q2 <- orth_B(Q1)
    Logger::get().startEvent(ortho_event);
    block_classical_gram_schmidt(B, Q1, Q2, own_methods, own_methods);
    Logger::get().endEvent(ortho_event);

    // Step 3: Rayleigh-Ritz procedure
    if (it % rayleigh_ritz_every == 0) {
      Logger::ScopedLog rrlog(rayleigh_ritz_event);

      // Compute T = Q2^T*A*Q2
      Q2.apply_to_mat(A, Q1);
      Qtmp = Q1; // Save the result since we might need it below
      Q2.dot(Q1, T);
      auto T_flat = T.to_flat_column_major(); // TODO: This allocates memory, we should just pass the storage vector instead and overwrite it every time

      // Solve the small eigenvalue problem
      int info = lapacke::syevd(LAPACK_COL_MAJOR, 'V', 'L', nev_adjusted, T_flat.data(), nev_adjusted, eigenvalues.data());
      if (info != 0) DUNE_THROW(Dune::MathError, "LAPACKE_dsyevd failed with info = " << info);

      T.from_flat_column_major(T_flat);
      Q2.mult(T, Q1);

      // Check convergence using eigenvalues from Rayleigh-Ritz
      {
        Logger::ScopedLog reslog(residual_event);

        // Check relative change in eigenvalues (using eigenvalues from Rayleigh-Ritz)
        if (it > 0) {
          Real max_rel_change = 0.0;
          Real max_eigenvalue = *std::max_element(eigenvalues.begin(), eigenvalues.end());

          // Here we use nev instead of nev_adjusted, because we're only interested in nev
          for (std::size_t i = 0; i < nev; ++i) {
            Real rel_change = std::abs(eigenvalues[i] - prev_eigenvalues[i]) / max_eigenvalue;
            max_rel_change = std::max(max_rel_change, rel_change);
          }

          // Here we check if we are somewhat close; if yes, then we compute a more accurate residual
          if (max_rel_change < 1000 * params.tolerance) {
            // Now compute the full residual to make sure we have actually converged;
            // if not, iterate again, but immediatly check again, because it's likely that
            // we're close to convergence
            // TODO: Check if this residual makes sense.
            rayleigh_ritz_every = 1;
            Q2.apply_to_mat(B, Qtmp2); // Qtmp2 <- B*Q2, Qtmp already contains A*Q2
            Qtmp2.scale_columns(eigenvalues);
            Qtmp -= Qtmp2;
            auto err = Qtmp.norm() / ((Anorm + max_eigenvalue * Bnorm) * Q1.norm()); //  / Q2.norm();
            if (err < tolerance) {
              // If we're done, orthonormalise and put the result in Q2
              block_classical_gram_schmidt(B, Q1, Q2, own_methods, own_methods);
              break;
            }
          }
          error = max_rel_change;
        }

        // Store current eigenvalues for next iteration
        prev_eigenvalues = eigenvalues;
      }
    }
    else {
      std::swap(Q1, Q2); // The first step expects Q1 to contain the current approximations
    }
  }

  if (logger::get_level() <= logger::Level::debug) {
    auto eigstring =
        std::accumulate(eigenvalues.begin() + 1, eigenvalues.begin() + nev, to_string_with_precision(eigenvalues[0]), [](const std::string& a, double b) { return a + ", " + std::to_string(b); });

    logger::info_all("Computed {} eigenvalues: {}", nev, eigstring);
  }
  else {
    logger::info_all("Computed {} eigenvalues: smallest {}, largest {}", nev, eigenvalues[0], eigenvalues[nev - 1]);
  }

  Logger::get().startEvent(reorder_event);
  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;
  std::vector<Vec> eigenvectors(nev, Vec(A.N()));

  // Extract eigenvectors from Q1 (the final B-orthogonal Ritz vectors)
  for (std::size_t i = 0; i < nev; ++i) {
    // Determine which block and local index this vector belongs to
    std::size_t block_idx = i / blocksize;
    std::size_t local_idx = i % blocksize;

    // Get the block view
    auto block = Q1.block_view(block_idx);

    // Copy the vector data from the block to the output format
    for (std::size_t row = 0; row < A.N(); ++row) eigenvectors[i][row][0] = block(row, local_idx);
  }

  Logger::get().endEvent(reorder_event);

  return eigenvectors;
}
