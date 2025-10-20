#pragma once

#include "blockmatrix.hh"
#include "blockmultivector.hh"
#include "concepts.hh"
#include "dune/common/exceptions.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/logger.hh"
#include "lapacke.hh"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

/** @file This file implements different (block) orthogonalisation routines.

    Here we use the skeleton-muscle analogy of Hoemmen [1] (see also [2]),
    where the outer block method (the skeleton) uses an "intra orthogonalisation"
    method (the muscle) to orthogonalise one block vector of a blocked multivector.

    This could be, for instance, a block variant of modified Gram-Schmidt combined
    with a Cholesky QR method (which is the only combination currently implemented).

    References:
    [1] M. Hoemmen. 2010. Communication-avoiding krylov subspace methods. Ph.D. Dissertation. University of California at Berkeley, USA.
    [2] E. Carson, K. Lund, M. Rozložník, and S. Thomas, "Block Gram-Schmidt algorithms and their stability properties", Linear Algebra and its Applications, vol. 638, pp. 150–195, 2022.a
 */

enum class BetweenBlocks : std::uint8_t { ModifiedGramSchmidt, ClassicalGramSchmidt };
enum class WithinBlocks : std::uint8_t { CholQR, CholQR2, PreCholQR, ShiftedCholQR3, ModifiedGramSchmidt, ClassicalGramSchmidt };

namespace orthogonalisation {

/** @brief Block orthogonalisation class using configurable inner products
 *
 * This class provides efficient block orthogonalisation routines that can work with
 * different inner products (standard Euclidean, matrix-induced, etc.).
 *
 * The class follows the skeleton-muscle pattern where:
 * - Skeleton: Inter-block orthogonalisation (Modified Gram-Schmidt)
 * - Muscle: Intra-block orthogonalisation (CholQR variants)
 */
template <InnerProduct InnerProd>
class BlockOrthogonalisation {
public:
  BlockOrthogonalisation(BetweenBlocks skeleton, WithinBlocks muscle, std::shared_ptr<InnerProd> ip, int max_reorth_iterations = 5, bool enable_reorth = true)
      : skeleton(skeleton)
      , muscle(muscle)
      , ip(std::move(ip))
      , max_reorth_iterations(max_reorth_iterations)
      , enable_reorth(enable_reorth)
  {
  }

  /** @brief Set reorthogonalization parameters */
  void set_reorthogonalization(bool enable, int max_iterations = 5)
  {
    enable_reorth = enable;
    max_reorth_iterations = max_iterations;
  }

  /** @brief Get current reorthogonalization settings */
  std::pair<bool, int> get_reorthogonalization_settings() const { return {enable_reorth, max_reorth_iterations}; }

  /** @brief Orthogonalises a block of \p Q against all previous blocks and orthonormalises the block itself.
   *
   * This function is designed to be used in block Lanczos iterations where we need to:
   * 1. Orthogonalize block `block` against all previous blocks 0, 1, ..., block-1
   * 2. Extract the coefficients that form the tridiagonal Lanczos matrix
   *
   * @param Q Block multivector containing the Lanczos vectors
   * @param block Index of the block to orthogonalize (0-based)
   * @param alpha Optional output: Will contain Q_{j}^T A Q_block for j < block (off-diagonal blocks of T)
   * @param beta Optional output: Will contain the R factor from QR of the final orthogonalized block (diagonal block of T)
   */
  template <class BMV>
  void orthonormalise_block_against_previous(BMV& Q, std::size_t block, typename BMV::BlockMatrixBlockView* alpha = nullptr, typename BMV::BlockMatrixBlockView* beta = nullptr)
  {
    if (block == 0) {
      // First block: just orthonormalize
      auto Q0 = Q.block_view(0);
      if (beta) {
        if (!call_muscle(Q0, Q0, *beta)) std::cout << "Breakdown, trying to continue\n";
      }
      else {
        typename BMV::BlockMatrix dummy_beta(1);
        auto dummy_view = dummy_beta.block_view(0, 0);
        call_muscle(Q0, Q0, dummy_view);
      }
      return;
    }

    // For subsequent blocks, use the selected orthogonalisation method
    if (skeleton == BetweenBlocks::ModifiedGramSchmidt) orthonormalise_block_mgs(Q, block, alpha, beta);
    else if (skeleton == BetweenBlocks::ClassicalGramSchmidt) orthonormalise_block_cgs(Q, block, alpha, beta);
    else TODO("Not implemented");
  }

  /** @brief Orthonormalises a full block multivector in-place */
  template <class BMV>
  void orthonormalise(BMV& Q)
  {
    typename BMV::BlockMatrix R(1);
    auto Rv = R.block_view(0, 0);

    BMV W(Q.rows(), Q.blocksize); // Working block vector
    auto Wv = W.block_view(0);

    BMV Z(Q.rows(), Q.blocksize); // Temporary for projections
    auto Zv = Z.block_view(0);

    // Orthonormalize first block
    auto Q0 = Q.block_view(0);
    Wv = Q0;                 // Copy to working vector
    call_muscle(Wv, Q0, Rv); // Orthonormalize and write back

    for (std::size_t k = 0; k < Q.blocks() - 1; ++k) {
      auto Qnext = Q.block_view(k + 1);
      Wv = Qnext; // Copy block k+1 to working vector

      for (std::size_t j = 0; j < k; ++j) {
        auto Qj = Q.block_view(j);
        ip->dot(Qj, Wv, Rv);
        Qj.mult(Rv, Zv);
        Wv -= Zv;
      }

      call_muscle(Wv, Qnext, Rv); // Orthonormalize and write back to Q
    }
  }

  // private:
  BetweenBlocks skeleton;
  WithinBlocks muscle;

  std::shared_ptr<InnerProd> ip;
  int max_reorth_iterations;
  bool enable_reorth;

  /** @brief Compute maximum absolute value in a dense block matrix */
  template <class DenseMatView>
  typename DenseMatView::Real max_abs_entry(const DenseMatView& mat) const
  {
    using Real = typename DenseMatView::Real;
    Real max_val = 0;
    for (std::size_t i = 0; i < mat.rows(); ++i)
      for (std::size_t j = 0; j < mat.cols(); ++j) max_val = std::max(max_val, std::abs(mat(i, j)));
    return max_val;
  }

  /** @brief Compute Frobenius norm of a block vector */
  template <class BlockView>
  typename BlockView::Real block_norm(const BlockView& block) const
  {
    using Real = typename BlockView::Real;
    Real norm_sq = 0;
    for (std::size_t i = 0; i < block.rows() * block.cols(); ++i) {
      Real val = block.data()[i];
      norm_sq += val * val;
    }
    return std::sqrt(norm_sq);
  }

  template <class ConstVecView, class VecView, class DenseMatView>
  bool call_muscle(ConstVecView X, VecView Q, DenseMatView R)
  {
    // If the blocksize is 1, there is no need to call any sophisticated method,
    // just divide by the norm of the given "block" (=column).
    if constexpr (VecView::blocksize == 1) {
      ip->dot(X, X, R);
      auto& r = R(0, 0);
      r = std::sqrt(r);
      Q *= 1. / r;
      return r < std::numeric_limits<typename VecView::Real>::epsilon();
    }
    else {
      if (muscle == WithinBlocks::CholQR) return chol_qr_ip(X, Q, R);
      else if (muscle == WithinBlocks::CholQR2) return chol_qr2_ip(X, Q, R);
      else if (muscle == WithinBlocks::PreCholQR) return precholqr_ip(X, Q, R);
      else if (muscle == WithinBlocks::ShiftedCholQR3) return shifted_cholqr3_ip(X, Q, R);
      else if (muscle == WithinBlocks::ModifiedGramSchmidt) return mgs_ip(X, Q, R);
      else if (muscle == WithinBlocks::ClassicalGramSchmidt) return cgs_ip(X, Q, R);
      else {
        TODO("Not implemented");
        return false;
      }
    }
  }

  /** @brief Helper function for Modified Gram-Schmidt orthogonalisation with coefficient extraction */
  template <class BMV>
  bool orthonormalise_block_mgs(BMV& Q, std::size_t block, typename BMV::BlockMatrixBlockView* alpha, typename BMV::BlockMatrixBlockView* beta)
  {
    using Real = typename BMV::Real;
    using BlockViewType = typename BMV::BlockViewType;
    using DenseBlockMatrix = typename BMV::BlockMatrix;
    using DenseBlockMatrixBlockView = typename BMV::BlockMatrixBlockView;

    const Real machine_eps = std::numeric_limits<Real>::epsilon();
    const Real reorth_tol = 10.0 * machine_eps; // Slightly more conservative than machine epsilon

    // Work vectors
    auto W_ = static_cast<Real*>(std::aligned_alloc(BMV::alignment, Q.rows() * Q.blocksize * sizeof(Real)));
    BlockViewType W(W_, Q.rows());

    auto Z_ = static_cast<Real*>(std::aligned_alloc(BMV::alignment, Q.rows() * Q.blocksize * sizeof(Real)));
    BlockViewType Z(Z_, Q.rows());

    auto V = Q.block_view(block);
    W = V; // Copy the block to be orthogonalized

    // Allocate coefficient matrices
    DenseBlockMatrix coeff_storage(1);
    auto coeff = coeff_storage.block_view(0, 0);

    DenseBlockMatrix coeff_correction_storage(1);
    auto coeff_correction = coeff_correction_storage.block_view(0, 0);

    // Modified Gram-Schmidt: orthogonalise against one Qⱼ at a time
    for (std::size_t j = 0; j < block; ++j) {
      auto Qj = Q.block_view(j);

      // Initial orthogonalization: coeff = <Qⱼ, W>_IP
      ip->dot(Qj, W, alpha ? *alpha : coeff);

      // Store the initial coefficient for accumulation
      if (alpha) coeff = *alpha;

      // Subtract projection: W = W - Qⱼ * coeff
      Qj.mult(alpha ? *alpha : coeff, Z); // Z = Qⱼ * coeff
      for (std::size_t i = 0; i < Q.rows() * Q.blocksize; ++i) W.data()[i] -= Z.data()[i];

      // Iterative reorthogonalization (if enabled)
      if (enable_reorth) {
        Real W_norm = block_norm(W);

        for (int reorth_iter = 0; reorth_iter < max_reorth_iterations; ++reorth_iter) {
          // Measure orthogonality error
          ip->dot(Qj, W, coeff_correction);
          Real ortho_err = max_abs_entry(coeff_correction);
          logger::trace_all("Ortho iteration {}, ortho_err = {}, tol = {}", reorth_iter, ortho_err, reorth_tol * W_norm);

          // Check convergence
          if (ortho_err <= reorth_tol * W_norm) {
            if (reorth_iter > 0) logger::info_all("Reorthogonalization converged after {} iterations, ortho_err = {}, tol = {}", reorth_iter, ortho_err, reorth_tol * W_norm);
            break;
          }

          // Apply correction: W = W - Qⱼ * coeff_correction
          Qj.mult(coeff_correction, Z);
          for (std::size_t i = 0; i < Q.rows() * Q.blocksize; ++i) W.data()[i] -= Z.data()[i];

          // Accumulate coefficients (crucial for maintaining accuracy)
          if (alpha) {
            for (std::size_t i = 0; i < alpha->rows(); ++i)
              for (std::size_t k = 0; k < alpha->cols(); ++k) (*alpha)(i, k) += coeff_correction(i, k);
          }
          else {
            for (std::size_t i = 0; i < coeff.rows(); ++i)
              for (std::size_t k = 0; k < coeff.cols(); ++k) coeff(i, k) += coeff_correction(i, k);
          }

          // Update norm for next iteration
          W_norm = block_norm(W);

          if (reorth_iter == max_reorth_iterations - 1)
            logger::warn_all("Reorthogonalization did not converge after {} iterations, final ortho_err = {}, tol = {}", max_reorth_iterations, ortho_err, reorth_tol * W_norm);
        }
      }
    }

    // Final QR step to orthonormalize the result
    auto Q_block = Q.block_view(block);
    if (beta) {
      if (!call_muscle(W, Q_block, *beta)) {
        std::free(W_);
        std::free(Z_);
        return false;
      }
    }
    else {
      DenseBlockMatrix dummy_beta(1);
      auto dummy_view = dummy_beta.block_view(0, 0);
      if (!call_muscle(W, Q_block, dummy_view)) {
        std::free(W_);
        std::free(Z_);
        return false;
      }
    }

    std::free(W_);
    std::free(Z_);
    return true;
  }

  template <class BMV>
  bool orthonormalise_block_cgs(BMV& Q, std::size_t block, typename BMV::BlockMatrixBlockView* alpha, typename BMV::BlockMatrixBlockView* beta)
  {
    typename BMV::BlockMatrix R(std::max(block, 2UL)); // TODO: Avoid these heap allocations
    BMV W_storage(Q.rows(), BMV::blocksize);
    auto W = W_storage.block_view(0);

    auto Qb = Q.block_view(block);

    // First pass
    for (std::size_t j = 0; j < block; ++j) {
      auto Qj = Q.block_view(j);
      auto Rj = R.block_view(0, j);
      ip->dot(Qj, Qb, Rj);
      // Qj.dot(Qb, Rj);
      Qj.mult(Rj, W);
      Qb -= W;
    }

    // Second pass
    if (enable_reorth) {
      for (int it = 0; it < max_reorth_iterations; ++it) {
        for (std::size_t j = 0; j < block; ++j) {
          auto Qj = Q.block_view(j);
          auto Rj = R.block_view(1, j);
          auto Rj0 = R.block_view(0, j);
          ip->dot(Qj, Qb, Rj);
          // Qj.dot(Qb, Rj);
          Rj0 += Rj;

          Qj.mult(Rj, W);
          Qb -= W;
        }
      }
    }

    if (alpha) {
      auto Rb = R.block_view(0, block - 1);
      *alpha = Rb;
    }

    W = Qb;
    return call_muscle(W, Qb, *beta);
  }

  /** @brief CholQR factorisation using configurable inner product */
  template <class ConstVecView, class VecView, class DenseMatView>
  bool chol_qr_ip(ConstVecView X, VecView Q, DenseMatView R)
  {
    using Real = typename VecView::Real;
    bool breakdown = false;

    // Compute Gram matrix: R = <X, X>_IP
    ip->dot(X, X, R);

    // // Check for breakdown: if R is nearly zero, the vectors are linearly dependent
    // Real max_diag = 0;
    // for (std::size_t i = 0; i < Q.cols(); ++i) max_diag = std::max(max_diag, std::abs(R(i, i)));
    // const Real breakdown_tol = 1e-14;
    // if (max_diag < breakdown_tol) {
    //   logger::warn_all("Breakdown in CholQR");a
    //   breakdown = true; // Breakdown: vectors are (nearly) zero or linearly dependent, try to continue anyway
    // }

    // Compute Cholesky decomposition
    int info = lapacke::potrf(LAPACK_ROW_MAJOR, 'U', Q.cols(), R.data(), Q.cols());
    if (info != 0) {
      logger::warn_all("Cholesky factorisation failed in CholQR");
      return false; // Cholesky failed - matrix is not positive definite
    }

    // Compute Q = X * R^-1
    Q = X;
    blas::trsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, Q.rows(), Q.cols(), 1., R.data(), Q.cols(), Q.data(), Q.cols());

    // Zero out lower triangle
    for (std::size_t i = 0; i < Q.cols(); ++i) {
      for (std::size_t j = 0; j < Q.cols(); ++j)
        if (i > j) R.data()[i * Q.cols() + j] = 0;
    }

    return !breakdown;
  }

  /** @brief Enhanced CholQR with multiple applications (CholQR2 variant) */
  template <class ConstVecView, class VecView, class DenseMatView>
  bool chol_qr2_ip(ConstVecView X, VecView Q, DenseMatView R)
  {
    using Real = typename VecView::Real;
    auto* tmp_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * X.rows() * X.cols()));
    VecView tmp(tmp_ptr, X.rows());

    // Allocate temporary matrices for R factors
    auto N = R.rows();
    auto* R1_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * N * N));
    auto* R2_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * N * N));
    DenseMatView R1(R1_ptr);
    DenseMatView R2(R2_ptr);

    bool ok1 = true;
    bool ok2 = true;

    // First QR: X = tmp * R1
    ok1 = chol_qr_ip(X, tmp, R1);
    if (!ok1) logger::warn_all("First orthonormalisation in CholQR2 failed");

    // Second QR: tmp = Q * R2
    ok2 = chol_qr_ip(tmp, Q, R2);
    if (!ok2) logger::warn_all("Second orthonormalisation in CholQR2 failed");

    // Combine R factors: R = R2 * R1 (so that X = Q * R)
    blas::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, R2.data(), N, R1.data(), N, 0.0, R.data(), N);

    std::free(tmp_ptr);
    std::free(R1_ptr);
    std::free(R2_ptr);
    return ok1 && ok2;
  }

  /** @brief PreCholQR factorisation: standard QR followed by CholQR */
  template <class ConstVecView, class VecView, class DenseMatView>
  bool precholqr_ip(ConstVecView X, VecView Q, DenseMatView R)
  {
    using Real = typename VecView::Real;

    // Debug: Check input matrix condition
    Real X_norm = 0;
    for (std::size_t i = 0; i < X.rows() * X.cols(); ++i) X_norm = std::max(X_norm, std::abs(X.data()[i]));
    logger::info_all("PreCholQR: Input matrix max norm = {}", X_norm);

    // Step 1: Standard QR factorization [Y, S] = qr(X)
    // Allocate memory for Y (Q factor) and tau (elementary reflectors)
    auto* Y_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * X.rows() * X.cols()));
    VecView Y(Y_ptr, X.rows());

    auto* tau_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * X.cols()));

    // Allocate memory for S (R factor from standard QR)
    auto N = R.rows();
    auto* S_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * N * N));
    DenseMatView S(S_ptr);

    // Copy X to Y for in-place QR factorization
    Y = X;

    // Compute QR factorization: Y = Q_householder * S
    int info = lapacke::geqrf(LAPACK_ROW_MAJOR, Y.rows(), Y.cols(), Y.data(), Y.cols(), tau_ptr);
    if (info != 0) {
      logger::warn_all("Standard QR factorization failed in PreCholQR with info = {}", info);
      std::free(Y_ptr);
      std::free(tau_ptr);
      std::free(S_ptr);
      return false;
    }

    // Extract upper triangular R factor (S) from Y
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < N; ++j)
        if (i <= j) S(i, j) = Y.data()[i * Y.cols() + j];
        else S(i, j) = 0.0;
    }

    // Generate explicit Q factor (Y) from elementary reflectors
    info = lapacke::orgqr(LAPACK_ROW_MAJOR, Y.rows(), Y.cols(), Y.cols(), Y.data(), Y.cols(), tau_ptr);
    if (info != 0) {
      logger::warn_all("Q generation failed in PreCholQR with info = {}", info);
      std::free(Y_ptr);
      std::free(tau_ptr);
      std::free(S_ptr);
      return false;
    }

    // Step 2: CholQR2 on Y with inner product: [Q, U] = cholQR2(Y)
    auto* U_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * N * N));
    DenseMatView U(U_ptr);

    bool cholqr_ok = chol_qr2_ip(Y, Q, U);
    if (!cholqr_ok) {
      logger::warn_all("CholQR2 step failed in PreCholQR");
      std::free(Y_ptr);
      std::free(tau_ptr);
      std::free(S_ptr);
      std::free(U_ptr);
      return false;
    }

    // Step 3: Combine R factors: R = U * S
    blas::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, U.data(), N, S.data(), N, 0.0, R.data(), N);

    // Clean up
    std::free(Y_ptr);
    std::free(tau_ptr);
    std::free(S_ptr);
    std::free(U_ptr);

    return true;
  }

  /** @brief ShiftedCholQR3: Shifted Cholesky QR with adaptive regularization followed by CholQR2 refinement */
  template <class ConstVecView, class VecView, class DenseMatView>
  bool shifted_cholqr3_ip(ConstVecView X, VecView Q, DenseMatView R)
  {
    using Real = typename VecView::Real;

    const auto m = X.rows();
    const auto n = X.cols();
    const Real machine_eps = std::numeric_limits<Real>::epsilon();

    // Step 1: Q := X
    Q = X;

    // Step 2: Compute Gram matrix A := Q^T * B * Q (using inner product)
    auto* A_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * n * n));
    DenseMatView A(A_ptr);
    ip->dot(Q, Q, A);

    // Step 3: Compute Frobenius norm squared of X
    Real X_frob_sq = 0;
    for (std::size_t i = 0; i < m * n; ++i) {
      Real val = X.data()[i];
      X_frob_sq += val * val;
    }

    // Compute adaptive shift: s := 11 * (mn + n(n+1)) * u * ||X||_F^2
    Real shift = 11.0 * (m * n + n * (n + 1)) * machine_eps * X_frob_sq;

    // Step 4: Add shift to diagonal: A := A + s*I
    for (std::size_t i = 0; i < n; ++i) A(i, i) += shift;

    // Step 4: R := chol(A + sI)
    auto* R1_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * n * n));
    DenseMatView R1(R1_ptr);

    // Copy A to R1 for Cholesky decomposition
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) R1(i, j) = A(i, j);

    int info = lapacke::potrf(LAPACK_ROW_MAJOR, 'U', n, R1.data(), n);
    if (info != 0) {
      logger::warn_all("Shifted Cholesky factorization failed in ShiftedCholQR3 with info = {}", info);
      std::free(A_ptr);
      std::free(R1_ptr);
      return false;
    }

    // Zero out lower triangle of R1
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j)
        if (i > j) R1(i, j) = 0.0;
    }

    // Step 5: Q := Q * R1^-1
    blas::trsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1.0, R1.data(), n, Q.data(), n);

    // Step 6: Apply CholQR2 for refinement
    auto* R2_ptr = static_cast<Real*>(std::aligned_alloc(VecView::alignment, sizeof(Real) * n * n));
    DenseMatView R2(R2_ptr);

    bool cholqr2_ok = chol_qr2_ip(Q, Q, R2);
    if (!cholqr2_ok) {
      logger::warn_all("CholQR2 refinement failed in ShiftedCholQR3");
      std::free(A_ptr);
      std::free(R1_ptr);
      std::free(R2_ptr);
      return false;
    }

    // Combine R factors: R = R2 * R1
    blas::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, R2.data(), n, R1.data(), n, 0.0, R.data(), n);

    // Clean up
    std::free(A_ptr);
    std::free(R1_ptr);
    std::free(R2_ptr);

    return true;
  }

  template <class ConstVecView, class VecView, class DenseMatView>
  bool cgs_ip(ConstVecView X, VecView Q, DenseMatView R)
  {
    (void)X;
    (void)Q;
    (void)R;
    TODO("cgs");
    return true;
  }

  /** @brief Modified Gram-Schmidt within one block
   *
   * This is quite inefficient because this method has to copy each column of
   * the block to a vector (thus essentially transforming it from a row-major to a
   * column-major layout).
   */
  template <class ConstVecView, class VecView, class DenseMatView>
  bool mgs_ip(ConstVecView X, VecView Q, DenseMatView R)
  {
    using Real = typename VecView::Real;
    const auto m = X.rows();
    const auto n = X.cols();
    const Real breakdown_tol = 1e-14;
    const Real machine_eps = std::numeric_limits<Real>::epsilon();
    const Real reorth_tol = 10.0 * machine_eps;

    Q = X;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) R(i, j) = 0.0;

    bool breakdown = false;

    // Temporary vectors for extracting columns
    std::vector<Real> qj_col(m);
    std::vector<Real> qk_col(m);

    for (std::size_t k = 0; k < n; ++k) {

      // Extract column k into temporary vector
      for (std::size_t i = 0; i < m; ++i) qk_col[i] = Q(i, k);

      // Orthogonalize column k against all previous columns 0, 1, ..., k-1
      for (std::size_t j = 0; j < k; ++j) {
        // Extract column j into temporary vector
        for (std::size_t i = 0; i < m; ++i) qj_col[i] = Q(i, j);

        // Initial orthogonalization
        Real coeff = ip->dot(qj_col, qk_col);
        R(j, k) = coeff;

        for (std::size_t i = 0; i < m; ++i) qk_col[i] -= coeff * qj_col[i];

        // Iterative reorthogonalization (if enabled)
        if (enable_reorth) {
          Real qk_norm = std::sqrt(ip->dot(qk_col, qk_col));

          for (int reorth_iter = 0; reorth_iter < max_reorth_iterations; ++reorth_iter) {
            Real coeff_correction = ip->dot(qj_col, qk_col);

            // Check convergence
            if (std::abs(coeff_correction) <= reorth_tol * qk_norm) {
              if (reorth_iter > 0) logger::trace_all("MGS reorthogonalization converged after {} iterations for column {},{}, {}", reorth_iter, k, j, std::abs(coeff_correction));
              break;
            }

            // Apply correction
            for (std::size_t i = 0; i < m; ++i) qk_col[i] -= coeff_correction * qj_col[i];

            // Accumulate coefficient
            R(j, k) += coeff_correction;

            // Update norm
            qk_norm = std::sqrt(ip->dot(qk_col, qk_col));

            if (reorth_iter == max_reorth_iterations - 1)
              logger::warn_all("MGS reorthogonalization did not converge after {} iterations for column {},{}, final error = {}", max_reorth_iterations, k, j, std::abs(coeff_correction));
          }
        }
      }

      Real norm_squared = ip->dot(qk_col, qk_col);
      Real norm_k = std::sqrt(norm_squared);

      // Check for breakdown
      if (norm_k < breakdown_tol) {
        logger::warn_all("Breakdown in MGS: column {} has norm {}", k, norm_k);
        breakdown = true;
        norm_k = 1.0; // Continue with unit normalization to avoid division by zero
      }

      R(k, k) = norm_k;

      for (std::size_t i = 0; i < m; ++i) qk_col[i] /= norm_k;
      for (std::size_t i = 0; i < m; ++i) Q.data()[i * n + k] = qk_col[i];
    }

    return !breakdown;
  }
};

// Close the namespace
} // namespace orthogonalisation
