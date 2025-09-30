#pragma once

#include "dune/common/exceptions.hh"

#include "blockmatrix.hh"
#include "blockmultivector.hh"
#include "lapacke.hh"

#include <vector>

/** @file This file implements different (block) orthogonalisation routines.

    Here we use the skeleton-muscle analogy of Hoemmen [1] (see also [2]),
    where the outer block method (the skeleton) uses an "intra orthogonalisation"
    method (the muscle) to orthogonalise one block vector of a blocked multivector.

    This could be, for instance, a block variant of modified Gram-Schmidt combined
    with a Cholesky QR method (which is the only combination currently implemented).

    References:
    [1] M. Hoemmen. 2010. Communication-avoiding krylov subspace methods. Ph.D. Dissertation. University of California at Berkeley, USA.
    [2] E. Carson, K. Lund, M. Rozložník, and S. Thomas, "Block Gram-Schmidt algorithms and their stability properties", Linear Algebra and its Applications, vol. 638, pp. 150–195, 2022.

    Other references used:
    https://arxiv.org/pdf/1401.5171 - Pre-CholQR

 */

template <class Mat, class ConstVecView, class VecView, class DenseMatView>
void chol_qr(const Mat &A, ConstVecView X, VecView Q, DenseMatView R)
{
  // 1. Compute R = X^T A X
  X.apply_to_mat(A, Q);
  X.dot(Q, R);

  // 2. Compute R = chol(C)
  auto info = lapacke::potrf(LAPACK_ROW_MAJOR, 'U', Q.cols(), R.data(), Q.cols());
  if (info != 0) { DUNE_THROW(Dune::Exception, "Cholesky factorization failed with code " << info); }

  // Compute Q = A * R^-1
  Q = X;
  blas::trsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, Q.rows(), Q.cols(), 1., R.data(), Q.cols(), Q.data(), Q.cols());

  for (std::size_t i = 0; i < Q.cols(); ++i) {
    for (std::size_t j = 0; j < Q.cols(); ++j) {
      if (i > j) { R.data()[i * Q.cols() + j] = 0; }
    }
  }
}

template <class ConstVecView, class VecView, class DenseMatView>
void chol_qr(ConstVecView X, VecView Q, DenseMatView R)
{
  // Standard QR: R = X^T * X (not A-inner product)
  X.dot(X, R);

  // Cholesky factorization
  auto info = lapacke::potrf(LAPACK_ROW_MAJOR, 'U', Q.cols(), R.data(), Q.cols());
  if (info != 0) { DUNE_THROW(Dune::Exception, "Standard Cholesky factorization failed with code " << info); }

  // Q = X * R^(-1)
  Q = X;
  blas::trsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, Q.rows(), Q.cols(), 1., R.data(), Q.cols(), Q.data(), Q.cols());

  // Zero out lower triangle
  for (std::size_t i = 0; i < Q.cols(); ++i) {
    for (std::size_t j = 0; j < Q.cols(); ++j) {
      if (i > j) { R.data()[i * Q.cols() + j] = 0; }
    }
  }
}

template <class Mat, class BMV>
void block_classical_gram_schmidt(const Mat &A, const BMV &X, BMV &Q, bool use_own_dot = false, bool use_own_mult = false)
{
  if (!X.matches(Q)) { DUNE_THROW(Dune::Exception, "block_classical_gram_schmidt: The blocked multivectors X and Q must have matching dimension and blocksize"); }

  using Real = typename BMV::Real;
  using BlockViewType = typename BMV::BlockViewType;

  DenseSquareBlockMatrix<Real, BMV::alignment> R(X.blocks(), X.blocksize());

  // Orthogonalize the first block using standard QR
  auto X0 = X.block_view(0);
  auto Q0 = Q.block_view(0);
  auto R00 = R.block_view(0, 0);

  chol_qr(A, X0, Q0, R00);

  // Work vectors
  auto W_ = static_cast<Real *>(std::aligned_alloc(BMV::alignment, X.rows() * X.blocksize() * sizeof(Real)));
  BlockViewType W(W_, X.rows(), X.blocksize());

  auto Z_ = static_cast<Real *>(std::aligned_alloc(BMV::alignment, X.rows() * X.blocksize() * sizeof(Real)));
  BlockViewType Z(Z_, X.rows(), X.blocksize());

  // Temporary storage for all projection coefficients
  std::vector<DenseSquareBlockMatrix<Real, BMV::alignment>> all_projections;
  
  // Block Gram-Schmidt for remaining blocks
  for (std::size_t k = 0; k < X.blocks() - 1; ++k) {
    auto V = X.block_view(k + 1);
    W = V;

    // Classical Gram-Schmidt: 
    // Step 1: Compute ALL projection coefficients against ALL previous blocks
    V.apply_to_mat(A, Z);  // Compute A*V once (key difference from MGS)
    
    all_projections.clear();
    all_projections.reserve(k + 1);
    
    for (std::size_t j = 0; j <= k; ++j) {
      auto Qj = Q.block_view(j);
      
      // Store projection coefficient: R[j,k+1] = Q_j^T * A * V
      all_projections.emplace_back(1, X.blocksize());
      auto curr_proj = all_projections.back().block_view(0, 0);
      auto Rj_k1 = R.block_view(j, k + 1);
      
      if (use_own_dot) { 
        Qj.dot_manual(Z, curr_proj); 
      } else {
        Qj.dot(Z, curr_proj);
      }
      
      // Copy to R matrix for output
      for (std::size_t i = 0; i < X.blocksize(); ++i) {
        for (std::size_t l = 0; l < X.blocksize(); ++l) {
          Rj_k1.data()[i * X.blocksize() + l] = curr_proj.data()[i * X.blocksize() + l];
        }
      }
    }

    // Step 2: Apply ALL projections at once
    // W = V - Σⱼ Qⱼ * R[j,k+1]
    W = V;  // Start with original vector
    
    for (std::size_t j = 0; j <= k; ++j) {
      auto Qj = Q.block_view(j);
      auto curr_proj = all_projections[j].block_view(0, 0);
      
      // Compute Qⱼ * R[j,k+1]
      if (use_own_mult) { 
        Qj.mult_manual(curr_proj, Z); 
      } else {
        Qj.mult(curr_proj, Z);
      }
      
      // Subtract: W = W - Qⱼ * R[j,k+1]
      for (std::size_t i = 0; i < X.rows() * X.blocksize(); ++i) {
        W.data()[i] -= Z.data()[i];
      }
    }

    // Step 3: Orthogonalize the resulting W using standard QR
    auto Qk1 = Q.block_view(k + 1);
    auto Rk1_k1 = R.block_view(k + 1, k + 1);

    Qk1 = W;
    chol_qr(A, W, Qk1, Rk1_k1);
  }

  std::free(W_);
  std::free(Z_);
}

template <class Mat, class BMV>
void block_modified_gram_schmidt(const Mat &A, const BMV &X, BMV &Q, bool use_own_dot = false, bool use_own_mult = false)
{
  if (!X.matches(Q)) { DUNE_THROW(Dune::Exception, "block_modified_gram_schmidt: The blocked multivectors X and Q must have matching dimension and blocksize"); }

  using Real = typename BMV::Real;
  using BlockViewType = typename BMV::BlockViewType;

  DenseSquareBlockMatrix<Real, BMV::alignment> R(X.blocks(), X.blocksize());

  // Orthogonalize the first block using standard QR
  auto X0 = X.block_view(0);
  auto Q0 = Q.block_view(0);
  auto R00 = R.block_view(0, 0);

  chol_qr(A, X0, Q0, R00);

  // Work vectors
  auto W_ = static_cast<Real *>(std::aligned_alloc(BMV::alignment, X.rows() * X.blocksize() * sizeof(Real)));
  BlockViewType W(W_, X.rows(), X.blocksize());

  auto Z_ = static_cast<Real *>(std::aligned_alloc(BMV::alignment, X.rows() * X.blocksize() * sizeof(Real)));
  BlockViewType Z(Z_, X.rows(), X.blocksize());

  // Block Gram-Schmidt for remaining blocks
  for (std::size_t k = 0; k < X.blocks() - 1; ++k) {
    auto V = X.block_view(k + 1);
    W = V;

    // Modified Gram-Schmidt: orthogonalize against one Q_j at a time
    for (std::size_t j = 0; j <= k; ++j) {
      auto Qj = Q.block_view(j);
      auto Rj_k1 = R.block_view(j, k + 1);

      // Compute projection coefficient: R[j,k+1] = Q_j^T * A * W
      W.apply_to_mat(A, Z);
      if (use_own_dot) { Qj.dot_manual(Z, Rj_k1); }
      else {
        Qj.dot(Z, Rj_k1);
      }

      // Subtract projection: W = W - Q_j * R[j,k+1]
      if (use_own_mult) { Qj.mult_manual(Rj_k1, Z); }
      else {
        Qj.mult(Rj_k1, Z);
      }
      for (std::size_t i = 0; i < X.rows() * X.blocksize(); ++i) {
        W.data()[i] -= Z.data()[i];
      }
    }

    // Orthogonalize the resulting W using standard QR
    auto Qk1 = Q.block_view(k + 1);
    auto Rk1_k1 = R.block_view(k + 1, k + 1);

    Qk1 = W;
    chol_qr(A, W, Qk1, Rk1_k1);
  }

  std::free(W_);
  std::free(Z_);
}
