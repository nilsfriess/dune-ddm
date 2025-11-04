#pragma once

#include "dune/ddm/eigensolvers/blockmatrix.hh"
#include "dune/ddm/eigensolvers/concepts.hh"
#include "dune/ddm/eigensolvers/eigensolver_params.hh"
#include "dune/ddm/eigensolvers/inner_products.hh"
#include "dune/ddm/eigensolvers/orthogonalisation.hh"
#include "dune/ddm/logger.hh"

#include <cstddef>

template <Eigenproblem EVP>
class BlockLanczos {
  using BMV = typename EVP::BlockMultiVec;
  using BlockMatrix = typename EVP::BlockMatrix;
  using BlockMatrixBlockView = typename EVP::BlockMatrixBlockView;
  using Real = typename EVP::Real;
  using InnerProduct = typename EVP::InnerProduct;
  using Ortho = orthogonalisation::BlockOrthogonalisation<InnerProduct>;
  static constexpr std::size_t blocksize = EVP::blocksize;
  static constexpr Real eps = std::numeric_limits<Real>::epsilon();

public:
  BlockLanczos(std::shared_ptr<EVP> evp, std::shared_ptr<Ortho> orth, const EigensolverParams& params)
      : evp(std::move(evp))
      , orth(std::move(orth))
      , V(this->evp->mat_size(), params.ncv + blocksize)
      , W(this->evp->mat_size(), 2 * blocksize)
      , T(params.ncv / blocksize)
      , B(1)
      , Z(1)
  {
    W.set_random();
    this->evp->apply_block(0, W, V);
    auto beta = B.block_view(0, 0);
    this->orth->orthonormalise_block_against_previous(V, 0, nullptr, &beta);
  }

  //  Extend a step-k Lanczos decomposition to a step-m Lanczos decomposition
  void extend(std::size_t k, std::size_t m)
  {
    const auto beta_threshold = eps * std::sqrt(evp->mat_size());
    auto beta = B.block_view(0, 0); // Will hold the current off-diagonal block (R from QR)
    auto z00 = Z.block_view(0, 0);  // Will hold the diagonal correction to T(i,i)
    auto w = W.block_view(0);       // Work block
    auto y = W.block_view(1);       // Work block
    bool breakdown = false;

    // We follow the suggestions of Paige (1971, 1980) and compute the coefficients of
    // the (block) tridiagonal matrix T in the following order:
    // 1. Apply the operator: v_{i+1} = A v_i
    // 2. Subtract the previous basis vector scaled by the previous off-diagonal block
    //    v_{i+1} -= v_{i-1} * beta
    // 3. Compute the diagonal block
    //    alpha = <v_i, v_{i+1}>
    // 4. Subtract the current basis vector scaled by the diagonal block
    //    v_{i+1} -= v_i * alpha
    // To ensure orthogonality, we then perform a full orthogonalisation step against all
    // previous basis vectors. This gives us the final v_{i+1} and also the value of beta
    // (the off-diagonal block) for the next iteration.
    for (std::size_t i = k; i < m; ++i) {
      // logger::trace_all("Lanczos extend step, k = {}, m = {}, iteration {}", k, m, i);
      auto vi = V.block_view(i);
      auto vi1 = V.block_view(i + 1);

      if (breakdown) {
        vi.set_random();
        orth->orthonormalise_block_against_previous(V, i, nullptr, &beta);
        breakdown = false;
      }

      // Step 1: v_{i+1} = A v_i
      evp->apply_block(i, V);

      // Step 2: v_{i+1} -= v_i * T(i,i-1)
      if (i > 0) {
        auto vprev = V.block_view(i - 1);
        vprev.mult_transpose(beta, w);
        vi1 -= w;
      }

      // Step 3: Compute T(i,i) = <v_i, v_{i+1}>
      auto Tii = T.block_view(i, i);
      evp->get_inner_product()->dot(vi, vi1, Tii);

      // Step 4: Compute v_{i+1} -= v_i * T(i,i)
      vi.mult(Tii, w);
      vi1 -= w;

      // Orthonormalise the current block
      orth->call_muscle(vi1, vi1, beta);

      // In exact arithmetic, the above would be sufficient to maintain orthogonality.
      // However, due to numerical errors, we need to reorthogonalise.
      {
        // Compute the block-wise inner product between v_{i+1} and V_i
        const auto needs_orthogonalisation = [&]() {
          Real max_ortho_error = 0;
          for (std::size_t j = 0; j <= i; ++j) {
            auto vj = V.block_view(j);
            evp->get_inner_product()->dot(vj, vi1, z00);
            auto error = z00.frobenius_norm();
            if (error > max_ortho_error) max_ortho_error = error;
          }
          // logger::trace_all("Block Lanczos: orthogonalisation error {}, tolerance {}", max_ortho_error, eps * beta.frobenius_norm());
          return max_ortho_error > eps * beta.frobenius_norm();
        };

        int count = 0;
        while (count++ < 5 and needs_orthogonalisation()) {
          // logger::trace_all("Block Lanczos: orthogonalisation loop {}", count - 1);

          if (beta.frobenius_norm() < beta_threshold) {
            breakdown = true;
            std::cout << "########################################\n";
            std::cout << "BREAKDOWN DETECTED\n";
            std::cout << "########################################\n";
            break;
          }

          // Perform a block modified Gram Schmidt step
          w = vi1;
          for (std::size_t j = 0; j <= i; ++j) {
            auto vj = V.block_view(j);
            evp->get_inner_product()->dot(vj, vi1, z00);
            vj.mult(z00, y);
            w -= y;

            if (j == i - 1) {
              auto Ti_i1 = T.block_view(i, i - 1);
              auto Ti1_i = T.block_view(i - 1, i);
              Ti_i1 += z00;

              Ti1_i = Ti_i1;
              Ti1_i.transpose();
            }
            else if (j == i) {
              Tii += z00;
            }
          }

          z00 = beta;
          orth->call_muscle(w, vi1, beta);
          beta *= z00;
        }
      }
      /* z00.set_zero();
      // orth->orthonormalise_block_against_previous(V, i + 1, &z00, &beta);
      // orth->call_muscle(vi1, vi1, beta);
      orth->orthonormalise_block_against_previous(V, i + 1, nullptr, &beta);
      // Tii += z00;

      // Step 5: Write the new off-diagonal blocks T(i+1,i) and T(i,i+1)
      std::cout << "m_beta = " << beta.frobenius_norm() << "\n";
      if (beta.frobenius_norm() < beta_threshold) {
        logger::warn("BlockLanczos: breakdown detected at step {} (||beta||_F = {})", i, beta.frobenius_norm());
        // break; // Stop extension upon breakdown
      } */

      // Put the computed beta into T
      if (i + 1 < T.block_rows()) {
        auto Ti1_i = T.block_view(i + 1, i);
        Ti1_i = beta;

        auto Ti_i1 = T.block_view(i, i + 1);
        Ti_i1 = beta;
        Ti_i1.transpose();
      }
    }
  }

  BlockMatrix& get_T_matrix() { return T; }
  BMV& get_basis() { return V; }
  typename BMV::BlockMatrixBlockView get_beta() { return B.block_view(0, 0); }

private:  
  std::shared_ptr<EVP> evp;
  std::shared_ptr<Ortho> orth;

  BMV V;         // Current basis vectors
  BMV W;         // Work block (two blocks wide)
  BlockMatrix T; // Block tridiagonal matrix
  BlockMatrix B; // 1x1 block matrix that stores the current residual "norm"
  BlockMatrix Z; // 1x1 work matrix for diagonal correction
};

/** @brief Assembles the block-tridiagonal matrix T from Lanczos coefficients.
 *
 * This function constructs the dense, column-major matrix T from the alpha (diagonal) and beta (off-diagonal)
 * coefficient blocks generated by the Lanczos process. The resulting matrix T is symmetric
 * and block-tridiagonal.
 *
 * The block structure of T is:
 * T = [A_0  B_0^T  0    ...  ]
 *     [B_0  A_1    B_1^T ...  ]
 *     [0    B_1    A_2   ...  ]
 *     [...  ...    ...   ...  ]
 *
 * @param alpha_coeffs A vector of DenseMatrixBlockView objects for the diagonal blocks (A_k).
 * @param beta_coeffs  A vector of DenseMatrixBlockView objects for the sub-diagonal blocks (B_k).
 * @param T_data       The output vector where the column-major matrix data will be stored. If T is not correctly sized,
 *                     the function throws an exception. The correct size is determined by the number of blocks in the
 *                     \p alpha_coeffs and \p beta_coeffs arrays and the size of the individual blocks.
 * @param k_start      The block index to start from. If, for example, k = 2, then the entries corresponding to A_0, B_0,
 *                     B_0^T, A_1, B_1, and B_1^T will remain unchanged. The entries corresponding to A_2, B_2, B_2^T etc.
 *                     will be overwritten. By default, this value is zero, so all blocks will be overwritten.
 */
template <class MatBlockView>
void build_tridiagonal_matrix(const std::vector<MatBlockView>& alpha_coeffs, const std::vector<MatBlockView>& beta_coeffs, std::vector<typename MatBlockView::Real>& T_data, std::size_t k_start = 0)
{
  const std::size_t m = alpha_coeffs.size();
  const std::size_t blocksize = alpha_coeffs[0].rows();
  const std::size_t dim = m * blocksize;

  if (T_data.size() != dim * dim) throw std::invalid_argument("T has incorrect dimension, got " + std::to_string(T_data.size()) + ", expected " + std::to_string(dim * dim));

  // Place alpha blocks on the diagonal
  for (std::size_t k = k_start; k < m; ++k) {
    const auto& alpha_k = alpha_coeffs[k];
    for (std::size_t j = 0; j < blocksize; ++j) {   // column in block
      for (std::size_t i = 0; i < blocksize; ++i) { // row in block
        const std::size_t T_row = k * blocksize + i;
        const std::size_t T_col = k * blocksize + j;
        T_data[T_col * dim + T_row] = alpha_k(i, j);
      }
    }
  }

  // Place beta blocks on the super- and sub-diagonals
  if (m > 1) {
    for (std::size_t k = k_start; k < m - 1; ++k) {
      const auto& beta_k = beta_coeffs[k];
      for (std::size_t j = 0; j < blocksize; ++j) {   // column in block
        for (std::size_t i = 0; i < blocksize; ++i) { // row in block
          // Sub-diagonal block B_k at block position (k+1, k)
          std::size_t T_row_sub = (k + 1) * blocksize + i;
          std::size_t T_col_sub = k * blocksize + j;
          T_data[T_col_sub * dim + T_row_sub] = beta_k(i, j);

          // Super-diagonal block B_k^T at block position (k, k+1)
          std::size_t T_row_super = k * blocksize + i;
          std::size_t T_col_super = (k + 1) * blocksize + j;
          T_data[T_col_super * dim + T_row_super] = beta_k(j, i); // Transpose
        }
      }
    }
  }
}
