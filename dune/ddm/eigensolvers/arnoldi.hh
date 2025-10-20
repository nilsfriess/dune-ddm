#pragma once

#include "dune/ddm/eigensolvers/blockmatrix.hh"
#include "dune/ddm/eigensolvers/concepts.hh"
#include "dune/ddm/eigensolvers/eigensolver_params.hh"
#include "dune/ddm/eigensolvers/inner_products.hh"
#include "dune/ddm/eigensolvers/orthogonalisation.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/logger.hh"

#include <cstddef>
#include <format>
#include <ios>

struct NoCallback {
  template <class T>
  void operator()(T&, std::size_t) const noexcept
  {
  }
};

template <Eigenproblem EVP, class Callback = NoCallback>
class BlockLanczos {
  using BMV = typename EVP::BlockMultiVec;
  using BlockMatrix = typename EVP::BlockMatrix;
  using BlockMatrixBlockView = typename EVP::BlockMatrixBlockView;
  using Real = typename EVP::Real;
  using InnerProduct = typename EVP::InnerProduct;
  using Ortho = orthogonalisation::BlockOrthogonalisation<InnerProduct>;
  static constexpr std::size_t blocksize = EVP::blocksize;
  static constexpr Real eps = std::numeric_limits<Real>::epsilon();
  static constexpr Real near_0 = EVP::blocksize * std::numeric_limits<Real>::min() * Real(10);

public:
  BlockLanczos(std::shared_ptr<EVP> evp, std::shared_ptr<Ortho> orth, const EigensolverParams& params, Callback&& cb = Callback{})
      : evp(std::move(evp))
      , orth(std::move(orth))
      , V(this->evp->mat_size(), params.ncv + blocksize)
      , W(this->evp->mat_size(), params.ncv)
      , F(this->evp->mat_size(), blocksize)
      , T(params.ncv / blocksize)
      , B(2)
      , Z(1)
      , nev{params.nev}
      , ncv{params.ncv}
      , tolerance{params.tolerance}
      , cb(std::move(cb))
  {
    // Initialise V
    V.set_random();
    this->evp->apply_block(0, V, V); // Make sure V is in the range of the operator
    this->cb(V, 0);
    this->orth->orthonormalise_block_against_previous(V, 0);

    // Compute the first step of the Lanczos decomposition
    auto v0 = V.block_view(0);
    auto v1 = V.block_view(1);
    auto w0 = W.block_view(0);
    auto t00 = T.block_view(0, 0);
    auto beta1 = B.block_view(0, 0);
    auto z00 = Z.block_view(0, 0);

    // V{1} <- A V{0}
    this->evp->apply_block(0, V);
    this->cb(V, 1);
    this->orth->orthonormalise_block_against_previous(V, 1, &t00, &beta1);

    if (beta1.frobenius_norm() < eps) TODO("Initial beta norm is too small, likely because the initial vectors are eigenvectors already");
    else logger::debug("BlockLanczos: Initial beta norm is {}", beta1.frobenius_norm());
  }

  //  Extend a step-k Lanczos decomposition to a step-m Lanczos decomposition
  void extend(std::size_t k, std::size_t m)
  {
    const auto beta_threshold = eps * evp->mat_size();

    auto beta1 = B.block_view(0, 0);
    auto z00 = Z.block_view(0, 0);
    auto w = W.block_view(0);

    bool restart = false;
    for (std::size_t i = k; i < m; ++i) {
      auto vi = V.block_view(i);
      auto vi1 = V.block_view(i + 1);

      auto beta_norm = beta1.frobenius_norm();

      // The following criteria are taken from Spectra (adapted to the block case)
      restart = restart or (beta_norm < near_0); // If restart was set to true in the previous iteration, keep it
      if (!restart) {
        if (beta_norm < std::sqrt(eps)) {
          auto vprev = V.block_view(i - 1);
          evp->get_inner_product()->dot(vprev, vi, z00);
          if (z00.frobenius_norm() > std::sqrt(eps)) restart = true;
        }
      }

      if (restart) {
        // Here we should create a new random block
        TODO("BlockLanczos: Need to create a new random block");
      }

      auto Ti_i1 = T.block_view(i, i - 1);
      auto Ti1_i = T.block_view(i - 1, i);
      if (!restart) {
        Ti_i1 = beta1;
        Ti1_i = beta1;
        Ti1_i.transpose();
      }
      else {
        Ti_i1.set_zero();
        Ti1_i.set_zero();
      }

      // V{i+1} = A V{i}
      evp->apply_block(i, V);
      cb(V, i + 1); // Call the user-provided callback, or do nothing by default

      // Now we need to do
      //     V{i+1} <- V{i+1} - V V^T B V{i+1}
      // to orthogonalise V{i+1} w.r.t. all previous Lanczos vectors. In exact arithmetic the
      // orthogonalisation against the V{0}, ..., V{i-2} is automatic, i.e.
      //     V{i+1} <- V{i+1} - V{i} * T{i, i-1} - V{i-1} * T{i, i}
      // would be enough. In finite precision, this is no longer true; therefore, we orthogonalise
      // against all previous basis vectors. However, as advocated by Paige (1972, 1980), see also
      // Cullum and Willoughby (2002), for improved stability we should first compute
      //     V{i+1} <- V{i+1} - V{i} * T{i, i-1} and use this value to compute T{i,i}. The projection
      // coefficient that will be computed during the full orthogonalisation below then has to be added
      // to this value of T{i,i}.
      if (!restart) {
        // We have T{i+1, i} != 0 only if we're not restarting.
        auto vi_1 = V.block_view(i - 1);
        vi_1.mult(Ti_i1, w);
        vi1 -= w;
      }

      // Now compute the next diagonal block, T{i,i} = <V{i} , V{i+1}> = <V{i} , A V{i}>
      auto Tii = T.block_view(i, i);
      evp->get_inner_product()->dot(vi, vi1, Tii);

      // V{i+1} <- V{i+1} - V{i+1} * T{i,i}
      vi.mult(Tii, w);
      vi1 -= w;

      orth->orthonormalise_block_against_previous(V, i + 1, &z00, &beta1);
      Tii += z00; // Add accumulated projection coefficients to diagonal
      auto beta2 = B.block_view(0, 1);
      orth->orthonormalise_block_against_previous(V, i + 1, &z00, &beta2);
      Tii += z00; // Add accumulated projection coefficients to diagonal

      if (beta1.frobenius_norm() < beta_threshold) {
        logger::warn("Lanczos: beta norm is too small at step {}, trying to restart", i);
        restart = true;
        continue;
      }
      restart = false;
    }

    // T.print(false, false, 10, 0);
  }

  BlockMatrix& get_T_matrix() { return T; }
  BMV& get_basis() { return V; }
  typename BMV::BlockMatrixBlockView get_beta() { return B.block_view(0, 0); }

private:
  std::shared_ptr<EVP> evp;
  std::shared_ptr<Ortho> orth;

  BMV V;         // Current basis vectors
  BMV W;         // Work vector. TODO: We can probably get rid of W, and use the next column of V as temporary storage
  BMV F;         // Residual vector
  BlockMatrix T; // Block tridiagonal matrix
  BlockMatrix B; // 1x1 block matrix that stores the current residual "norm"
  BlockMatrix Z; // 1x1 work matrix

  // Eigensolver params
  std::size_t nev;
  std::size_t ncv;
  double tolerance;

  // Optional callback that will be called after each extension step
  Callback cb;
};

/** @brief Performs one step of block Lanczos extension
 *
 * This function extends the Krylov subspace from span{Q_0, ..., Q_k} to span{Q_0, ..., Q_{k+1}}
 * by computing Q_{k+1} = A * Q_k and then orthogonalising it against all previous blocks.
 *
 * The coefficients computed during orthogonalisation can be used to build the
 * tridiagonal Lanczos matrix T.
 */
template <Eigenproblem EVP>
bool lanczos_extend_step(EVP& evp, typename EVP::BlockMultiVec& Q, std::size_t k, orthogonalisation::BlockOrthogonalisation<typename EVP::InnerProduct>& ortho,
                         typename EVP::BlockMultiVec::DenseBlockMatrixBlockView* alpha = nullptr, typename EVP::BlockMultiVec::DenseBlockMatrixBlockView* beta = nullptr)
{
  // Check bounds
  if (k + 1 >= Q.blocks()) throw std::invalid_argument("Cannot extend to block with index " + std::to_string(k + 1) + " for blockvector with " + std::to_string(Q.blocks()) + " blocks");

  logger::debug_all("lanczos_extend_step() with k = {}", k);

  // Step 1: Compute Q_{k+1} = A * Q_k (this is the candidate before orthogonalisation)
  evp.apply_block(k, Q);

  auto Qk = Q.block_view(k);
  auto Qk1 = Q.block_view(k + 1);
  assert(alpha);
  evp.get_inner_product()->dot(Qk, Qk1, *alpha);

  // Debug: Check for problematic alpha values
  auto alpha_norm = alpha->frobenius_norm();
  logger::debug_all("lanczos_extend_step k={}: alpha norm = {}", k, alpha_norm);

  // Check for numerical issues with alpha coefficients
  if (alpha_norm < 1e-12) {
    logger::warn_all("lanczos_extend_step k={}: Extremely small alpha coefficient (norm={})", k, alpha_norm);
    logger::warn_all("This may indicate numerical breakdown or near-singularity");
  }

  // Check for very large alpha coefficients too
  if (alpha_norm > 1e6) {
    logger::warn_all("lanczos_extend_step k={}: Very large alpha coefficient (norm={})", k, alpha_norm);
    logger::warn_all("This may indicate numerical scaling issues");
  }

  alpha->print();

  // Step 2: Orthogonalise Q_{k+1} against all previous blocks Q_0, ..., Q_k
  // This will extract the coefficients needed for the T matrix
  // Note: Pass nullptr for alpha to prevent overwriting our computed symmetric alpha
  ortho.orthonormalise_block_against_previous(Q, k + 1, nullptr, beta);

  // Debug: Print orthognality of the new block w.r.t. to the previous two blocks
  {
    auto ip = evp.get_inner_product();
    typename EVP::BlockMultiVec::BlockMatrix R(2);
    auto r0 = R.block_view(0, 0);
    auto r1 = R.block_view(1, 0);

    ip->dot(Q.block_view(k), Q.block_view(k + 1), r0);
    logger::debug_all("Inner product of new block Q_{} against previous block Q_{}:", k + 1, k);
    std::cout << std::scientific << std::setprecision(18) << r0(0, 0) << "\n";
    if (k > 0) {
      ip->dot(Q.block_view(k - 1), Q.block_view(k + 1), r1);
      logger::debug_all("Inner product of new block Q_{} against previous block Q_{}:", k + 1, k - 1);
      std::cout << std::scientific << std::setprecision(18) << r1(0, 0) << "\n";
    }
  }

  // Step 3: Check for breakdown (linearly dependent vectors)
  // This would be indicated by a very small beta coefficient
  if (beta) {
    // Use size-dependent breakdown tolerance like Spectra
    const typename EVP::Real machine_eps = std::numeric_limits<typename EVP::Real>::epsilon();
    const typename EVP::Real breakdown_tol = std::max(static_cast<typename EVP::Real>(1e-12), machine_eps * std::sqrt(static_cast<typename EVP::Real>(evp.mat_size())));
    auto beta_norm = beta->frobenius_norm();
    logger::debug_all("lanczos_extend_step k={}: beta norm = {}", k, beta_norm);

    // Check for numerical breakdown patterns
    bool breakdown_detected = false;

    // Primary breakdown criterion: beta too small
    if (beta_norm < breakdown_tol) {
      logger::error_all("PRIMARY BREAKDOWN at step k={}: beta norm = {} < {}", k, beta_norm, breakdown_tol);
      breakdown_detected = true;
    }

    // Secondary breakdown criterion: alpha much larger than beta (scaling issue)
    if (alpha_norm > 0 && beta_norm > 0) {
      auto alpha_beta_ratio = alpha_norm / beta_norm;
      if (alpha_beta_ratio > 1e8) {
        logger::error_all("SCALING BREAKDOWN at step k={}: alpha/beta ratio = {} is too large", k, alpha_beta_ratio);
        breakdown_detected = true;
      }
    }

    // Tertiary breakdown criterion: alpha coefficients becoming too small
    if (alpha_norm < 1e-10 && k > 2) {
      logger::error_all("ALPHA BREAKDOWN at step k={}: alpha norm = {} is too small", k, alpha_norm);
      breakdown_detected = true;
    }

    if (breakdown_detected) {
      logger::error_all("NUMERICAL BREAKDOWN DETECTED - Lanczos process is unreliable from step k={}", k);
      logger::error_all("Consider using restart, deflation, or different numerical parameters");
    }
  }

  return true; // Success
}

/** @brief Creates an m-step block Lanczos decomposition
 *
 * The first \p k blocks are not modified.
 *
 */
// clang-format off
template <Eigenproblem EVP>
bool lanczos_extend_decomposition(EVP& evp,
                                  typename EVP::BlockMultiVec& Q,
                                  std::size_t k,
                                  std::size_t m,
                                  orthogonalisation::BlockOrthogonalisation<typename EVP::InnerProduct>& ortho,
                                  std::vector<typename EVP::BlockMultiVec::DenseBlockMatrixBlockView>& alpha_coeffs,
                                  std::vector<typename EVP::BlockMultiVec::DenseBlockMatrixBlockView>& beta_coeffs,
                                  typename EVP::BlockMultiVec::DenseBlockMatrixBlockView* final_beta = nullptr)
// clang-format on
{
  if (k >= Q.blocks()) throw std::invalid_argument("Value of k " + std::to_string(k) + " is too large, must be smaller than number of blocks " + std::to_string(Q.blocks()));
  if (m >= Q.blocks()) throw std::invalid_argument("Value of m " + std::to_string(m) + " is too large, must be smaller than number of blocks " + std::to_string(Q.blocks()));
  if (m <= k) throw std::invalid_argument("Value of m " + std::to_string(m) + " must be larger than value of k " + std::to_string(k));

  // Coefficient storage should be sized for the total decomposition
  if (alpha_coeffs.size() < m || beta_coeffs.size() < m) throw std::invalid_argument("alpha or beta not large enough");

  for (std::size_t block = k; block < m; ++block) {
    auto* current_alpha = &alpha_coeffs[block];
    auto* current_beta = &beta_coeffs[block];

    bool success = lanczos_extend_step(evp, Q, block, ortho, current_alpha, current_beta);

    if (!success) {
      // Breakdown occurred, but the decomposition is valid up to step k
      // The final beta is the one that was just computed and caused the breakdown
      if (final_beta) *final_beta = *current_beta;
      throw std::runtime_error("Breakdown");
    }
  }

  // If successful, the final beta is the last one we computed
  if (final_beta && !beta_coeffs.empty()) *final_beta = beta_coeffs.back();

  return true;
}

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

  if (T_data.size() != dim * dim) throw std::invalid_argument(std::format("T has incorrect dimension, got {}, expected {}", T_data.size(), dim * dim));

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
