#pragma once

#include "blockmatrix.hh"
#include "blockmultivector.hh"
#include "dune/ddm/eigensolvers/inner_products.hh"
#include "dune/ddm/eigensolvers/umfpack.hh"
#include "dune/ddm/logger.hh"
#include "orthogonalisation.hh"

// TODO: Many of the methods probably don't belong here.
template <class Mat, std::size_t blocksize_ = 8>
class ShiftInvertEigenproblem {
public:
  using Real = typename Mat::field_type;
  using BlockMultiVec = BlockMultiVector<Real, blocksize_>;
  using BlockMultiVecView = typename BlockMultiVec::BlockViewType;
  using BlockMatrix = typename BlockMultiVec::BlockMatrix;
  using BlockMatrixBlockView = typename BlockMultiVec::BlockMatrixBlockView;
  using InnerProduct = orthogonalisation::MatrixInnerProduct<Mat, BlockMultiVec>;
  static constexpr std::size_t blocksize = blocksize_;

  ShiftInvertEigenproblem(const Mat& A, std::shared_ptr<Mat> B, double shift)
      //    : A(A)
      : B(std::move(B))
      , shift_(shift)
      , AshiftB(build_shifted(A, *(this->B), shift))
      , inner_prod(std::make_unique<InnerProduct>(this->B))
      , solver(std::make_unique<UMFPackMultivecSolver<Mat>>(AshiftB))
  {
  }

  ShiftInvertEigenproblem(const ShiftInvertEigenproblem&) = delete;
  ShiftInvertEigenproblem(const ShiftInvertEigenproblem&&) = delete;
  ShiftInvertEigenproblem& operator=(const ShiftInvertEigenproblem&) = delete;
  ShiftInvertEigenproblem& operator=(const ShiftInvertEigenproblem&&) = delete;
  ~ShiftInvertEigenproblem() = default;

  bool is_symmetric() const
  {
    return true; // TODO: Support unsymmetric problems
  }

  void apply(const BlockMultiVec& Xin, BlockMultiVec& Yout)
  {
    if (!W or W->rows() != Xin.rows() or W->cols() != Xin.cols()) W = std::make_unique<BlockMultiVec>(Xin.rows(), Xin.cols());
    Xin.apply_to_mat(*B, *W);
    solver->apply(Yout, *W);
  }

  void apply_block(std::size_t block, const BlockMultiVec& Xin, BlockMultiVec& Yout)
  {
    if (!W or W->rows() != Xin.rows() or W->cols() != Xin.cols()) W = std::make_unique<BlockMultiVec>(Xin.rows(), Xin.cols());

    auto Wv = W->block_view(block);
    auto Xv = Xin.block_view(block);
    Xv.apply_to_mat(*B, Wv);
    solver->apply(Yout, *W, block, block + 1);
  }

  /** @brief Applies the operator to block \p block of \p Xin and stores the result in the next block of \p Xin */
  void apply_block(std::size_t block, BlockMultiVec& Xin)
  {
    if (!W or W->cols() != Xin.rows() or W->rows() != Xin.rows()) W = std::make_unique<BlockMultiVec>(Xin.rows(), Xin.cols());

    auto Wv = W->block_view(block + 1);
    auto Xv = Xin.block_view(block);
    Xv.apply_to_mat(*B, Wv);
    solver->apply(Xin, *W, block + 1, block + 2);
  }

  bool compare_untransformed(Real a, Real b) { return std::abs(transform_eigenvalue(a)) < std::abs(transform_eigenvalue(b)); }

  // void applyA(const BlockMultiVec& Xin, BlockMultiVec& Yout) { Xin.apply_to_mat(A, Yout); }

  // void dot(const BlockMultiVec& X, const BlockMultiVec& Y, BlockMatrix& result)
  // {
  //   // Compute result = X^T B Y (B-inner product)
  //   // TODO: Handle case where X and Y have different sizes than w
  //   if (!W) W = std::make_unique<BlockMultiVec>(X.rows(), X.cols());
  //   X.apply_to_mat(B, *W); // w = B * X
  //   W->dot(Y, result);     // result = w^T Y = X^T B Y
  // }

  // void orthonormalise(const BlockMultiVec& Xin, BlockMultiVec& Yout)
  // {
  //   block_modified_gram_schmidt(B, Xin, Yout, true, true);
  //   // Debug: check B-orthonormality of Q
  //   if (logger::get_level() <= logger::Level::debug) log_B_orthonormality(Yout, "Q");
  // }

  // void orthonormalise(BlockMultiVec& X)
  // {
  //   if (!W) W = std::make_unique<BlockMultiVec>(X.rows(), X.cols());
  //   block_modified_gram_schmidt(B, X, *W);
  //   std::swap(*W, X);
  //   // Debug: check B-orthonormality of Q (in-place)
  //   if (logger::get_level() <= logger::Level::debug) log_B_orthonormality(X, "Q");
  // }

  // template <class DenseView>
  // void orthonormalise_block(std::size_t block, BlockMultiVec& X, DenseView R)
  // {
  //   if (!W) W = std::make_unique<BlockMultiVec>(X.rows(), X.cols());
  //   block_modified_gram_schmidt_single_block(block, B, X, *W, R);
  //   std::swap(*W, X);
  // }

  void block_column_norms(const BlockMultiVec& X, std::size_t block, std::vector<Real>& norms)
  {
    if (!W) W = std::make_unique<BlockMultiVec>(X.rows(), X.cols());
    if (!T) T = std::make_unique<BlockMatrix>(X.cols());

    auto Xb = X.block_view(block);
    *W = X;
    auto Wb = W->block_view(block);
    auto Tb = T->block_view(block, block);
    Xb.apply_to_mat(*B, Wb); // Wb = B * Xb
    Xb.dot(Wb, Tb);
    assert(norms.size() == blocksize);
    for (std::size_t i = 0; i < blocksize; ++i) norms[i] = std::sqrt(Tb(i, i));
  }

  // Check B-orthonormality of a multivector X by forming G = X^T B X and
  // logging compact diagnostics: max off-block magnitude, max off-diagonal
  // within diagonal blocks, max diagonal deviation from 1, and ||G - I||_F.
  // void log_B_orthonormality(const BlockMultiVec& X, const std::string& tag = "X")
  // {
  //   BlockMatrix G(X.blocks(), blocksize);
  //   // G = X^T B X
  //   dot(X, X, G);

  //   Real max_offblock = 0.0;
  //   Real max_offdiag = 0.0;
  //   Real max_diag_dev = 0.0;
  //   long double fro_err_sq = 0.0L;

  //   for (std::size_t bi = 0; bi < X.blocks(); ++bi) {
  //     for (std::size_t bj = 0; bj < X.blocks(); ++bj) {
  //       auto Bij = G.block_view(bi, bj);
  //       for (std::size_t i = 0; i < blocksize; ++i) {
  //         for (std::size_t j = 0; j < blocksize; ++j) {
  //           Real val = Bij(i, j);
  //           if (bi == bj) {
  //             if (i == j) {
  //               Real dev = std::abs(val - Real(1));
  //               if (dev > max_diag_dev) max_diag_dev = dev;
  //               long double d = static_cast<long double>(val - Real(1));
  //               fro_err_sq += d * d;
  //             }
  //             else {
  //               Real absv = std::abs(val);
  //               if (absv > max_offdiag) max_offdiag = absv;
  //               long double d = static_cast<long double>(val);
  //               fro_err_sq += d * d;
  //             }
  //           }
  //           else {
  //             Real absv = std::abs(val);
  //             if (absv > max_offblock) max_offblock = absv;
  //             long double d = static_cast<long double>(val);
  //             fro_err_sq += d * d;
  //           }
  //         }
  //       }
  //     }
  //   }

  //   double fro_err = std::sqrt(static_cast<double>(fro_err_sq));
  //   logger::debug("B-orthonormality[{}]: blocks={}, blocksize={}, max|off-block|={}, max|offdiag|={}, max|diag-1|={}, ||G-I||_F={}", tag, X.blocks(), X.blocksize(), max_offblock, max_offdiag,
  //                 max_diag_dev, fro_err);
  // }

  // // Compute and log true GEVP residuals for current vectors X and Rayleigh-Ritz values theta
  // // Residual per column i: r_i = A v_i - lambda_i B v_i, lambda_i = shift + 1/theta_i
  // // Report normalized by |lambda_i| when |lambda_i|>0, else unnormalized ||r_i||_2
  // void log_true_residuals(const BlockMultiVec& X, const std::vector<Real>& theta)
  // {
  //   if (logger::get_level() > logger::Level::debug) return;

  //   // Convert to lambda
  //   std::vector<Real> lambda = transform_eigenvalues(theta);

  //   // Compute Av and Bv
  //   BlockMultiVec Av(X.rows(), X.cols());
  //   BlockMultiVec Bv(X.rows(), X.cols());
  //   X.apply_to_mat(A, Av);
  //   X.apply_to_mat(B, Bv);
  //   Bv.scale_columns(lambda); // Bv <- Bv * lambda
  //   Av -= Bv;                 // Av <- Av - Bv = A v - lambda B v

  //   std::vector<Real> normalisations(X.cols());
  //   for (std::size_t i = 0; i < X.cols(); ++i) normalisations[i] = lambda[i] != 0.0 ? std::abs(lambda[i]) : 1.0;
  //   Av.scale_columns(normalisations);
  //   std::vector<Real> res_norms(X.cols());
  //   for (std::size_t i = 0; i < Av.blocks(); ++i) logger::debug("Residual of block {}: {}", i, Av.block_view(i).two_norm());
  // }

  // Transform eigenvalue from shift-invert space (θ) to original space (λ)
  // For shift-invert: (A - σB)^(-1)B v = θ v  =>  A v = λ B v  where λ = σ + 1/θ
  Real transform_eigenvalue(Real theta) const { return shift_ + 1.0 / theta; }

  // Transform a vector of eigenvalues
  std::vector<Real> transform_eigenvalues(const std::vector<Real>& theta) const
  {
    std::vector<Real> lambda(theta.size());
    for (std::size_t i = 0; i < theta.size(); ++i) lambda[i] = transform_eigenvalue(theta[i]);
    return lambda;
  }

  std::size_t mat_size() const noexcept { return AshiftB.N(); }

  std::shared_ptr<InnerProduct> get_inner_product() { return inner_prod; }

private:
  static Mat build_shifted(const Mat& A_in, const Mat& B_in, double shift)
  {
    Mat M = A_in;
    M.axpy(-shift, B_in);
    return M;
  }

  // const Mat& A;
  std::shared_ptr<Mat> B;
  double shift_;
  Mat AshiftB;

  std::shared_ptr<InnerProduct> inner_prod;

  std::unique_ptr<UMFPackMultivecSolver<Mat>> solver;
  std::unique_ptr<BlockMultiVec> W{nullptr}; // work multivector
  std::unique_ptr<BlockMatrix> T{nullptr};   // work multivector
};
