#pragma once

#include "blockmatrix.hh"
#include "blockmultivector.hh"
#include "dune/ddm/eigensolvers/inner_products.hh"
#include "orthogonalisation.hh"

/** @brief Standard eigenvalue problem Ax = λx
 *
 * This class implements the operator interface for a standard eigenvalue problem,
 * where we seek eigenvalues λ and eigenvectors x such that Ax = λx.
 *
 * For the standard problem, the operator application is simply y = Ax, and
 * the natural inner product is the Euclidean inner product <x,y> = x^T * y.
 *
 * @tparam Mat The matrix type (e.g., Dune::BCRSMatrix)
 * @tparam blocksize_ The block size for block multivectors
 */
template <class Mat, std::size_t blocksize_ = 8>
class StandardEigenproblem {
public:
  using Real = typename Mat::field_type;
  using BlockMultiVec = BlockMultiVector<Real, blocksize_>;
  using BlockMultiVecView = typename BlockMultiVec::BlockViewType;
  using BlockMatrix = typename BlockMultiVec::BlockMatrix;
  using BlockMatrixBlockView = typename BlockMultiVec::BlockMatrixBlockView;
  using InnerProduct = orthogonalisation::EuclideanInnerProduct<BlockMultiVec>;
  static constexpr std::size_t blocksize = blocksize_;

  /** @brief Constructor for standard eigenvalue problem
   *
   * @param A The matrix defining the eigenvalue problem Ax = λx
   */
  explicit StandardEigenproblem(std::shared_ptr<Mat> A)
      : A_(std::move(A))
      , inner_prod_(std::make_shared<InnerProduct>())
  {
  }

  // Delete copy/move operations for safety
  StandardEigenproblem(const StandardEigenproblem&) = delete;
  StandardEigenproblem(StandardEigenproblem&&) = delete;
  StandardEigenproblem& operator=(const StandardEigenproblem&) = delete;
  StandardEigenproblem& operator=(StandardEigenproblem&&) = delete;
  ~StandardEigenproblem() = default;

  /** @brief Returns true since the standard problem is symmetric if A is symmetric */
  bool is_symmetric() const
  {
    return true; // TODO: Could check matrix structure if needed
  }

  /** @brief Apply the operator: y = Ax
   *
   * For the standard eigenvalue problem, this simply applies the matrix A.
   * Works with both full BlockMultiVec and BlockView.
   */
  template <class VecIn, class VecOut>
  void apply(const VecIn& Xin, VecOut& Yout)
  {
    Xin.apply_to_mat(*A_, Yout);
  }

  /** @brief Apply the operator to a single block
   *
   * @param block The block index to apply
   * @param Xin Input block multivector
   * @param Yout Output block multivector (result written to the specified block)
   */
  void apply_block(std::size_t block, const BlockMultiVec& Xin, BlockMultiVec& Yout)
  {
    auto Xv = Xin.block_view(block);
    auto Yv = Yout.block_view(block);
    Xv.apply_to_mat(*A_, Yv);
  }

  void apply_block(std::size_t block, BlockMultiVec& X)
  {
    auto Xk = X.block_view(block);
    auto Xk1 = X.block_view(block + 1);
    Xk.apply_to_mat(*A_, Xk1);
  }

  /** @brief Compute block column norms using the Euclidean inner product
   *
   * For standard problems, the norm is just the 2-norm: ||x||_2 = sqrt(x^T x)
   */
  void block_column_norms(const BlockMultiVec& X, std::size_t block, std::vector<Real>& norms)
  {
    if (!T_) T_ = std::make_unique<BlockMatrix>(X.blocks());

    auto Xb = X.block_view(block);
    auto Tb = T_->block_view(block, block);

    // Compute G = X^T X for the block
    Xb.dot(Xb, Tb);

    assert(norms.size() == blocksize);
    for (std::size_t i = 0; i < blocksize; ++i) norms[i] = std::sqrt(Tb(i, i));
  }

  /** @brief Transform eigenvalue (identity for standard problem)
   *
   * For the standard problem, eigenvalues are not transformed.
   */
  Real transform_eigenvalue(Real lambda) const { return lambda; }

  /** @brief Transform a vector of eigenvalues (identity for standard problem) */
  std::vector<Real> transform_eigenvalues(const std::vector<Real>& lambda) const
  {
    return lambda; // No transformation needed
  }

  /** @brief Get the matrix size */
  std::size_t mat_size() const noexcept { return A_->N(); }

  /** @brief Get the inner product object */
  std::shared_ptr<InnerProduct> get_inner_product() { return inner_prod_; }

  /** @brief Get the matrix A (const access) */
  const Mat& get_matrix() const { return *A_; }

private:
  std::shared_ptr<Mat> A_;
  std::shared_ptr<InnerProduct> inner_prod_;
  std::unique_ptr<BlockMatrix> T_{nullptr};  // Work storage for norms
  std::unique_ptr<BlockMultiVec> W{nullptr}; // Work multivector
};
