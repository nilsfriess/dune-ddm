#pragma once

#include "blockmatrix.hh"
#include "blockmultivector.hh"
#include "dune/ddm/eigensolvers/inner_products.hh"
#include "dune/ddm/eigensolvers/umfpack.hh"
#include "dune/ddm/logger.hh"
#include "orthogonalisation.hh"

struct NoCallback {
  template <class T>
  void operator()(T&, std::size_t) const noexcept
  {
  }
};

// TODO: Many of the methods probably don't belong here.
template <class Mat, std::size_t blocksize_ = 8, class Callback = NoCallback>
class ShiftInvertEigenproblem {
public:
  using Real = typename Mat::field_type;
  using BlockMultiVec = BlockMultiVector<Real, blocksize_>;
  using BlockMultiVecView = typename BlockMultiVec::BlockViewType;
  using BlockMatrix = typename BlockMultiVec::BlockMatrix;
  using BlockMatrixBlockView = typename BlockMultiVec::BlockMatrixBlockView;
  using InnerProduct = orthogonalisation::MatrixInnerProduct<Mat, BlockMultiVec>;
  static constexpr std::size_t blocksize = blocksize_;

  ShiftInvertEigenproblem(const Mat& A, std::shared_ptr<Mat> B, double shift, Callback&& cb = Callback{})
      : B(std::move(B))
      , shift_(shift)
      , AshiftB(build_shifted(A, *(this->B), shift))
      , inner_prod(std::make_unique<InnerProduct>(this->B))
      , solver(std::make_unique<UMFPackMultivecSolver<Mat>>(AshiftB))
      , cb(std::move(cb))
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
    TODO("Not implemented");
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
    cb(Yout, block);
  }

  /** @brief Applies the operator to block \p block of \p Xin and stores the result in the next block of \p Xin */
  void apply_block(std::size_t block, BlockMultiVec& Xin)
  {
    if (!W or W->cols() != Xin.rows() or W->rows() != Xin.rows()) W = std::make_unique<BlockMultiVec>(Xin.rows(), Xin.cols());

    auto Wv = W->block_view(block + 1);
    auto Xv = Xin.block_view(block);
    Xv.apply_to_mat(*B, Wv);
    solver->apply(Xin, *W, block + 1, block + 2);
    cb(Xin, block + 1);
  }

  bool compare_untransformed(Real a, Real b) { return std::abs(transform_eigenvalue(a)) < std::abs(transform_eigenvalue(b)); }

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

  // Optional callback that will be after applying the operator
  Callback cb;
};
