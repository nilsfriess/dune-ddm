#pragma once

#include "dune/common/exceptions.hh"

#include <cblas-openblas.h>
#include <memory>
#include <type_traits>
#include <vector>

namespace orthogonalisation {

/** @brief Standard Euclidean inner product implementation
 *
 * This implements the standard inner product <x,y> = x^T * y
 * It's the most efficient implementation since no matrix operations are needed.
 */
template <class BlockMultiVec_>
class EuclideanInnerProduct {
public:
  using BlockMultiVec = BlockMultiVec_;
  using Real = typename BlockMultiVec::Real;
  using BlockMultiVecView = typename BlockMultiVec::BlockViewType;
  using ConstBlockMultiVecView = typename BlockMultiVec::ConstBlockViewType;
  using BlockMatrix = typename BlockMultiVec::BlockMatrix;
  using BlockMatrixBlockView = typename BlockMultiVec::BlockMatrixBlockView;

  void dot(const BlockMultiVec& x, const BlockMultiVec& y, BlockMatrix& R) const { x.dot(y, R); }

  void dot(ConstBlockMultiVecView x, ConstBlockMultiVecView y, BlockMatrixBlockView R) const { x.dot(y, R); }

  /** @brief Dot product for single vectors using Euclidean inner product */
  template <typename T>
  T dot(const std::vector<T>& x, const std::vector<T>& y) const
  {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>, "dot: only double and float are supported");

    if (x.size() != y.size()) DUNE_THROW(Dune::Exception, "dot: vectors must have the same size");

    if constexpr (std::is_same_v<T, double>) return cblas_ddot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
    else if constexpr (std::is_same_v<T, float>) return cblas_sdot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
    else static_assert(false, "dot: unsupported type - only double and float are supported");
  }

  /** @brief Returns true since this is the Euclidean inner product */
  bool is_euclidean() const { return true; }
};

/** @brief Matrix-induced inner product implementation
 *
 * This implements the inner product <x,y>_M = x^T * M * y
 * where M is a symmetric positive definite matrix.
 *
 * Common use cases:
 * - Mass matrix inner product in finite elements
 * - Weighted inner products
 */
template <class Mat, class BlockMultiVec_>
class MatrixInnerProduct {
public:
  using BlockMultiVec = BlockMultiVec_;
  using Real = typename BlockMultiVec::Real;
  using BlockMultiVecView = typename BlockMultiVec::BlockViewType;
  using ConstBlockMultiVecView = typename BlockMultiVec::ConstBlockViewType;
  using BlockMatrix = typename BlockMultiVec::BlockMatrix;
  using BlockMatrixBlockView = typename BlockMultiVec::BlockMatrixBlockView;

  explicit MatrixInnerProduct(std::shared_ptr<const Mat> M)
      : matrix_(std::move(M))
  {
    if (!matrix_) DUNE_THROW(Dune::Exception, "MatrixInnerProduct: matrix cannot be null");
  }

  void dot(const BlockMultiVec& x, const BlockMultiVec& y, BlockMatrix& R) const
  {
    if (!temp1 or x.rows() != temp1->rows() || x.blocksize != temp1->blocksize) temp1 = std::make_unique<BlockMultiVec>(x);

    y.apply_to_mat(*matrix_, *temp1);
    x.dot(*temp1, R);
  }

  void dot(ConstBlockMultiVecView x, ConstBlockMultiVecView y, BlockMatrixBlockView R) const
  {
    if (!temp2 or x.rows() != temp2->rows() || x.blocksize != temp2->blocksize) temp2 = std::make_unique<BlockMultiVec>(x.rows(), x.blocksize); // allocate a blockvector that can hold one block
    auto tv = temp2->block_view(0);

    y.apply_to_mat(*matrix_, tv);
    x.dot(tv, R);
  }

  /** @brief Matrix-based inner product for single vectors: <x,y>_M = x^T * M * y */
  template <class Vec>
  auto dot(const Vec& x, const Vec& y) const
  {
    using T = typename Vec::value_type;

    if (x.size() != y.size()) DUNE_THROW(Dune::Exception, "dot: vectors must have the same size");

    std::vector<T> temp_vec(y.size());
    matrix_->mv(y, temp_vec);

    if constexpr (std::is_same_v<T, double>) return cblas_ddot(static_cast<int>(x.size()), x.data(), 1, temp_vec.data(), 1);
    else if constexpr (std::is_same_v<T, float>) return cblas_sdot(static_cast<int>(x.size()), x.data(), 1, temp_vec.data(), 1);
    else {
      // Here we just try to use the *-operator and hope for the best
      auto temp = x;
      matrix_->mv(y, temp);

      T res = 0;
      for (std::size_t i = 0; i < x.size(); ++i) res += x[i] * temp[i];
      return res;
    }
  }

  /** @brief Returns false since this is not Euclidean */
  bool is_euclidean() const { return false; }

private:
  std::shared_ptr<const Mat> matrix_;
  mutable std::unique_ptr<BlockMultiVec> temp1{};
  mutable std::unique_ptr<BlockMultiVec> temp2{};
};

} // namespace orthogonalisation
