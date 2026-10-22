#pragma once

#include "ddm/backend_id.hh"
#include "ddm/index.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/vec/host_view.hh"
#include "ddm/vec/vec.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <dune/common/math.hh>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace ddm {
template <class T>
class IstlMultiVec;

// Only valid for multivectors with backend() == BackendId::istl, see as_istl() for vectors
template <class T>
IstlMultiVec<T>& as_istl(MultiVec<T>& X)
{
  return static_cast<IstlMultiVec<T>&>(X);
}

template <class T>
const IstlMultiVec<T>& as_istl(const MultiVec<T>& X)
{
  return static_cast<const IstlMultiVec<T>&>(X);
}

// The multivector of a vector with backend() == BackendId::istl, its data() are the entries of the vector
template <class T>
IstlMultiVec<T>& as_istl(Vec<T>& v)
{
  return as_istl(v.multivec());
}

template <class T>
const IstlMultiVec<T>& as_istl(const Vec<T>& v)
{
  return as_istl(v.multivec());
}

template <class T>
class IstlMultiVec final : public MultiVec<T> {
public:
  IstlMultiVec(Index rows, Index cols)
      : MultiVec<T>(rows, cols)
      , storage_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), T(0))
      , data_(storage_)
  {
  }

  BackendId backend() const override { return BackendId::istl; }

  // The data, column by column
  std::span<T> data() { return data_; }
  std::span<const T> data() const { return data_; }

private:
  // A view of the columns [first, first + count) of parent
  IstlMultiVec(const IstlMultiVec& parent, Index first, Index count)
      : MultiVec<T>(parent, first, count)
      , data_(parent.data_.subspan(static_cast<std::size_t>(first) * static_cast<std::size_t>(parent.rows()), static_cast<std::size_t>(count) * static_cast<std::size_t>(parent.rows())))
  {
  }

  std::unique_ptr<MultiVec<T>> do_columns(Index first, Index count) const override { return std::unique_ptr<MultiVec<T>>(new IstlMultiVec(*this, first, count)); }

  std::unique_ptr<MultiVec<T>> do_create_multivector(Index rows, Index cols) const override { return std::make_unique<IstlMultiVec>(rows, cols); }

  void do_fill(T value) override { std::ranges::fill(data_, value); }

  void do_scale(T alpha) override
  {
    for (auto& x : data_) x *= alpha;
  }

  void do_copy_from(const MultiVec<T>& X) override { std::ranges::copy(as_istl(X).data(), data_.begin()); }

  void do_axpy(T alpha, const MultiVec<T>& X) override
  {
    const auto x = as_istl(X).data();
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += alpha * x[i];
  }

  void do_pointwise_mult(const MultiVec<T>& Z, MultiVec<T>& Y) const override
  {
    const auto z = as_istl(Z).data();
    auto y = as_istl(Y).data();
    for (std::size_t i = 0; i < data_.size(); ++i) y[i] = data_[i] * z[i];
  }

  // result[j] = x_j^H y_j, the dot product of column j of X = *this and column j of Y
  void do_column_dots(const MultiVec<T>& Y, std::span<T> result) const override
  {
    const auto y = as_istl(Y).data();
    const std::size_t n = this->rows();
    for (std::size_t j = 0; j < result.size(); ++j) {
      T sum{0};
      for (std::size_t i = 0; i < n; ++i) sum += Dune::conjugateComplex(data_[j * n + i]) * y[j * n + i];
      result[j] = sum;
    }
  }

  // result[j] = ||x_j||_2, the Euclidean norm of column j
  void do_column_norms(std::span<typename MultiVec<T>::real_type> result) const override
  {
    using Real = typename MultiVec<T>::real_type;
    const std::size_t n = this->rows();
    for (std::size_t j = 0; j < result.size(); ++j) {
      Real sum{0};
      for (std::size_t i = 0; i < n; ++i) {
        const T v = data_[j * n + i];
        if constexpr (std::is_same_v<T, Real>) sum += v * v;
        else sum += v.real() * v.real() + v.imag() * v.imag();
      }
      result[j] = std::sqrt(sum);
    }
  }

  // X(i, j) = src(i, j) for i < n, i.e. copies the first n rows of every column of src
  void do_copy_rows_from(const MultiVec<T>& src, Index n) override
  {
    const auto s = as_istl(src).data();
    const std::size_t rows = this->rows();
    const std::size_t src_rows = src.rows();
    const std::size_t m = this->cols();
    for (std::size_t j = 0; j < m; ++j) std::copy_n(s.data() + j * src_rows, n, data_.data() + j * rows);
  }

  void do_dot(const MultiVec<T>& Y, MultiVec<T>& C) const override
  {
    const auto y = as_istl(Y).data();
    auto c = as_istl(C).data();
    const std::size_t n = this->rows();
    const std::size_t m_x = this->cols();
    const std::size_t m_y = Y.cols();

    for (std::size_t b = 0; b < m_y; ++b) {
      for (std::size_t a = 0; a < m_x; ++a) {
        T sum{0};
        for (std::size_t i = 0; i < n; ++i) sum += Dune::conjugateComplex(data_[a * n + i]) * y[b * n + i];
        c[b * m_x + a] = sum;
      }
    }
  }

  // Y = alpha * X M + beta * Y
  // TODO: Use BLAS for this
  void do_mult(const MultiVec<T>& M, MultiVec<T>& Y, T alpha, T beta) const override
  {
    const auto m = as_istl(M).data();
    auto y = as_istl(Y).data();
    const std::size_t n = this->rows();
    const std::size_t m_x = this->cols();
    const std::size_t m_y = Y.cols();

    // Column b of Y is a linear combination of the columns of X, so every loop over the rows is contiguous
    for (std::size_t b = 0; b < m_y; ++b) {
      auto y_b = y.subspan(b * n, n);
      if (beta == T{0}) std::ranges::fill(y_b, T{0});
      else if (beta != T{1})
        for (auto& v : y_b) v *= beta;

      for (std::size_t a = 0; a < m_x; ++a) {
        const T coeff = alpha * m[b * m_x + a];
        for (std::size_t i = 0; i < n; ++i) y_b[i] += coeff * data_[a * n + i];
      }
    }
  }

  void do_pack_rows(std::span<const Index> idx, MultiVec<T>& buffer) const override
  {
    auto b = as_istl(buffer).data();
    const std::size_t n = this->rows();
    const std::size_t k_max = idx.size();
    const std::size_t m = this->cols();

    for (std::size_t c = 0; c < m; ++c)
      for (std::size_t k = 0; k < k_max; ++k) b[c * k_max + k] = data_[c * n + idx[k]];
  }

  void do_unpack_rows(const MultiVec<T>& buffer, std::span<const Index> idx, Index first_col) override
  {
    const auto b = as_istl(buffer).data();
    const std::size_t n = this->rows();
    const std::size_t k_max = idx.size();
    const std::size_t m = buffer.cols();
    const std::size_t first = first_col;

    for (std::size_t c = 0; c < m; ++c)
      for (std::size_t k = 0; k < k_max; ++k) data_[(first + c) * n + idx[k]] = b[c * k_max + k];
  }

  void do_scale_rows(const MultiVec<T>& d) override
  {
    const auto d_istl = as_istl(d).data();
    const std::size_t n = this->rows();
    const std::size_t m = this->cols();
    for (std::size_t c = 0; c < m; ++c)
      for (std::size_t i = 0; i < n; ++i) data_[c * n + i] *= d_istl[i];
  }

  // The data is in host memory already, so there is nothing to copy
  std::span<T> acquire_host(Access /*mode*/) override { return data_; }
  void release_host(Access /*mode*/) override {}

  // TODO(20261007-071012): Use an AlignedAllocator to allocate the MultiVec data
  std::vector<T> storage_; ///< column by column, empty for views
  std::span<T> data_;      ///< storage_, or the columns of the owner for views
};
} // namespace ddm
