#pragma once

#include "../../types.hh"
#include "../backend.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/multivector.hh"

#include <algorithm>
#include <dune/common/exceptions.hh>
#include <dune/common/fmatrix.hh>
#include <dune/istl/bvector.hh>
#include <variant>
#include <vector>

namespace ddm::backend {
struct HostBackend {
  using context_type = std::monostate;
  static constexpr bool is_device = false;

  template <class Container>
  static context_type context(const Container&)
  {
    return {};
  }

  template <class T>
  using buffer_type = Buffer<T, HostBackend>;

  template <class T>
  static buffer_type<T> make_buffer(context_type ctx, std::size_t n)
  {
    return buffer_type<T>(ctx, n);
  }

  template <class T>
  static buffer_type<T> make_buffer_from_host(context_type ctx, const std::vector<T>& host_data)
  {
    auto b = make_buffer<T>(ctx, host_data.size());
    std::copy(host_data.begin(), host_data.end(), b.data());
    return b;
  }

  template <class T>
  static T* malloc([[maybe_unused]] context_type& ctx, std::size_t n)
  {
    return new T[n];
  }

  template <class T>
  static void free([[maybe_unused]] context_type& ctx, T* p)
  {
    delete[] p;
  }

  /// Blocks until all work enqueued on the context has completed. Nothing is enqueued on the
  /// host, so there is nothing to wait for.
  static void sync([[maybe_unused]] context_type ctx) {}

  template <class T>
  static void gather([[maybe_unused]] context_type ctx, const T* src, const buffer_type<int>& indices, T* dst)
  {
    for (std::size_t i = 0; i < indices.size(); ++i) dst[i] = src[indices.data()[i]];
  }

  template <class T>
  static void scatter([[maybe_unused]] context_type ctx, const T* src, const buffer_type<int>& indices, T* dst)
  {
    for (std::size_t i = 0; i < indices.size(); ++i) dst[indices.data()[i]] = src[i];
  }

  template <ReductionOperation ReduceOp, class T>
  static void scatter_reduce([[maybe_unused]] context_type ctx, const T* src, const buffer_type<int>& indices, T* dst)
  {
    if constexpr (ReduceOp == ReductionOperation::Addition)
      for (std::size_t i = 0; i < indices.size(); ++i) dst[indices.data()[i]] += src[i];
    else static_assert(false);
  }

  /// Column-wise gather from a column-major multivector: dst[i + c*len] = src[indices[i] + c*rows]
  /// for i in [0, len) and c in [0, cols), where len = indices.size().
  template <class T>
  static void gather_columns([[maybe_unused]] context_type ctx, const T* src, const buffer_type<int>& indices, std::size_t rows, std::size_t cols, T* dst)
  {
    const std::size_t len = indices.size();
    for (std::size_t c = 0; c < cols; ++c)
      for (std::size_t i = 0; i < len; ++i) dst[i + c * len] = src[indices.data()[i] + c * rows];
  }

  /// Column-wise scatter into a column-major multivector, the inverse of gather_columns():
  /// dst[indices[i] + c*rows] = src[i + c*len] for i in [0, len) and c in [0, cols).
  template <class T>
  static void scatter_columns([[maybe_unused]] context_type ctx, const T* src, const buffer_type<int>& indices, std::size_t rows, std::size_t cols, T* dst)
  {
    const std::size_t len = indices.size();
    for (std::size_t c = 0; c < cols; ++c)
      for (std::size_t i = 0; i < len; ++i) dst[indices.data()[i] + c * rows] = src[i + c * len];
  }

  template <class V>
  static void copy_n(const V& src, std::size_t n, V& dst)
  {
    std::copy_n(src.data(), n, dst.data());
  }

  /// Pointer-level overload of copy_n() above, for destinations that are not backend vectors
  /// (e.g. a column inside a MultiVector). Synchronous.
  template <class T>
  static void copy_n([[maybe_unused]] context_type ctx, const T* src, std::size_t n, T* dst)
  {
    std::copy_n(src, n, dst);
  }

  // The host backend's "device" memory is host memory, so the two copies below are plain
  // std::copy_n; they exist so that generic code can name the direction it means.

  /// Copies \p n entries from host memory into backend memory. Synchronous.
  template <class T>
  static void copy_from_host([[maybe_unused]] context_type ctx, const T* src, T* dst, std::size_t n)
  {
    std::copy_n(src, n, dst);
  }

  /// Copies \p n entries from backend memory into host memory. Synchronous.
  template <class T>
  static void copy_to_host([[maybe_unused]] context_type ctx, const T* src, T* dst, std::size_t n)
  {
    std::copy_n(src, n, dst);
  }

  template <class V>
  static void pointwise_mult(const V& x, V& y)
  {
    DDM_ASSERT(x.size() == y.size(), "Vectors in pointwise_mult do not match");

    for (std::size_t i = 0; i < x.size(); ++i) y[i] *= x[i];
  }

  template <class V>
  static auto masked_dot(const V& x, const V& mask, const V& y)
  {
    auto tmp = x;
    pointwise_mult(mask, tmp);
    return tmp.dot(y);
  }

  template <class T>
  static void zero([[maybe_unused]] context_type ctx, T* dst, std::size_t n)
  {
    std::fill_n(dst, n, T{});
  }

  /// Computes out[k] = col_k(R) . v for every column k of \p R; out must have room for R.cols() entries.
  /// Synchronous, i.e. out (host memory) is valid on return.
  template <class Scalar, class Index>
  static void batched_dot([[maybe_unused]] context_type ctx, const MultiVector<Scalar, HostBackend, Index>& R, const Scalar* v, Scalar* out)
  {
    for (Index k = 0; k < R.cols(); ++k) {
      const Scalar* col = R.col(k);
      Scalar sum{};
      for (Index i = 0; i < R.rows(); ++i) sum += col[i] * v[i];
      out[k] = sum;
    }
  }

  /// x += sum_k c[k] * col_k(R); c has R.cols() entries, x has R.rows() entries, all pointers on the
  /// backend (host memory here).
  template <class Scalar, class Index>
  static void gemv_t([[maybe_unused]] context_type ctx, const MultiVector<Scalar, HostBackend, Index>& R, const Scalar* c, Scalar* x)
  {
    // Column-major friendly loop order: each column is read sequentially
    for (Index k = 0; k < R.cols(); ++k) {
      const Scalar* col = R.col(k);
      const Scalar ck = c[k];
      for (Index i = 0; i < R.rows(); ++i) x[i] += ck * col[i];
    }
  }

  template <class Matrix, class Scalar, class Index>
  static void spmm(const Matrix& A, const MultiVector<Scalar, HostBackend, Index>& X, MultiVector<Scalar, HostBackend, Index>& Y)
  {
    DDM_CHECK(A.M() == X.rows(), "The number of columns in A ({}) and rows in X ({}) do not match", A.M(), X.rows());
    DDM_CHECK(A.N() == Y.rows(), "The number of rows in A ({}) and rows in Y ({}) do not match", A.N(), Y.rows());
    DDM_CHECK(X.cols() == Y.cols(), "The number of cols in X ({}) and cols in Y ({}) do not match", X.cols(), Y.cols());

    const Index m = X.cols();
    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      const Index i = ri.index();
      for (Index k = 0; k < m; ++k) {
        const Scalar* x = X.col(k);

        Scalar sum = 0;
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) sum += *ci * x[ci.index()];
        Y.col(k)[i] = sum;
      }
    }
  }

  // template <class T>
  // static void copy_at_indices(const buffer_type<T>& src, const buffer_type<int>& indices, buffer_type<T>& dst)
  // {
  //   std::for_each_n(indices.data(), indices.size(), [&](auto i) { dst.data()[i] = src.data()[i]; });
  // }
};

template <class B, class A>
struct backend_traits<std::vector<B, A>> {
  using type = HostBackend;
};

template <class B, class A>
struct backend_traits<Dune::BlockVector<B, A>> {
  using type = HostBackend;
};

template <class B, class A>
struct backend_traits<Dune::BCRSMatrix<B, A>> {
  using type = HostBackend;
};

template <class S, class I>
struct backend_traits<ddm::MultiVector<S, HostBackend, I>> {
  using type = HostBackend;
};
} // namespace ddm::backend
