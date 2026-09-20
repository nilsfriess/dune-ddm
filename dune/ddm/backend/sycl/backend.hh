#pragma once

#if __has_include(<sycl/sycl.hpp>)

#include "../backend.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/multivector.hh"
#include "dune/ddm/types.hh"

#include <sycl/sycl.hpp>

namespace ddm::Sycl {
template <class Scalar, class Index>
class Mat;
template <class Scalar, class Index>
class Vec;
} // namespace ddm::Sycl

namespace ddm::backend {
struct SyclBackend {
  using context_type = sycl::queue;
  static constexpr bool is_device = true;

  template <class Container>
  static context_type context(const Container& c)
  {
    // Backend vectors carry their queue as .queue(), multivectors as .context()
    if constexpr (requires { c.queue(); }) return c.queue();
    else return c.context();
  }

  template <class T>
  using buffer_type = Buffer<T, SyclBackend>;

  template <class T>
  static buffer_type<T> make_buffer(context_type ctx, std::size_t n)
  {
    return buffer_type<T>(ctx, n);
  }

  template <class T>
  static buffer_type<T> make_buffer_from_host(context_type ctx, const std::vector<T>& host_data)
  {
    auto b = make_buffer<T>(ctx, host_data.size());
    ctx.memcpy(b.data(), host_data.data(), host_data.size() * sizeof(T)).wait();
    return b;
  }

  template <class T>
  static T* malloc([[maybe_unused]] context_type& ctx, std::size_t n)
  {
    return sycl::malloc_device<T>(n, ctx);
  }

  template <class T>
  static void free(context_type& ctx, T* p)
  {
    ctx.wait();
    sycl::free(p, ctx);
  }

  /// Blocks until all work enqueued on the queue has completed. Needed whenever the result of an
  /// enqueued kernel is handed to something that does not know about the queue, e.g. MPI.
  static void sync(context_type ctx) { ctx.wait(); }

  // The n == 0 guard has the same reason as in zero(): a memcpy with a null pointer is recorded
  // as an asynchronous error by AdaptiveCpp, even with a zero count.

  /// Pointer-level overload of copy_n() below, for destinations that are not backend vectors
  /// (e.g. a column inside a MultiVector). Synchronous, unlike the kernel-based overload.
  template <class T>
  static void copy_n(context_type ctx, const T* src, std::size_t n, T* dst)
  {
    if (n == 0) return;
    ctx.memcpy(dst, src, n * sizeof(T)).wait();
  }

  /// Copies \p n entries from host memory into device memory. Synchronous.
  template <class T>
  static void copy_from_host(context_type ctx, const T* src, T* dst, std::size_t n)
  {
    if (n == 0) return;
    ctx.memcpy(dst, src, n * sizeof(T)).wait();
  }

  /// Copies \p n entries from device memory into host memory. Synchronous.
  template <class T>
  static void copy_to_host(context_type ctx, const T* src, T* dst, std::size_t n)
  {
    if (n == 0) return;
    ctx.memcpy(dst, src, n * sizeof(T)).wait();
  }

  template <class T>
  static void zero(context_type ctx, T* dst, std::size_t n)
  {
    if (n == 0) return; // memset(nullptr, 0, 0) is recorded as an asynchronous error by AdaptiveCpp
    // The all-zero bit pattern is 0.0 in every scalar type we support
    ctx.memset(dst, 0, n * sizeof(T));
  }

  /// Computes out[k] = col_k(R) . v for every column k of \p R; out must have room for R.cols() entries.
  /// Synchronous, i.e. out (host memory) is valid on return.
  // TODO(20260917-103910): One kernel launch plus one malloc/free per call. If this ever shows up
  //                         in a profile, let callers reuse a scratch buffer across calls.
  template <class Scalar, class Index>
  static void batched_dot(context_type ctx, const MultiVector<Scalar, SyclBackend, Index>& R, const Scalar* v,
      Scalar* out)
  {
    const std::size_t cols = R.cols();
    if (cols == 0) return;

    const auto rows = R.rows();
    const Scalar* r = R.data();
    Scalar* out_dev = malloc<Scalar>(ctx, cols);
    ctx.parallel_for(sycl::range<1>(cols), [=](auto item) {
      const std::size_t k = item[0];
      const Scalar* col = r + k * rows;
      Scalar sum{};
      for (Index i = 0; i < rows; ++i) sum += col[i] * v[i];
      out_dev[k] = sum;
    });
    ctx.memcpy(out, out_dev, cols * sizeof(Scalar)).wait();
    free(ctx, out_dev);
  }

  /// x += sum_k c[k] * col_k(R); c has R.cols() entries, x has R.rows() entries, all pointers on the
  /// backend (device memory here).
  template <class Scalar, class Index>
  static void gemv_t(context_type ctx, const MultiVector<Scalar, SyclBackend, Index>& R, const Scalar* c, Scalar* x)
  {
    const auto cols = R.cols();
    if (cols == 0) return;

    const auto rows = R.rows();
    const Scalar* r = R.data();
    ctx.parallel_for(sycl::range<1>(rows), [=](auto item) {
      const std::size_t i = item[0];
      Scalar sum{};
      for (Index k = 0; k < cols; ++k) sum += c[k] * r[k * rows + i];
      x[i] += sum;
    });
  }

  template <class T>
  static void gather(context_type ctx, const T* src, const buffer_type<int>& indices, T* dst)
  {
    const auto* id_data = indices.data();
    ctx.parallel_for(sycl::range<1>(indices.size()), [=](auto id) { dst[id] = src[id_data[id]]; });
  }

  template <class T>
  static void scatter(context_type ctx, const T* src, const buffer_type<int>& indices, T* dst)
  {
    const auto* id_data = indices.data();
    ctx.parallel_for(sycl::range<1>(indices.size()), [=](auto id) { dst[id_data[id]] = src[id]; });
  }

  template <ReductionOperation ReduceOp, class T>
  static void scatter_reduce(context_type ctx, const T* src, const buffer_type<int>& indices, T* dst)
  {
    const auto* id_data = indices.data();
    ctx.parallel_for(sycl::range<1>(indices.size()), [=](auto id) {
      if constexpr (ReduceOp == ReductionOperation::Addition) {
        // A duplicated index would otherwise be a read-modify-write race between work-items and
        // lose updates. The communication plans never produce duplicates, so the atomic is a
        // correctness net, not a hot path: this runs on small boundary buffers.
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> dst_ref(dst[id_data[id]]);
        dst_ref += src[id];
      }
      else static_assert(false);
    });
  }

  template <class V>
  static void copy_n(const V& src, std::size_t n, V& dst)
  {
    const auto* s = src.data();
    auto* d = dst.data();

    auto ctx = context(src);
    ctx.parallel_for(sycl::range<1>(n), [=](auto id) { d[id] = s[id]; });
  }

  /// Column-wise gather from a column-major multivector: dst[i + c*len] = src[indices[i] + c*rows]
  /// for i in [0, len) and c in [0, cols), where len = indices.size().
  template <class T>
  static void gather_columns(context_type ctx, const T* src, const buffer_type<int>& indices, std::size_t rows,
      std::size_t cols, T* dst)
  {
    const std::size_t len = indices.size();
    if (len == 0 or cols == 0) return; // parallel_for rejects empty ranges
    const auto* id_data = indices.data();
    ctx.parallel_for(sycl::range<2>(len, cols), [=](auto item) {
      const std::size_t i = item[0];
      const std::size_t c = item[1];
      dst[i + c * len] = src[id_data[i] + c * rows];
    });
  }

  /// Column-wise scatter into a column-major multivector, the inverse of gather_columns():
  /// dst[indices[i] + c*rows] = src[i + c*len] for i in [0, len) and c in [0, cols).
  template <class T>
  static void scatter_columns(context_type ctx, const T* src, const buffer_type<int>& indices, std::size_t rows,
      std::size_t cols, T* dst)
  {
    const std::size_t len = indices.size();
    if (len == 0 or cols == 0) return; // parallel_for rejects empty ranges
    const auto* id_data = indices.data();
    ctx.parallel_for(sycl::range<2>(len, cols), [=](auto item) {
      const std::size_t i = item[0];
      const std::size_t c = item[1];
      dst[id_data[i] + c * rows] = src[i + c * len];
    });
  }

  template <class V>
  static void pointwise_mult(const V& x, V& y)
  {
    const auto* xd = x.data();
    auto* yd = y.data();

    auto ctx = context(x);
    ctx.parallel_for(sycl::range<1>(x.size()), [=](auto id) { yd[id] *= xd[id]; });
  }

  template <class V>
  static auto masked_dot(const V& x, const V& mask, const V& y)
  {
    return x.masked_dot(mask, y);
  }

  template <class MatrixScalar, class MatrixIndex, class VectorScalar>
  static void spmm(const Sycl::Mat<MatrixScalar, MatrixIndex>& A, const MultiVector<VectorScalar, SyclBackend>& X, MultiVector<VectorScalar, SyclBackend>& Y)
  {
    A.mv(X, Y);
  }
};

template <class S, class I>
struct backend_traits<ddm::Sycl::Mat<S, I>> {
  using type = SyclBackend;
};

template <class S, class I>
struct backend_traits<ddm::Sycl::Vec<S, I>> {
  using type = SyclBackend;
};

template <class S, class I>
struct backend_traits<ddm::MultiVector<S, SyclBackend, I>> {
  using type = SyclBackend;
};
} // namespace ddm::backend

#endif
