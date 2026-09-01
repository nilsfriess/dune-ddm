#pragma once

#include "../backend.hh"
#include "dune/ddm/types.hh"

#include <sycl/sycl.hpp>

namespace ddm::backend {
struct SyclBackend {
  using context_type = sycl::queue;

  template <class Container>
  static context_type context(const Container& c)
  {
    return c.queue();
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
  static void free([[maybe_unused]] context_type& ctx, T* p)
  {
    ctx.wait();
    sycl::free(p, ctx);
  }

  /// Blocks until all work enqueued on the queue has completed. Needed whenever the result of an
  /// enqueued kernel is handed to something that does not know about the queue, e.g. MPI.
  static void sync(context_type ctx) { ctx.wait(); }

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

  // TODO: This assumes indices does not contain repeated entries
  template <ReductionOperation ReduceOp, class T>
  static void scatter_reduce(context_type ctx, const T* src, const buffer_type<int>& indices, T* dst)
  {
    const auto* id_data = indices.data();
    ctx.parallel_for(sycl::range<1>(indices.size()), [=](auto id) {
      if constexpr (ReduceOp == ReductionOperation::Addition) dst[id_data[id]] += src[id];
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

  template <class V>
  static void pointwise_mult(const V& x, V& y)
  {
    const auto* xd = x.data();
    auto* yd = y.data();

    auto ctx = context(x);
    ctx.parallel_for(sycl::range<1>(x.size()), [=](auto id) { yd[id] *= xd[id]; });
  }
};
} // namespace ddm::backend
