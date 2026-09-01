#pragma once

#include "../backend.hh"
#include "../../types.hh"

#include <algorithm>
#include <dune/istl/bvector.hh>
#include <variant>
#include <vector>

namespace ddm::backend {
struct HostBackend {
  using context_type = std::monostate;

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

  template <class V>
  static void copy_n(const V& src, std::size_t n, V& dst)
  {
    std::copy_n(src.data(), n, dst.data());
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
} // namespace ddm::backend
