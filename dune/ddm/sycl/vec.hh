#pragma once

#include "../backend/backend.hh"
#include "../backend/sycl/backend.hh"
#include "mat.hh"

#include <algorithm>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

namespace ddm::Sycl {
/** @brief A Dune ISTL-compatible vector class that stores memory on the device associated with a sycl::queue
 *
 *  Data is allocated using SYCL's USM function sycl::malloc_device. It is assumed that the queue that is passed
 *  to the constructor is an in-order queue. Because of this assumption, most members don't call queue.wait()
 *  after enqueueing their operations; only operations that return values on the host (like dot() and two_norm())
 *  on the queue. The destructor also calls queue.wait().
 *
 *  Methods that accept other Vec's as arguments must only be called for vectors that live on the same queue.
 */
template <class Scalar, class Index = std::uint_least32_t>
class Vec {
public:
  using field_type = Scalar; // TODO: Remove?
  using value_type = Scalar;

  Vec(sycl::queue q_, Index n_)
      : q(q_)
      , n(n_)
      , data_(sycl::malloc_device<field_type>(n, q))
      , red_dev(sycl::malloc_device<field_type>(1, q))
      , red_host(sycl::malloc_host<field_type>(1, q))
  {
  }

  template <class HostContainer>
  static Vec from_host_vector(sycl::queue q_, const HostContainer& v_host)
  {
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*v_host.data())>, Scalar>) {
      Vec v(q_, v_host.size());
      q_.memcpy(v.data(), v_host.data(), v_host.size() * sizeof(Scalar)).wait();
      return v;
    }
    else {
      std::vector<Scalar> v_host_scalar(v_host.begin(), v_host.end());
      Vec v(q_, v_host_scalar.size());
      q_.memcpy(v.data(), v_host_scalar.data(), v_host_scalar.size() * sizeof(Scalar)).wait();
      return v;
    }
  }

  std::vector<Scalar> to_host_vector() const
  {
    std::vector<Scalar> v_host(size());
    q.memcpy(v_host.data(), data(), size() * sizeof(Scalar)).wait();
    return v_host;
  }

  Vec(const Vec& other)
      : q(other.q)
      , n(other.n)
      , data_(sycl::malloc_device<field_type>(other.n, q))
      , red_dev(sycl::malloc_device<field_type>(1, q))
      , red_host(sycl::malloc_host<field_type>(1, q))
  {
    q.memcpy(data_, other.data_, n * sizeof(field_type));
  }

  Vec(Vec&& other) noexcept
      : q(other.q)
      , n(std::exchange(other.n, 0))
      , data_(std::exchange(other.data_, nullptr))
      , red_dev(std::exchange(other.red_dev, nullptr))
      , red_host(std::exchange(other.red_host, nullptr))
  {
  }

  Vec& operator=(const Vec& other)
  {
    if (this == &other) return *this;

    if (n != other.n) {
      release();
      q = other.q;
      n = other.n;
      data_ = sycl::malloc_device<field_type>(n, q);
      red_dev = sycl::malloc_device<field_type>(1, q);
      red_host = sycl::malloc_host<field_type>(1, q);
    }
    q.memcpy(data_, other.data_, n * sizeof(field_type));
    return *this;
  }

  Vec& operator=(Vec&& other) noexcept
  {
    if (this == &other) return *this;

    release();
    q = other.q;
    n = std::exchange(other.n, 0);
    data_ = std::exchange(other.data_, nullptr);
    red_dev = std::exchange(other.red_dev, nullptr);
    red_host = std::exchange(other.red_host, nullptr);
    return *this;
  }

  ~Vec() { release(); }

  Index size() const { return n; }

  field_type dot(const Vec& y) const
  {
    const auto* v = data_;
    const auto* w = y.data_;
    reduce([=](Index i) { return v[i] * w[i]; });
    return finish_reduction();
  }

  field_type masked_dot(const Vec& mask, const Vec& y) const
  {
    const auto* u = mask.data_;
    const auto* v = data_;
    const auto* w = y.data_;
    reduce([=](Index i) { return u[i] * v[i] * w[i]; });
    return finish_reduction();
  }

  field_type two_norm() const
  {
    using std::sqrt;
    return sqrt(two_norm2());
  }

  field_type two_norm2() const
  {
    const auto* v = data_;
    reduce([=](Index i) { return v[i] * v[i]; });
    return finish_reduction();
  }

  Vec& operator=(field_type a)
  {
    q.fill(data_, a, n);
    return *this;
  }

  Vec& operator*=(field_type a)
  {
    if (a == field_type(1)) return *this;

    auto* v = data_;
    q.parallel_for(sycl::range<1>(n), [=](auto idx) { v[idx] *= a; });
    return *this;
  }

  Vec& operator+=(const Vec& other)
  {
    DDM_CHECK(n == other.n, "Size mismatch in operator+= ({} vs {})", n, other.n);
    auto* v = data_;
    const auto* w = other.data_;
    q.parallel_for(sycl::range<1>(n), [=](auto idx) { v[idx] += w[idx]; });
    return *this;
  }

  Vec& operator-=(const Vec& other)
  {
    DDM_CHECK(n == other.n, "Size mismatch in operator-= ({} vs {})", n, other.n);
    auto* v = data_;
    const auto* w = other.data_;
    q.parallel_for(sycl::range<1>(n), [=](auto idx) { v[idx] -= w[idx]; });
    return *this;
  }

  void axpy(field_type a, const Vec& y)
  {
    DDM_CHECK(n == y.n, "Size mismatch in axpy ({} vs {})", n, y.n);
    auto* v = data_;
    const auto* w = y.data_;
    q.parallel_for(sycl::range<1>(n), [=](auto idx) { v[idx] += a * w[idx]; });
  }

  sycl::queue queue() const { return q; }
  field_type* data() const { return data_; }

private:
  friend class Mat<Scalar, Index>;

  /** Sums f(0) + ... + f(n-1) on the device.
   *
   *  A hand-written chunked two-stage reduction. Each of nchunks work items sums one contiguous
   *  chunk of elements (measurable, cache-friendly work per item), a second kernel folds the
   *  partial sums, and finish_reduction() copies the result to the host. On the AdaptiveCpp OpenMP
   *  target this is ~3-5x faster than sycl::reduction, whose hierarchical work-group reduction
   *  engine plus per-submit scratch setup dominates small (~1 MB) reductions; a microbenchmark
   *  measured 0.34 ms vs 0.10 ms per 3-array dot of 66k doubles under 4-way rank contention.
   */
  template <class F>
  void reduce(F f) const
  {
    if (n == 0) return;

    if (partials == nullptr) {
      nchunks = std::clamp<Index>(n / 64, 8, 1024);
      partials = sycl::malloc_device<field_type>(nchunks, q);
    }

    const auto count = n;
    const auto nc = nchunks;
    const std::size_t chunk = (count + nc - 1) / nc;
    auto* part = partials;
    auto* sum = red_dev;
    q.parallel_for(sycl::range<1>(nc), [=](auto idx) {
      const std::size_t c = idx[0];
      const std::size_t lo = c * chunk;
      const std::size_t hi = std::min<std::size_t>(count, (c + 1) * chunk);
      field_type s{0};
      for (std::size_t i = lo; i < hi; ++i) s += f(i);
      part[c] = s;
    });
    q.parallel_for(sycl::range<1>(1), [=](auto) {
      field_type s{0};
      for (Index c = 0; c < nc; ++c) s += part[c];
      sum[0] = s;
    });
  }

  /// Copies the result of a preceding reduce() back to the host. Blocks until it has arrived.
  field_type finish_reduction() const
  {
    if (n == 0) return field_type{0};
    q.memcpy(red_host, red_dev, sizeof(field_type)).wait();
    return *red_host;
  }

  void release()
  {
    if (data_ == nullptr) return;
    q.wait(); // make sure all kernels are done when we try to free the memory
    sycl::free(data_, q);
    sycl::free(red_dev, q);
    sycl::free(red_host, q);
    if (partials) sycl::free(partials, q);
  }

  mutable sycl::queue q;
  Index n;
  field_type* data_;

  // Scratch for the reductions in dot/masked_dot/two_norm: one partial sum per chunk on the
  // device, plus the device and pinned host locations of the final result.
  mutable Index nchunks = 0;
  mutable field_type* partials = nullptr;
  field_type* red_dev;
  field_type* red_host;
};
} // namespace ddm::Sycl

namespace ddm::backend {
template <class B, class A>
struct backend_traits<ddm::Sycl::Vec<B, A>> {
  using type = SyclBackend;
};

} // namespace ddm::backend
