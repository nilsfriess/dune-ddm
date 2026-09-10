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
 *  The destructor also calls queue.wait() to ensure that memory that is used in queued kernels is not freed.
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

  /** Sums f(0) + ... + f(n-1) on the device.  */
  template <class F>
  void reduce(F f) const
  {
    if (n == 0) return;

    if (local_size == 0) {
      const auto dev = q.get_device();
      const std::size_t cus = std::max<std::size_t>(1, dev.get_info<sycl::info::device::max_compute_units>());

      local_size = 256;
      std::size_t num_groups = 8 * cus;
      global_size = local_size * num_groups;
    }

    auto* sum = red_dev;
    const auto N = n;
#if 1
    q.memset(sum, 0, sizeof(Scalar));
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> it) {
        const Index gid = it.get_global_id(0);
        const Index gsize = it.get_global_range(0);

        // Reduce locally on this work-item ...
        Scalar msum = 0;
        for (Index i = gid; i < N; i += gsize) msum += f(static_cast<Index>(i));

        // ... then reduce within the work-group ...
        auto reduced_sum = sycl::reduce_over_group(it.get_group(), msum, sycl::plus<Scalar>());

        // ... and finally let the leader reduce into the global sum using atomics
        if (it.get_group().leader()) {
          sycl::atomic_ref<Scalar, sycl::memory_order::relaxed, sycl::memory_scope::device> sum_ref(sum[0]);
          sum_ref += reduced_sum;
        }
      });
    });
#else
    // initialize_to_identity is required: without it the reduction combines into whatever the
    // previous reduce() left in red_dev.
    q.parallel_for(sycl::range<1>(n), sycl::reduction(sum, sycl::plus<field_type>{}, sycl::property::reduction::initialize_to_identity{}),
                   [=](sycl::id<1> idx, auto& s) { s += f(static_cast<Index>(idx[0])); });
#endif
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
  }

  mutable sycl::queue q;
  Index n;
  field_type* data_;

  // Scratch for the reductions in dot/masked_dot/two_norm: the device and pinned host locations
  // of the final result.
  field_type* red_dev;
  field_type* red_host;

  mutable std::size_t local_size = 0;
  mutable std::size_t global_size;
};
} // namespace ddm::Sycl
