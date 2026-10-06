#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/registry.hh"
#include "ddm/vec/exchange_plan.hh"
#include "ddm/vec/host_view.hh"

#include <cstddef>
#include <dune/common/parametertree.hh>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace ddm {
// Interface implemented by the vector backends. Users of the library work with Vec<T>, which owns a VecImpl<T> and
// gives it value semantics. Arguments of type VecImpl<T> are guaranteed to have the same size and backend as *this,
// Vec<T> checks this before calling into the implementation.
template <class T>
class VecImpl {
public:
  virtual ~VecImpl() = default;
  VecImpl(const VecImpl&) = delete;
  VecImpl& operator=(const VecImpl&) = delete;

  virtual Index size() const = 0;
  virtual BackendId backend() const = 0;

  // Returns a deep copy
  virtual std::unique_ptr<VecImpl> clone() const = 0;

  // *this = other, reusing the existing storage
  virtual void copy_from(const VecImpl& other) = 0;

  virtual void fill(T value) = 0;
  virtual void scale(T alpha) = 0;

  // *this += alpha * x
  virtual void axpy(T alpha, const VecImpl& x) = 0;

  virtual T dot(const VecImpl& y) const = 0;
  virtual T two_norm() const = 0;

  // y[i] = *this[i] * x[i]
  virtual void pointwise_mult(const VecImpl& x, VecImpl& y) const = 0;

  // *this[i] = src[i] for i < n. src may have a different size than *this, n is at most the smaller one
  virtual void copy_n_from(const VecImpl& src, Index n) = 0;

  // Returns the data in host memory. For Access::write, the contents of the returned memory may be arbitrary. Vec<T>
  // guarantees that at most one access is active at a time and that every acquire_host() is followed by a
  // release_host() with the same mode.
  virtual std::span<T> acquire_host(Access mode) = 0;

  // Ends the access started by acquire_host(). For Access::write and Access::read_write, the data written to the span
  // must be part of the vector afterwards.
  virtual void release_host(Access mode) = 0;

  // Creates the buffers to exchange the entries at the indices in idxs with other ranks. The default works through
  // acquire_host() and release_host(), backends that keep their data elsewhere can override this to avoid the copies.
  virtual std::unique_ptr<ExchangePlan<T>> make_exchange_plan(const CommunicationPattern::IndexMap& idxs) const
  {
    return std::make_unique<HostExchangePlan<T>>(idxs);
  }

protected:
  VecImpl() = default;
};

// Vector with value semantics: copies are deep copies. This is what ISTL's solvers expect from their vector type.
// Copying allocates (through VecImpl::clone()), assigning does not.
//
// The data can be accessed through host_view(). While a view is open, the vector cannot be used otherwise (all
// operations, including access through impl(), fail), and only one view can be open at a time.
template <class T>
class Vec {
public:
  using field_type = T;

  explicit Vec(std::unique_ptr<VecImpl<T>> impl)
      : impl_(std::move(impl))
  {
    DDM_CHECK(impl_ != nullptr, "vec: implementation is nullptr");
  }

  Vec(const Vec& other)
      : impl_(other.impl().clone())
  {
  }

  // The view holds a pointer to the vector, so a vector with an open view must neither be moved nor destroyed
  Vec(Vec&& other) noexcept
      : impl_(std::move(other.impl_))
  {
    DDM_ASSERT(!other.view_open_, "vec: moved from while a host view is open");
  }

  ~Vec() { DDM_ASSERT(!view_open_, "vec: destroyed while a host view is open"); }

  Vec& operator=(const Vec& other)
  {
    if (this == &other) return *this;
    if (!impl_) impl_ = other.impl().clone(); // *this was moved from
    else {
      check_compatible(other, "operator=");
      impl().copy_from(other.impl());
    }
    return *this;
  }

  Vec& operator=(Vec&& other) noexcept
  {
    DDM_ASSERT(!view_open_ && !other.view_open_, "vec: move assignment while a host view is open");
    impl_ = std::move(other.impl_);
    return *this;
  }

  // Sets all entries to value
  Vec& operator=(T value)
  {
    fill(value);
    return *this;
  }

  // size() and backend() are allowed while a view is open
  Index size() const { return checked_impl().size(); }
  BackendId backend() const { return checked_impl().backend(); }

  HostView<const T> host_view(read_t) const
  {
    auto data = open_view(Access::read);
    return HostView<const T>(this, std::span<const T>(data), Access::read);
  }

  HostView<T> host_view(write_t) { return HostView<T>(this, open_view(Access::write), Access::write); }
  HostView<T> host_view(read_write_t) { return HostView<T>(this, open_view(Access::read_write), Access::read_write); }

  void zero() { fill(T{0}); }
  void fill(T value) { impl().fill(value); }

  Vec& operator*=(T alpha)
  {
    impl().scale(alpha);
    return *this;
  }

  Vec& operator+=(const Vec& x) { return axpy(T{1}, x); }
  Vec& operator-=(const Vec& x) { return axpy(T{-1}, x); }

  // *this += alpha * x
  Vec& axpy(T alpha, const Vec& x)
  {
    check_compatible(x, "axpy()");
    impl().axpy(alpha, x.impl());
    return *this;
  }

  T dot(const Vec& y) const
  {
    check_compatible(y, "dot()");
    return impl().dot(y.impl());
  }

  T two_norm() const { return impl().two_norm(); }

  // y[i] = *this[i] * x[i]
  void pointwise_mult(const Vec& x, Vec& y) const
  {
    check_compatible(x, "pointwise_mult");
    check_compatible(y, "pointwise_mult");
    return impl().pointwise_mult(x.impl(), y.impl());
  }

  // *this[i] = src[i] for i < n, e.g. to copy between a vector and one on a larger index set that starts with the
  // same indices. src may have a different size than *this
  void copy_n_from(const Vec& src, Index n)
  {
    DDM_CHECK(&src != this, "vec: copy_n_from() requires distinct vectors");
    DDM_CHECK(0 <= n && n <= size() && n <= src.size(), "vec: copy_n_from() copies {} entries, but the vectors have sizes {} and {}", n, size(), src.size());
    DDM_CHECK(backend() == src.backend(), "vec: copy_n_from() backend mismatch, got {} and {}", to_string(backend()), to_string(src.backend()));
    impl().copy_n_from(src.impl(), n);
  }

  // Access to the implementation, e.g. for a Mat implementation to reach the native vector of its backend. All
  // operations on the vector go through here, so this is also where open views are detected.
  VecImpl<T>& impl()
  {
    DDM_CHECK(!view_open_, "vec: used while a host view is open");
    return checked_impl();
  }

  const VecImpl<T>& impl() const
  {
    DDM_CHECK(!view_open_, "vec: used while a host view is open");
    return checked_impl();
  }

private:
  template <class>
  friend class HostView;

  VecImpl<T>& checked_impl() const
  {
    DDM_CHECK(impl_ != nullptr, "vec: used after move");
    return *impl_;
  }

  // const, because read views can be opened on const vectors. The bookkeeping is mutable, and the data is accessed
  // through impl_, whose constness does not propagate.
  std::span<T> open_view(Access mode) const
  {
    DDM_CHECK(!view_open_, "vec: host_view() called while another host view is open");
    auto data = checked_impl().acquire_host(mode);
    DDM_CHECK(data.size() == static_cast<std::size_t>(size()), "vec: acquire_host() returned {} entries for vector of size {}", data.size(), size());
    view_open_ = true;
    return data;
  }

  void release_host_view(Access mode) const
  {
    DDM_ASSERT(view_open_, "vec: releasing a host view that is not open");
    view_open_ = false;
    checked_impl().release_host(mode);
  }

  void check_compatible(const Vec& other, std::string_view op) const
  {
    DDM_CHECK(size() == other.size(), "vec: {} expects vectors of same size, got {} and {}", op, size(), other.size());
    DDM_CHECK(backend() == other.backend(), "vec: {} backend mismatch, got {} and {}", op, to_string(backend()), to_string(other.backend()));
  }

  std::unique_ptr<VecImpl<T>> impl_;
  mutable bool view_open_ = false;
};

template <class T>
using VecRegistry = Registry<std::unique_ptr<VecImpl<T>>, Index>;

template <class T>
void register_vec(std::string name, typename VecRegistry<T>::Factory factory)
{
  VecRegistry<T>::instance().add(std::move(name), std::move(factory));
}

template <class T>
Vec<T> create_vec(const Dune::ParameterTree& config, Index n)
{
  DDM_CHECK(n >= 0, "vec: negative size {}", n);
  initialize();
  return Vec<T>(VecRegistry<T>::instance().create("vec", config, "istl", n));
}
} // namespace ddm
