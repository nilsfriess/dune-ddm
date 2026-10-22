#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/registry.hh"
#include "ddm/vec/host_view.hh"

#include <complex>
#include <cstddef>
#include <dune/common/parametertree.hh>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace ddm {
// The host view of a vector is the host view of its multivector
template <class U>
using HostView = MultiVecHostView<U>;

// Vector with value semantics: copies are deep copies. This is what ISTL's solvers expect from their vector type.
// Copying allocates, assigning does not.
//
// A vector is a multivector with one column (see multivec()), which the backends implement. It either owns its
// multivector, or it is a view of a column of another multivector (see column()): writes to the vector change
// the column, and the vector must not outlive the multivector. Assignment never changes this, it always copies the
// values. Copies are owning vectors.
//
// The data can be accessed through host_view(). While a view is open, the vector cannot be used otherwise, and only one
// view can be open at a time.
template <class T>
class Vec {
public:
  using field_type = T;
  using real_type = decltype(std::real(std::declval<T>()));

  explicit Vec(std::unique_ptr<MultiVec<T>> mv)
      : mv_(std::move(mv))
  {
    DDM_CHECK(mv_ != nullptr, "vec: multivector is nullptr");
    DDM_CHECK(mv_->cols() == 1, "vec: a vector is a multivector with one column, got {} columns", mv_->cols());
  }

  Vec(const Vec& other)
      : mv_(other.multivec().create_multivector(other.size(), 1))
  {
    mv_->copy_from(other.multivec());
  }

  // The host views refer to the multivector, which is not moved, so vectors with an open view can be moved
  Vec(Vec&& other) noexcept = default;

  ~Vec() = default;

  Vec& operator=(const Vec& other)
  {
    if (this == &other) return *this;
    if (!mv_) mv_ = other.multivec().create_multivector(other.size(), 1); // *this was moved from
    mv_->copy_from(other.multivec());
    return *this;
  }

  // Takes over the multivector of other if both own theirs. Otherwise the values are copied, so that assignment does
  // not turn a view into an owning vector or vice versa
  Vec& operator=(Vec&& other)
  {
    if ((mv_ && mv_->is_view()) || (other.mv_ && other.mv_->is_view())) return *this = std::as_const(other);
    mv_ = std::move(other.mv_);
    return *this;
  }

  // Sets all entries to value
  Vec& operator=(T value)
  {
    fill(value);
    return *this;
  }

  // size() and backend() are allowed while a view is open
  Index size() const
  {
    return checked_mv().rows();
  }
  BackendId backend() const
  {
    return checked_mv().backend();
  }

  HostView<const T> host_view(read_t) const
  {
    return std::as_const(checked_mv()).host_view(read);
  }
  HostView<T> host_view(write_t)
  {
    return checked_mv().host_view(write);
  }
  HostView<T> host_view(read_write_t)
  {
    return checked_mv().host_view(read_write);
  }

  void zero()
  {
    fill(T{0});
  }
  void fill(T value)
  {
    multivec().fill(value);
  }

  Vec& operator*=(T alpha)
  {
    multivec().scale(alpha);
    return *this;
  }

  Vec& operator+=(const Vec& x)
  {
    return axpy(T{1}, x);
  }
  Vec& operator-=(const Vec& x)
  {
    return axpy(T{-1}, x);
  }

  // *this += alpha * x
  Vec& axpy(T alpha, const Vec& x)
  {
    multivec().axpy(alpha, x.multivec());
    return *this;
  }

  T dot(const Vec& y) const
  {
    T result{};
    multivec().column_dots(y.multivec(), std::span(&result, 1));
    return result;
  }

  real_type two_norm() const
  {
    real_type result{};
    multivec().column_norms(std::span(&result, 1));
    return result;
  }

  // y[i] = *this[i] * x[i]
  void pointwise_mult(const Vec& x, Vec& y) const
  {
    multivec().pointwise_mult(x.multivec(), y.multivec());
  }

  // *this[i] = src[i] for i < n, e.g. to copy between a vector and one on a larger index set that starts with the
  // same indices. src may have a different size than *this
  void copy_n_from(const Vec& src, Index n)
  {
    multivec().copy_rows_from(src.multivec(), n);
  }

  // The multivector with one column that holds the data, e.g. for a Mat implementation to reach the data in its
  // backend. Fails while a host view is open.
  MultiVec<T>& multivec()
  {
    auto& mv = checked_mv();
    mv.check_usable("vec: multivec()");
    return mv;
  }

  const MultiVec<T>& multivec() const
  {
    const auto& mv = checked_mv();
    mv.check_usable("vec: multivec()");
    return mv;
  }

private:
  MultiVec<T>& checked_mv() const
  {
    DDM_CHECK(mv_ != nullptr, "vec: used after move");
    return *mv_;
  }

  std::unique_ptr<MultiVec<T>> mv_; ///< one column
};

// A vector that is a view of column j of X: writes to it change the column, and it must not outlive X. Assigning a
// vector to it copies the vector into the column, assigning it to a vector copies the column (see Vec)
template <class T>
Vec<T> column(MultiVec<T>& X, Index j)
{
  return Vec<T>(X.columns(j, 1));
}

template <class T>
class ConstColumn;

// A read-only view of column j of X, see ConstColumn
template <class T>
ConstColumn<T> column(const MultiVec<T>& X, Index j);

// A read-only view of a column of a multivector, returned by column() for a const multivector. It can be passed
// wherever a const Vec<T>& is expected (use *c or c-> otherwise), and like a Vec view it must not outlive the
// multivector. It can neither be copied nor moved, so the view inside never ends up in a non-const Vec: copying the
// vector gives an owning copy.
template <class T>
class ConstColumn {
public:
  ConstColumn(const ConstColumn&) = delete;
  ConstColumn& operator=(const ConstColumn&) = delete;

  operator const Vec<T>&() const { return vec_; }
  const Vec<T>& operator*() const { return vec_; }
  const Vec<T>* operator->() const { return &vec_; }

private:
  friend ConstColumn column<T>(const MultiVec<T>& X, Index j);

  explicit ConstColumn(Vec<T> vec)
      : vec_(std::move(vec))
  {
  }

  Vec<T> vec_;
};

template <class T>
ConstColumn<T> column(const MultiVec<T>& X, Index j)
{
  // The view is only accessible as const Vec<T>& through ConstColumn, so dropping the const is safe
  return ConstColumn<T>(Vec<T>(const_cast<MultiVec<T>&>(X).columns(j, 1)));
}

// Returns a zero-initialized vector of size n with the same backend as X, e.g. for the coefficients of a linear
// combination of the columns of X
template <class T>
Vec<T> create_vector(const MultiVec<T>& X, Index n)
{
  return Vec<T>(X.create_multivector(n, 1));
}

// Factories return the multivector with one column of the vector
template <class T>
using VecRegistry = Registry<std::unique_ptr<MultiVec<T>>, Index>;

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
