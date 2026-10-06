#pragma once

#include "ddm/backend_id.hh"
#include "ddm/index.hh"
#include "ddm/vec/vec.hh"

#include <cstddef>
#include <dune/istl/bvector.hh>
#include <memory>
#include <span>

namespace ddm {
template <class T>
class IstlVec;

// Only valid for vectors with backend() == BackendId::istl. Vec<T> checks that both operands of a binary operation
// have the same backend, and Mat/Prec/Solver check their arguments before calling into an istl implementation.
template <class T>
IstlVec<T>& as_istl(VecImpl<T>& v)
{
  return static_cast<IstlVec<T>&>(v);
}

template <class T>
const IstlVec<T>& as_istl(const VecImpl<T>& v)
{
  return static_cast<const IstlVec<T>&>(v);
}

template <class T>
IstlVec<T>& as_istl(Vec<T>& v)
{
  return as_istl(v.impl());
}

template <class T>
const IstlVec<T>& as_istl(const Vec<T>& v)
{
  return as_istl(v.impl());
}

template <class T>
class IstlVec final : public VecImpl<T> {
public:
  using Native = Dune::BlockVector<T>;

  explicit IstlVec(Index n)
      : v_(static_cast<std::size_t>(n))
  {
    v_ = T{0};
  }

  Index size() const override { return static_cast<Index>(v_.size()); }
  BackendId backend() const override { return BackendId::istl; }

  std::unique_ptr<VecImpl<T>> clone() const override
  {
    auto copy = std::make_unique<IstlVec>(size());
    copy->v_ = v_;
    return copy;
  }

  void copy_from(const VecImpl<T>& other) override { v_ = as_istl(other).v_; }

  void fill(T value) override { v_ = value; }
  void scale(T alpha) override { v_ *= alpha; }
  void axpy(T alpha, const VecImpl<T>& x) override { v_.axpy(alpha, as_istl(x).v_); }

  T dot(const VecImpl<T>& y) const override { return v_.dot(as_istl(y).v_); }
  T two_norm() const override { return v_.two_norm(); }

  void pointwise_mult(const VecImpl<T>& x, VecImpl<T>& y) const override
  {
    const auto& x_istl = as_istl(x).v_;
    auto& y_istl = as_istl(y).v_;

    for (Index i = 0; i < size(); ++i) y_istl[i] = v_[i] * x_istl[i];
  }

  void copy_n_from(const VecImpl<T>& src, Index n) override
  {
    const auto& src_istl = as_istl(src).v_;
    for (Index i = 0; i < n; ++i) v_[i] = src_istl[i];
  }

  // The data is in host memory already, so there is nothing to copy
  std::span<T> acquire_host(Access /*mode*/) override { return {v_.data(), v_.size()}; }
  void release_host(Access /*mode*/) override {}

  Native& native() { return v_; }
  const Native& native() const { return v_; }

private:
  Native v_;
};
} // namespace ddm
