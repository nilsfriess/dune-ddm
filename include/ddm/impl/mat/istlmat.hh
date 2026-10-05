#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/impl/vec/istlvec.hh"
#include "ddm/index.hh"
#include "ddm/mat/mat.hh"
#include "ddm/mat/pattern.hh"

#include <cstddef>
#include <dune/istl/ibcrsmatrix.hh>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace ddm {
template <class T>
class IstlMat;

template <class T>
IstlMat<T>& as_istl(Mat<T>& A)
{
  return static_cast<IstlMat<T>&>(A);
}

template <class T>
const IstlMat<T>& as_istl(const Mat<T>& A)
{
  return static_cast<const IstlMat<T>&>(A);
}

template <class T>
class IstlMat final : public Mat<T> {
public:
  using Native = Dune::IBCRSMatrix<T, std::make_unsigned_t<Index>>;

  explicit IstlMat(const Pattern& pattern)
      : A_(pattern.ranges())
  {
  }

  Index rows() const override { return static_cast<Index>(A_.N()); }
  Index cols() const override { return static_cast<Index>(A_.M()); }
  BackendId backend() const override { return BackendId::istl; }

  Vec<T> create_domain_vector() const override { return Vec<T>(std::make_unique<IstlVec<T>>(cols())); }
  Vec<T> create_range_vector() const override { return Vec<T>(std::make_unique<IstlVec<T>>(rows())); }

  const Native& native() const { return A_; }

private:
  void do_add_values(std::span<const Index> rows, std::span<const Index> cols, std::span<const T> vals) override
  {
    for (std::size_t r = 0; r < rows.size(); ++r) {
      auto row = A_[static_cast<std::size_t>(rows[r])];
      for (std::size_t c = 0; c < cols.size(); ++c) {
        auto it = row.find(static_cast<std::size_t>(cols[c]));
        DDM_CHECK(it != row.end(), "istl mat: add_values() entry ({}, {}) not in pattern", rows[r], cols[c]);
        *it += vals[r * cols.size() + c];
      }
    }
  }

  // Assembling a host ISTL matrix is a no-op
  void do_assemble() override {}

  void do_zero_rows(std::span<const Index> rows, T diag) override
  {
    for (auto i : rows) {
      auto row = A_[static_cast<std::size_t>(i)];
      auto diag_it = row.find(static_cast<std::size_t>(i));
      DDM_CHECK(diag_it != row.end(), "istl mat: zero_rows() diagonal entry ({}, {}) not in pattern", i, i);
      for (auto it = row.begin(); it != row.end(); ++it) *it = T{0};
      *diag_it = diag;
    }
  }

  void do_apply(const Vec<T>& x, Vec<T>& y) const override { A_.mv(as_istl(x).native(), as_istl(y).native()); }
  void do_applyscaleadd(T alpha, const Vec<T>& x, Vec<T>& y) const override { A_.usmv(alpha, as_istl(x).native(), as_istl(y).native()); }

  void do_get_diag(Vec<T>& diag) const override
  {
    auto& istl_diag = as_istl(diag).native();
    diag.zero();
    for (std::size_t r = 0; r < A_.N(); ++r) {
      auto row = A_[r];
      auto it = row.find(r);
      if (it != row.end()) istl_diag[r] = *it;
    }
  }

  Native A_;
};
} // namespace ddm
