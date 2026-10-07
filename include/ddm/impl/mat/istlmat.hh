#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/impl/multivec/istlmultivec.hh"
#include "ddm/impl/vec/istlvec.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/pattern.hh"

#include <algorithm>
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
IstlMat<T>& as_istl(LocalMat<T>& A)
{
  return static_cast<IstlMat<T>&>(A);
}

template <class T>
const IstlMat<T>& as_istl(const LocalMat<T>& A)
{
  return static_cast<const IstlMat<T>&>(A);
}

template <class T>
class IstlMat final : public LocalMat<T> {
public:
  using Native = Dune::IBCRSMatrix<T, std::make_unsigned_t<Index>>;

  explicit IstlMat(const Pattern& pattern)
      : LocalMat<T>(pattern)
      , A_(pattern.ranges())
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

  void do_mv(const Vec<T>& x, Vec<T>& y) const override { A_.mv(as_istl(x).native(), as_istl(y).native()); }
  void do_usmv(T alpha, const Vec<T>& x, Vec<T>& y) const override { A_.usmv(alpha, as_istl(x).native(), as_istl(y).native()); }

  void do_spmm(const MultiVec<T>& X, MultiVec<T>& Y) const override
  {
    const auto x = as_istl(X).data();
    auto y = as_istl(Y).data();
    const std::size_t x_rows = X.rows();
    const std::size_t y_rows = Y.rows();
    const std::size_t m = X.cols();

    std::fill(y.begin(), y.end(), T{0});
    for (auto ri = A_.begin(); ri != A_.end(); ++ri)
      for (auto ci = ri->begin(); ci != ri->end(); ++ci)
        for (std::size_t j = 0; j < m; ++j) y[j * y_rows + ri.index()] += *ci * x[j * x_rows + ci.index()];
  }

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

  std::unique_ptr<MultiVec<T>> do_create_domain_multivector(Index m) const override { return std::make_unique<IstlMultiVec<T>>(this->cols(), m); }
  std::unique_ptr<MultiVec<T>> do_create_range_multivector(Index m) const override { return std::make_unique<IstlMultiVec<T>>(this->rows(), m); }

  HostCsr<T> do_host_csr() const override
  {
    HostCsr<T> csr;
    csr.row_ptr.reserve(A_.N() + 1);
    csr.row_ptr.push_back(0);
    for (auto ri = A_.begin(); ri != A_.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        csr.cols.push_back(static_cast<Index>(ci.index()));
        csr.values.push_back(*ci);
      }
      csr.row_ptr.push_back(static_cast<Index>(csr.cols.size()));
    }
    return csr;
  }

  void do_info() const override
  {
    logger::info("ISTL matrix of size {}x{}", rows(), cols());
    logger::info("Nonzero entries {}", A_.nonzeroes());
  }

  Native A_;
};
} // namespace ddm
