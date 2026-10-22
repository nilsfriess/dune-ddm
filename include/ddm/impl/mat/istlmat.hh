#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/impl/multivec/istlmultivec.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/pattern.hh"

#include <algorithm>
#include <cstddef>
#include <dune/istl/experimental/bcrsmatrix.hh>
#include <memory>
#include <span>
#include <type_traits>

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
  using Native = Dune::Experimental::BCRSMatrix<T, std::make_unsigned_t<Index>>;

  explicit IstlMat(const Dune::ParameterTree& config, const Pattern& pattern)
      : LocalMat<T>(config, pattern)
      , A_(pattern.ranges())
  {
  }

  Index rows() const override { return static_cast<Index>(A_.N()); }
  Index cols() const override { return static_cast<Index>(A_.M()); }
  BackendId backend() const override { return BackendId::istl; }

  Vec<T> create_domain_vector() const override { return Vec<T>(std::make_unique<IstlMultiVec<T>>(cols(), 1)); }
  Vec<T> create_range_vector() const override { return Vec<T>(std::make_unique<IstlMultiVec<T>>(rows(), 1)); }

  const Native& native() const { return A_; }

private:
  void do_add_values(std::span<const Index> rows, std::span<const Index> cols, std::span<const T> vals) override
  {
    for (std::size_t r = 0; r < rows.size(); ++r) {
      auto row = A_[static_cast<std::size_t>(rows[r])];
      for (std::size_t c = 0; c < cols.size(); ++c) {
        auto it = row.find(static_cast<std::size_t>(cols[c]));
        DDM_CHECK(it != row.end(), "istl mat: add_values() entry ({}, {}) not in pattern", rows[r], cols[c]);
        *it += vals[r + c * rows.size()];
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

  void do_zero_rows_and_cols(std::span<const Index> idx, T diag) override
  {
    std::vector<bool> zero(A_.N(), false);
    for (auto i : idx) zero[i] = true;

    for (auto ri = A_.begin(); ri != A_.end(); ++ri)
      for (auto ci = ri->begin(); ci != ri->end(); ++ci)
        if (zero[ri.index()] || zero[ci.index()]) *ci = T{0};

    for (auto i : idx) {
      auto row = A_[i];
      auto diag_it = row.find(i);
      DDM_CHECK(diag_it != row.end(), "istl mat: zero_rows_and_cols() diagonal entry ({}, {}) not in pattern", i, i);
      *diag_it = diag;
    }
  }

  void do_mv(const Vec<T>& x, Vec<T>& y) const override
  {
    const auto x_data = as_istl(x).data();
    auto y_data = as_istl(y).data();
    for (std::size_t i = 0; i < A_.N(); ++i) {
      const auto row = A_[i];
      T y_i{0};
      for (auto ci = row.begin(); ci != row.end(); ++ci) y_i += *ci * x_data[ci.index()];
      y_data[i] = y_i;
    }
  }

  void do_usmv(T alpha, const Vec<T>& x, Vec<T>& y) const override
  {
    const auto x_data = as_istl(x).data();
    auto y_data = as_istl(y).data();
    for (auto ri = A_.begin(); ri != A_.end(); ++ri) {
      auto& y_i = y_data[ri.index()];
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) y_i += alpha * *ci * x_data[ci.index()];
    }
  }

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
    diag.zero();
    auto istl_diag = as_istl(diag).data();
    for (std::size_t r = 0; r < A_.N(); ++r) {
      auto row = A_[r];
      auto it = row.find(r);
      if (it != row.end()) istl_diag[r] = *it;
    }
  }

  std::unique_ptr<MultiVec<T>> do_create_domain_multivector(Index m) const override { return std::make_unique<IstlMultiVec<T>>(this->cols(), m); }
  std::unique_ptr<MultiVec<T>> do_create_range_multivector(Index m) const override { return std::make_unique<IstlMultiVec<T>>(this->rows(), m); }
  std::shared_ptr<LocalMat<T>> do_create_like(const Dune::ParameterTree& config, const Pattern& pattern) const override { return std::make_shared<IstlMat<T>>(config, pattern); }
  std::shared_ptr<LocalMat<T>> do_clone() const override { return std::make_shared<IstlMat<T>>(*this); }

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

  void do_scale(const Vec<T>& d) override
  {
    const auto d_istl = as_istl(d).data();
    for (auto ri = A_.begin(); ri != A_.end(); ++ri)
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) *ci *= d_istl[ri.index()] * d_istl[ci.index()];
  }

  Native A_;
};
} // namespace ddm
