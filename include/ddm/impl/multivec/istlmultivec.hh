#pragma once

#include "ddm/backend_id.hh"
#include "ddm/impl/vec/istlvec.hh"
#include "ddm/index.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/vec/host_view.hh"

#include <cstddef>
#include <dune/common/math.hh>
#include <memory>
#include <span>
#include <vector>

namespace ddm {
template <class T>
class IstlMultiVec;

// Only valid for multivectors with backend() == BackendId::istl, see as_istl() for vectors
template <class T>
IstlMultiVec<T>& as_istl(MultiVec<T>& X)
{
  return static_cast<IstlMultiVec<T>&>(X);
}

template <class T>
const IstlMultiVec<T>& as_istl(const MultiVec<T>& X)
{
  return static_cast<const IstlMultiVec<T>&>(X);
}

template <class T>
class IstlMultiVec final : public MultiVec<T> {
public:
  IstlMultiVec(Index rows, Index cols)
      : MultiVec<T>(rows, cols)
      , data_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), T(0))
  {
  }

  BackendId backend() const override { return BackendId::istl; }

  // The data, column by column
  std::span<T> data() { return data_; }
  std::span<const T> data() const { return data_; }

private:
  std::unique_ptr<MultiVec<T>> do_create_multivector(Index rows, Index cols) const override { return std::make_unique<IstlMultiVec>(rows, cols); }

  void do_dot(const MultiVec<T>& Y, MultiVec<T>& C) const override
  {
    const auto y = as_istl(Y).data();
    auto c = as_istl(C).data();
    const std::size_t n = this->rows();
    const std::size_t m_x = this->cols();
    const std::size_t m_y = Y.cols();

    for (std::size_t b = 0; b < m_y; ++b) {
      for (std::size_t a = 0; a < m_x; ++a) {
        T sum{0};
        for (std::size_t i = 0; i < n; ++i) sum += Dune::conjugateComplex(data_[a * n + i]) * y[b * n + i];
        c[b * m_x + a] = sum;
      }
    }
  }

  Vec<T> do_create_vector(Index n) const override { return Vec<T>(std::make_unique<IstlVec<T>>(n)); }

  void do_dot(const Vec<T>& y, Vec<T>& c) const override
  {
    const auto& y_istl = as_istl(y).native();
    auto& c_istl = as_istl(c).native();
    const std::size_t n = this->rows();
    const std::size_t m = this->cols();

    for (std::size_t a = 0; a < m; ++a) {
      T sum{0};
      for (std::size_t i = 0; i < n; ++i) sum += Dune::conjugateComplex(data_[a * n + i]) * y_istl[i];
      c_istl[a] = sum;
    }
  }

  void do_mv(const Vec<T>& c, Vec<T>& y) const override
  {
    const auto& c_istl = as_istl(c).native();
    auto& y_istl = as_istl(y).native();
    const std::size_t n = this->rows();
    const std::size_t m = this->cols();

    y_istl = T{0};
    for (std::size_t a = 0; a < m; ++a)
      for (std::size_t i = 0; i < n; ++i) y_istl[i] += data_[a * n + i] * c_istl[a];
  }

  void do_pack_rows(std::span<const Index> idx, MultiVec<T>& buffer) const override
  {
    auto b = as_istl(buffer).data();
    const std::size_t n = this->rows();
    const std::size_t k_max = idx.size();
    const std::size_t m = this->cols();

    for (std::size_t c = 0; c < m; ++c)
      for (std::size_t k = 0; k < k_max; ++k) b[c * k_max + k] = data_[c * n + idx[k]];
  }

  void do_unpack_rows(const MultiVec<T>& buffer, std::span<const Index> idx, Index first_col) override
  {
    const auto b = as_istl(buffer).data();
    const std::size_t n = this->rows();
    const std::size_t k_max = idx.size();
    const std::size_t m = buffer.cols();
    const std::size_t first = first_col;

    for (std::size_t c = 0; c < m; ++c)
      for (std::size_t k = 0; k < k_max; ++k) data_[(first + c) * n + idx[k]] = b[c * k_max + k];
  }

  // The data is in host memory already, so there is nothing to copy
  std::span<T> acquire_host(Access /*mode*/) override { return data_; }
  void release_host(Access /*mode*/) override {}

  // TODO(20261007-071012): Use an AlignedAllocator to allocate the MultiVec data
  std::vector<T> data_; ///< column by column
};
} // namespace ddm
