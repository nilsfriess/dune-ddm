#pragma once

#include "backend/backend.hh"

#include <cstddef>
#include <cstdint>
#include <dune/common/exceptions.hh>
#include <utility>
#include <vector>

namespace ddm {
/** @brief A column-major multivector (a tall and skinny dense matrix) with backend-allocated memory
 *
 */
template <class Scalar, class Backend, class Index = std::uint_least32_t>
class MultiVector {
public:
  MultiVector(typename Backend::context_type ctx, Index rows, Index cols)
      : ctx_(std::move(ctx))
      , rows_(rows)
      , cols_(cols)
      , data_(Backend::template make_buffer<Scalar>(ctx_, rows_ * cols_))
  {
  }

  const Scalar* data() const { return data_.data(); }
  Scalar* data() { return data_.data(); }

  Index rows() const { return rows_; }
  Index cols() const { return cols_; }

  Scalar* col(Index k) { return data() + k * rows(); }
  const Scalar* col(Index k) const { return data() + k * rows(); }

  /// Sets every entry to zero
  void zero() { Backend::zero(ctx_, data(), static_cast<std::size_t>(rows_) * cols_); }

  /// The context (e.g. the sycl::queue) this multivector's memory lives on
  typename Backend::context_type& context() const { return ctx_; }

private:
  mutable typename Backend::context_type ctx_;

  Index rows_;
  Index cols_;
  typename Backend::template buffer_type<Scalar> data_;
};

namespace detail {
/// The scalar stored in one vector entry: the entry itself for scalar entries, the first
/// component for block entries (e.g. Dune::FieldVector). Returns a reference, so it can be
/// read from and assigned to.
template <class Entry>
decltype(auto) entry_scalar(Entry&& e)
{
  if constexpr (requires { e[0]; }) return e[0]; // block entry (e.g. Dune::FieldVector)
  else return std::forward<Entry>(e);            // plain scalar entry
}
} // namespace detail

/** @brief Copies the entries of a vector into one column of a multivector.
 *
 *  The bridge between the vector interface (entries may be blocks like Dune::FieldVector) and the
 *  scalar columns the multivector primitives work on. Supported combinations:
 *  - device vector -> multivector on the same backend: raw copy (one scalar per entry)
 *  - host vector -> multivector on any backend: element-wise conversion (blocks or scalars), then
 *    a host-to-device copy if the multivector lives on a device
 *  Only \p v.size() entries are written; the rest of the column is untouched.
 */
template <class Vec, class Scalar, class Backend, class Index>
void copy_into_column(const Vec& v, MultiVector<Scalar, Backend, Index>& mv, Index col)
{
  using VectorBackend = backend::backend_of_t<Vec>;
  if (v.size() > mv.rows()) DUNE_THROW(Dune::InvalidStateException, "The vector has " << v.size() << " entries, but the multivector column only has " << mv.rows());

  if constexpr (VectorBackend::is_device) {
    static_assert(std::is_same_v<VectorBackend, Backend>, "A device vector can only be copied into a multivector on the same backend");
    static_assert(std::is_same_v<backend::element_of_t<Vec>, Scalar>, "The raw device copy requires the vector to store one scalar per entry");
    Backend::copy_n(mv.context(), v.data(), v.size(), mv.col(col));
  }
  else if constexpr (Backend::is_device) {
    // Unpack the entries on the host, then upload them in one copy
    std::vector<Scalar> buf(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) buf[i] = detail::entry_scalar(v[i]);
    Backend::copy_from_host(mv.context(), buf.data(), mv.col(col), buf.size());
  }
  else {
    static_assert(std::is_same_v<VectorBackend, Backend>, "The vector and the multivector must live on the same backend");
    for (std::size_t i = 0; i < v.size(); ++i) mv.col(col)[i] = detail::entry_scalar(v[i]);
  }
}

/** @brief Copies one column of a multivector into a vector. Inverse of copy_into_column().
 *
 *  Supported combinations:
 *  - multivector -> vector on the same device backend: raw copy (one scalar per entry)
 *  - multivector on any backend -> host vector: a device-to-host copy if the multivector lives on
 *    a device, then an element-wise conversion (blocks or scalars)
 *  Copies \p v.size() entries; the vector must not be longer than the column.
 */
template <class Scalar, class Backend, class Index, class Vec>
void copy_from_column(const MultiVector<Scalar, Backend, Index>& mv, Index col, Vec& v)
{
  using VectorBackend = backend::backend_of_t<Vec>;
  if (v.size() > mv.rows()) DUNE_THROW(Dune::InvalidStateException, "The vector has " << v.size() << " entries, but the multivector column only has " << mv.rows());

  if constexpr (VectorBackend::is_device) {
    static_assert(std::is_same_v<VectorBackend, Backend>, "A device vector can only be filled from a multivector on the same backend");
    static_assert(std::is_same_v<backend::element_of_t<Vec>, Scalar>, "The raw device copy requires the vector to store one scalar per entry");
    Backend::copy_n(mv.context(), mv.col(col), v.size(), v.data());
  }
  else if constexpr (Backend::is_device) {
    // Download the column in one copy, then unpack the entries on the host
    std::vector<Scalar> buf(v.size());
    Backend::copy_to_host(mv.context(), mv.col(col), buf.data(), buf.size());
    for (std::size_t i = 0; i < v.size(); ++i) detail::entry_scalar(v[i]) = buf[i];
  }
  else {
    static_assert(std::is_same_v<VectorBackend, Backend>, "The vector and the multivector must live on the same backend");
    for (std::size_t i = 0; i < v.size(); ++i) detail::entry_scalar(v[i]) = mv.col(col)[i];
  }
}

} // namespace ddm
