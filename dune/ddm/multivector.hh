#pragma once

#include <cstdint>
#include <utility>

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

private:
  typename Backend::context_type ctx_;

  Index rows_;
  Index cols_;
  typename Backend::template buffer_type<Scalar> data_;
};

} // namespace ddm
