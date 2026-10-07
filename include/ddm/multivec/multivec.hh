#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/vec/host_view.hh"
#include "ddm/vec/vec.hh"

#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace ddm {
template <class T>
class MultiVec;

// Scoped access to the data of a MultiVec in host memory, U is T for writable and const T for read-only views.
// Works like HostView for Vec: released on destruction or by release(), and while it is open, the multivector cannot
// be used otherwise.
template <class U>
class MultiVecHostView {
  using T = std::remove_const_t<U>;

public:
  MultiVecHostView(const MultiVecHostView&) = delete;
  MultiVecHostView& operator=(const MultiVecHostView&) = delete;
  MultiVecHostView& operator=(MultiVecHostView&&) = delete;

  MultiVecHostView(MultiVecHostView&& other) noexcept
      : mv_(std::exchange(other.mv_, nullptr))
      , data_(other.data_)
      , mode_(other.mode_)
  {
  }

  ~MultiVecHostView()
  {
    // Destructors must not throw, call release() explicitly to handle errors
    try {
      release();
    }
    catch (const std::exception& e) {
      std::cerr << "ddm: releasing a multivector host view failed: " << e.what() << std::endl;
      std::terminate();
    }
  }

  // Ends the access, the view must not be used afterwards
  void release()
  {
    if (mv_ == nullptr) return;
    std::exchange(mv_, nullptr)->release_host_view(mode_);
  }

  // All entries, column by column, e.g. to pass them to MPI
  U* data() const { return data_.data(); }
  std::size_t size() const { return data_.size(); }

  // Entry (i, j), the data is stored column by column
  U& operator()(Index i, Index j) const
  {
    DDM_ASSERT(mv_ != nullptr, "multivector host view: used after release()");
    DDM_ASSERT(0 <= i && i < mv_->rows() && 0 <= j && j < mv_->cols(), "multivector host view: entry ({}, {}) out of range for {}x{}", i, j, mv_->rows(), mv_->cols());
    return data_[static_cast<std::size_t>(j) * static_cast<std::size_t>(mv_->rows()) + static_cast<std::size_t>(i)];
  }

private:
  friend class MultiVec<T>;

  MultiVecHostView(const MultiVec<T>* mv, std::span<U> data, Access mode)
      : mv_(mv)
      , data_(data)
      , mode_(mode)
  {
  }

  const MultiVec<T>* mv_;
  std::span<U> data_;
  Access mode_;
};

// A set of vectors of the same size, the columns of a rows() x cols() matrix. Implemented by the backends, created
// through a matrix (e.g. LocalMat::create_domain_multivector()) so that it has the matrix's backend.
template <class T>
class MultiVec {
public:
  virtual ~MultiVec() { DDM_ASSERT(!view_open_, "multivector: destroyed while a host view is open"); }
  MultiVec(const MultiVec&) = delete;
  MultiVec& operator=(const MultiVec&) = delete;

  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  virtual BackendId backend() const = 0;

  MultiVecHostView<const T> host_view(read_t) const
  {
    auto data = open_view(Access::read);
    return MultiVecHostView<const T>(this, std::span<const T>(data), Access::read);
  }

  // Returns a zero-initialized multivector of the given size with the same backend
  std::unique_ptr<MultiVec> create_multivector(Index rows, Index cols) const
  {
    DDM_CHECK(rows >= 0 && cols >= 0, "multivector: create_multivector() with negative size {}x{}", rows, cols);
    return do_create_multivector(rows, cols);
  }

  // Returns a zero-initialized vector of size n with the same backend, e.g. for the coefficients of a linear
  // combination of the columns
  Vec<T> create_vector(Index n) const
  {
    DDM_CHECK(n >= 0, "multivector: create_vector() with negative size {}", n);
    return do_create_vector(n);
  }

  // c = X^H y with X = *this, i.e. c[a] is the dot product of column a of X and y (the column is conjugated, like the
  // left argument of Vec::dot)
  void dot(const Vec<T>& y, Vec<T>& c) const
  {
    DDM_CHECK(y.size() == rows() && c.size() == cols(), "multivector: dot() size mismatch, X is {}x{}, y has size {}, c has size {}", rows(), cols(), y.size(), c.size());
    DDM_CHECK(y.backend() == backend() && c.backend() == backend(), "multivector: dot() backend mismatch, X is {}, y is {}, c is {}", to_string(backend()), to_string(y.backend()),
              to_string(c.backend()));
    do_dot(y, c);
  }

  // y = X c with X = *this, i.e. the linear combination of the columns with the coefficients c
  void mv(const Vec<T>& c, Vec<T>& y) const
  {
    DDM_CHECK(c.size() == cols() && y.size() == rows(), "multivector: mv() size mismatch, X is {}x{}, c has size {}, y has size {}", rows(), cols(), c.size(), y.size());
    DDM_CHECK(c.backend() == backend() && y.backend() == backend(), "multivector: mv() backend mismatch, X is {}, c is {}, y is {}", to_string(backend()), to_string(c.backend()),
              to_string(y.backend()));
    do_mv(c, y);
  }

  // C = X^H Y with X = *this, i.e. C(a, b) is the dot product of column a of X and column b of Y. Like Vec::dot, the
  // left argument is conjugated.
  void dot(const MultiVec& Y, MultiVec& C) const
  {
    DDM_CHECK(&C != this && &C != &Y, "multivector: dot() requires C to be distinct from X and Y");
    DDM_CHECK(rows() == Y.rows() && C.rows() == cols() && C.cols() == Y.cols(), "multivector: dot() size mismatch, X is {}x{}, Y is {}x{}, C is {}x{}", rows(), cols(), Y.rows(), Y.cols(),
              C.rows(), C.cols());
    DDM_CHECK(Y.backend() == backend() && C.backend() == backend(), "multivector: dot() backend mismatch, X is {}, Y is {}, C is {}", to_string(backend()), to_string(Y.backend()),
              to_string(C.backend()));
    do_dot(Y, C);
  }

  // buffer(k, c) = X(idx[k], c) with X = *this, i.e. copies the rows idx into buffer (|idx| x cols())
  void pack_rows(std::span<const Index> idx, MultiVec& buffer) const
  {
    DDM_CHECK(&buffer != this, "multivector: pack_rows() requires a distinct buffer");
    DDM_CHECK(buffer.rows() == static_cast<Index>(idx.size()) && buffer.cols() == cols(), "multivector: pack_rows() of {} rows of a {}x{} multivector into a {}x{} buffer", idx.size(),
              rows(), cols(), buffer.rows(), buffer.cols());
    DDM_CHECK(buffer.backend() == backend(), "multivector: pack_rows() backend mismatch, multivector is {}, buffer is {}", to_string(backend()), to_string(buffer.backend()));
    for (auto i : idx) DDM_CHECK(0 <= i && i < rows(), "multivector: pack_rows() row {} out of range for {} rows", i, rows());
    do_pack_rows(idx, buffer);
  }

  // X(idx[k], first_col + c) = buffer(k, c) with X = *this, i.e. the inverse of pack_rows() into the columns
  // [first_col, first_col + buffer.cols())
  void unpack_rows(const MultiVec& buffer, std::span<const Index> idx, Index first_col)
  {
    DDM_CHECK(&buffer != this, "multivector: unpack_rows() requires a distinct buffer");
    DDM_CHECK(buffer.rows() == static_cast<Index>(idx.size()) && 0 <= first_col && first_col + buffer.cols() <= cols(),
              "multivector: unpack_rows() of a {}x{} buffer into {} rows, starting at column {} of a {}x{} multivector", buffer.rows(), buffer.cols(), idx.size(), first_col, rows(), cols());
    DDM_CHECK(buffer.backend() == backend(), "multivector: unpack_rows() backend mismatch, multivector is {}, buffer is {}", to_string(backend()), to_string(buffer.backend()));
    for (auto i : idx) DDM_CHECK(0 <= i && i < rows(), "multivector: unpack_rows() row {} out of range for {} rows", i, rows());
    do_unpack_rows(buffer, idx, first_col);
  }

  MultiVecHostView<T> host_view(write_t) { return MultiVecHostView<T>(this, open_view(Access::write), Access::write); }
  MultiVecHostView<T> host_view(read_write_t) { return MultiVecHostView<T>(this, open_view(Access::read_write), Access::read_write); }

protected:
  MultiVec(Index rows, Index cols)
      : rows_(rows)
      , cols_(cols)
  {
    DDM_CHECK(rows >= 0 && cols >= 0, "multivector: negative size {}x{}", rows, cols);
  }

private:
  template <class>
  friend class MultiVecHostView;

  virtual std::unique_ptr<MultiVec> do_create_multivector(Index rows, Index cols) const = 0;
  virtual void do_dot(const MultiVec& Y, MultiVec& C) const = 0;
  virtual Vec<T> do_create_vector(Index n) const = 0;
  virtual void do_dot(const Vec<T>& y, Vec<T>& c) const = 0;
  virtual void do_mv(const Vec<T>& c, Vec<T>& y) const = 0;
  virtual void do_pack_rows(std::span<const Index> idx, MultiVec& buffer) const = 0;
  virtual void do_unpack_rows(const MultiVec& buffer, std::span<const Index> idx, Index first_col) = 0;

  // Returns the data in host memory, column by column. For Access::write, the contents may be arbitrary. Every
  // acquire_host() is followed by a release_host() with the same mode, and at most one access is active at a time.
  virtual std::span<T> acquire_host(Access mode) = 0;

  // Ends the access started by acquire_host(). For Access::write and Access::read_write, the data written to the span
  // must be part of the multivector afterwards.
  virtual void release_host(Access mode) = 0;

  // const, because read views can be opened on const multivectors (see Vec)
  std::span<T> open_view(Access mode) const
  {
    DDM_CHECK(!view_open_, "multivector: host_view() called while another host view is open");
    auto data = const_cast<MultiVec*>(this)->acquire_host(mode);
    DDM_CHECK(data.size() == static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_), "multivector: acquire_host() returned {} entries for a {}x{} multivector", data.size(), rows_,
              cols_);
    view_open_ = true;
    return data;
  }

  void release_host_view(Access mode) const
  {
    DDM_ASSERT(view_open_, "multivector: releasing a host view that is not open");
    view_open_ = false;
    const_cast<MultiVec*>(this)->release_host(mode);
  }

  Index rows_;
  Index cols_;
  mutable bool view_open_ = false;
};
} // namespace ddm
