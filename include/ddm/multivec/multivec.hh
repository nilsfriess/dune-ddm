#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/communication_pattern.hh"
#include "ddm/index.hh"
#include "ddm/vec/exchange_plan.hh"
#include "ddm/vec/host_view.hh"

#include <complex>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace ddm {
template <class T>
class MultiVec;

// Scoped access to the data of a MultiVec (or Vec) in host memory, U is T for writable and const T for read-only
// views. Released (i.e. written back if needed) on destruction or by an explicit call to release(). While a view is
// open, the multivector (and every other multivector that shares its storage, see MultiVec::columns()) cannot be used
// otherwise.
template <class U>
class MultiVecHostView {
  using T = std::remove_const_t<U>;

public:
  using value_type = T;
  using element_type = U;

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
    // Destructors must not throw, call release() explicitly to handle errors (e.g. a failing copy back to the device)
    try {
      release();
    }
    catch (const std::exception& e) {
      std::cerr << "ddm: releasing a host view failed: " << e.what() << std::endl;
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
  U* data() const
  {
    return data_.data();
  }
  std::size_t size() const
  {
    return data_.size();
  }
  auto begin() const
  {
    return data_.begin();
  }
  auto end() const
  {
    return data_.end();
  }
  std::span<U> span() const
  {
    return data_;
  }

  // Entry i of all entries, column by column (for a vector: entry i)
  U& operator[](std::size_t i) const
  {
    DDM_ASSERT(mv_ != nullptr, "host view: used after release()");
    DDM_ASSERT(i < data_.size(), "host view: index {} out of range for size {}", i, data_.size());
    return data_[i];
  }

  // Entry (i, j), the data is stored column by column
  U& operator()(Index i, Index j) const
  {
    DDM_ASSERT(mv_ != nullptr, "host view: used after release()");
    DDM_ASSERT(0 <= i && i < mv_->rows() && 0 <= j && j < mv_->cols(), "host view: entry ({}, {}) out of range for {}x{}", i, j, mv_->rows(), mv_->cols());
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

// A set of vectors of the same size, the columns of a rows() x cols() matrix, stored column by column. Implemented by
// the backends, created through a matrix (e.g. LocalMat::create_domain_multivector()) so that it has the matrix's
// backend. Small dense matrices, e.g. the coefficients of a linear combination of the columns (see dot() and mult()),
// are multivectors as well, so that they stay in the memory space of the backend. A vector (Vec) is a multivector with
// one column, which gives it value semantics, so the backends implement only this class.
//
// columns() returns a view of a range of columns: a multivector that does not own its data but shares the storage of
// the multivector it was created from (the owner). The columns of a view are contiguous, so a view is stored exactly
// like an owning multivector and can be used wherever a multivector is expected (column() in vec.hh returns a single
// column as a Vec). A view must not outlive its owner. The owner and all its views count as one storage: only one host view can
// be open on it at a time, and while it is open, none of them can be used otherwise. Operations whose output overlaps
// an input (see overlaps()) are rejected, except for operations that work entry by entry if the output and the input
// are the same columns.
template <class T>
class MultiVec {
public:
  using real_type = decltype(std::real(std::declval<T>()));

  virtual ~MultiVec()
  {
    DDM_ASSERT(root().host_view_holder_ != this, "multivector: destroyed while a host view is open");
    if (owner_ != nullptr) --owner_->live_views_;
    else DDM_ASSERT(live_views_ == 0, "multivector: destroyed while {} views of its columns are alive", live_views_);
  }

  MultiVec(const MultiVec&) = delete;
  MultiVec& operator=(const MultiVec&) = delete;

  Index rows() const
  {
    return rows_;
  }
  Index cols() const
  {
    return cols_;
  }
  virtual BackendId backend() const = 0;

  // Whether this is a view of the columns of another multivector, see columns()
  bool is_view() const
  {
    return owner_ != nullptr;
  }

  // Whether *this and other share storage, i.e. they are (views of) the same multivector and their columns intersect
  bool overlaps(const MultiVec& other) const
  {
    if (&root() != &other.root() || cols_ == 0 || other.cols_ == 0) return false;
    return col_offset_ < other.col_offset_ + other.cols_ && other.col_offset_ < col_offset_ + cols_;
  }

  // A view of the columns [first, first + count), see the class comment
  std::unique_ptr<MultiVec> columns(Index first, Index count)
  {
    check_columns(first, count);
    return checked_view(do_columns(first, count), first, count);
  }

  std::unique_ptr<const MultiVec> columns(Index first, Index count) const
  {
    check_columns(first, count);
    return checked_view(do_columns(first, count), first, count);
  }

  MultiVecHostView<const T> host_view(read_t) const
  {
    auto data = open_view(Access::read);
    return MultiVecHostView<const T>(this, std::span<const T>(data), Access::read);
  }

  MultiVecHostView<T> host_view(write_t)
  {
    return MultiVecHostView<T>(this, open_view(Access::write), Access::write);
  }
  MultiVecHostView<T> host_view(read_write_t)
  {
    return MultiVecHostView<T>(this, open_view(Access::read_write), Access::read_write);
  }

  // Returns a zero-initialized multivector of the given size with the same backend
  std::unique_ptr<MultiVec> create_multivector(Index rows, Index cols) const
  {
    DDM_CHECK(rows >= 0 && cols >= 0, "multivector: create_multivector() with negative size {}x{}", rows, cols);
    return do_create_multivector(rows, cols);
  }

  // Sets all entries to value
  void fill(T value)
  {
    check_usable("fill()");
    do_fill(value);
  }

  // *this = alpha * *this
  void scale(T alpha)
  {
    check_usable("scale()");
    do_scale(alpha);
  }

  // *this = X
  void copy_from(const MultiVec& X)
  {
    check_same_shape(X, "copy_from()");
    if (!same_columns(X)) do_copy_from(X);
  }

  // *this = *this + alpha * X
  void axpy(T alpha, const MultiVec& X)
  {
    check_same_shape(X, "axpy()");
    do_axpy(alpha, X);
  }

  // Y(i, j) = X(i, j) * Z(i, j) with X = *this, i.e. the entrywise product. Y may be the same columns as X or Z
  void pointwise_mult(const MultiVec& Z, MultiVec& Y) const
  {
    check_same_shape(Z, "pointwise_mult()", true);
    check_same_shape(Y, "pointwise_mult()");
    Y.check_same_shape(Z, "pointwise_mult()");
    do_pointwise_mult(Z, Y);
  }

  // result[j] = x_j^H y_j, the dot products of the columns of X = *this and Y (the columns of X are conjugated, like
  // the left argument of Vec::dot). X and Y may overlap.
  void column_dots(const MultiVec& Y, std::span<T> result) const
  {
    check_same_shape(Y, "column_dots()", true);
    DDM_CHECK(result.size() == static_cast<std::size_t>(cols()), "multivector: column_dots() of {} columns into {} results", cols(), result.size());
    do_column_dots(Y, result);
  }

  // result[j] is the Euclidean norm of column j
  void column_norms(std::span<real_type> result) const
  {
    check_usable("column_norms()");
    DDM_CHECK(result.size() == static_cast<std::size_t>(cols()), "multivector: column_norms() of {} columns into {} results", cols(), result.size());
    do_column_norms(result);
  }

  // X(i, j) = src(i, j) for i < n with X = *this, e.g. to copy between multivectors on index sets of which one starts
  // with the indices of the other. src may have a different number of rows than *this, n is at most the smaller one.
  void copy_rows_from(const MultiVec& src, Index n)
  {
    check_usable("copy_rows_from()");
    src.check_usable("copy_rows_from()");
    DDM_CHECK(!overlaps(src), "multivector: copy_rows_from() requires distinct multivectors");
    DDM_CHECK(cols() == src.cols() && 0 <= n && n <= rows() && n <= src.rows(), "multivector: copy_rows_from() of {} rows of a {}x{} multivector into a {}x{} multivector", n, src.rows(), src.cols(),
              rows(), cols());
    DDM_CHECK(backend() == src.backend(), "multivector: copy_rows_from() backend mismatch, {} and {}", to_string(backend()), to_string(src.backend()));
    do_copy_rows_from(src, n);
  }

  // C = X^H Y with X = *this, i.e. C(a, b) is the dot product of column a of X and column b of Y. Like Vec::dot, the
  // left argument is conjugated. X and Y may overlap (e.g. X^H X), C must not overlap either of them.
  void dot(const MultiVec& Y, MultiVec& C) const
  {
    check_usable("dot()");
    Y.check_usable("dot()");
    C.check_usable("dot()");
    DDM_CHECK(!C.overlaps(*this) && !C.overlaps(Y), "multivector: dot() requires C to be distinct from X and Y");
    DDM_CHECK(rows() == Y.rows() && C.rows() == cols() && C.cols() == Y.cols(), "multivector: dot() size mismatch, X is {}x{}, Y is {}x{}, C is {}x{}", rows(), cols(), Y.rows(), Y.cols(), C.rows(),
              C.cols());
    DDM_CHECK(Y.backend() == backend() && C.backend() == backend(), "multivector: dot() backend mismatch, X is {}, Y is {}, C is {}", to_string(backend()), to_string(Y.backend()),
              to_string(C.backend()));
    do_dot(Y, C);
  }

  // Y = alpha * X M + beta * Y with X = *this, where M is a cols() x Y.cols() matrix of coefficients. As in BLAS, Y is
  // not read if beta is zero. Y must not overlap X or M.
  void mult(const MultiVec& M, MultiVec& Y, T alpha = T{1}, T beta = T{0}) const
  {
    check_usable("mult()");
    M.check_usable("mult()");
    Y.check_usable("mult()");
    DDM_CHECK(!Y.overlaps(*this) && !Y.overlaps(M), "multivector: mult() requires Y to be distinct from X and M");
    DDM_CHECK(M.rows() == cols() && Y.rows() == rows() && Y.cols() == M.cols(), "multivector: mult() size mismatch, X is {}x{}, M is {}x{}, Y is {}x{}", rows(), cols(), M.rows(), M.cols(), Y.rows(),
              Y.cols());
    DDM_CHECK(M.backend() == backend() && Y.backend() == backend(), "multivector: mult() backend mismatch, X is {}, M is {}, Y is {}", to_string(backend()), to_string(M.backend()),
              to_string(Y.backend()));
    do_mult(M, Y, alpha, beta);
  }

  // buffer(k, c) = X(idx[k], c) with X = *this, i.e. copies the rows idx into buffer (|idx| x cols())
  void pack_rows(std::span<const Index> idx, MultiVec& buffer) const
  {
    check_usable("pack_rows()");
    buffer.check_usable("pack_rows()");
    DDM_CHECK(!buffer.overlaps(*this), "multivector: pack_rows() requires a distinct buffer");
    DDM_CHECK(buffer.rows() == static_cast<Index>(idx.size()) && buffer.cols() == cols(), "multivector: pack_rows() of {} rows of a {}x{} multivector into a {}x{} buffer", idx.size(), rows(), cols(),
              buffer.rows(), buffer.cols());
    DDM_CHECK(buffer.backend() == backend(), "multivector: pack_rows() backend mismatch, multivector is {}, buffer is {}", to_string(backend()), to_string(buffer.backend()));
    for (auto i : idx) DDM_CHECK(0 <= i && i < rows(), "multivector: pack_rows() row {} out of range for {} rows", i, rows());
    do_pack_rows(idx, buffer);
  }

  // X(idx[k], first_col + c) = buffer(k, c) with X = *this, i.e. the inverse of pack_rows() into the columns
  // [first_col, first_col + buffer.cols())
  void unpack_rows(const MultiVec& buffer, std::span<const Index> idx, Index first_col)
  {
    check_usable("unpack_rows()");
    buffer.check_usable("unpack_rows()");
    DDM_CHECK(!buffer.overlaps(*this), "multivector: unpack_rows() requires a distinct buffer");
    DDM_CHECK(buffer.rows() == static_cast<Index>(idx.size()) && 0 <= first_col && first_col + buffer.cols() <= cols(),
              "multivector: unpack_rows() of a {}x{} buffer into {} rows, starting at column {} of a {}x{} multivector", buffer.rows(), buffer.cols(), idx.size(), first_col, rows(), cols());
    DDM_CHECK(buffer.backend() == backend(), "multivector: unpack_rows() backend mismatch, multivector is {}, buffer is {}", to_string(backend()), to_string(buffer.backend()));
    for (auto i : idx) DDM_CHECK(0 <= i && i < rows(), "multivector: unpack_rows() row {} out of range for {} rows", i, rows());
    do_unpack_rows(buffer, idx, first_col);
  }

  // Multiplies row i of every column by d(i, 0), i.e. *this = diag(d) * *this for a multivector d with one column (e.g.
  // the multivector of a Vec)
  void scale_rows(const MultiVec& d)
  {
    check_usable("scale_rows()");
    d.check_usable("scale_rows()");
    DDM_CHECK(!overlaps(d), "multivector: scale_rows() requires d to be distinct from the multivector");
    DDM_CHECK(backend() == d.backend(), "multivector: scale_rows() backend mismatch, multivector is {}, d is {}", to_string(backend()), to_string(d.backend()));
    DDM_CHECK(d.cols() == 1 && rows() == d.rows(), "multivector: scale_rows() size mismatch, multivector has {} rows, d is {}x{}", rows(), d.rows(), d.cols());
    do_scale_rows(d);
  }

  // Creates the buffers to exchange the entries at the indices in idxs with other ranks, for a multivector with one
  // column (the multivector of a Vec)
  std::unique_ptr<ExchangePlan<T>> make_exchange_plan(const CommunicationPattern::IndexMap& idxs) const
  {
    DDM_CHECK(cols() == 1, "multivector: make_exchange_plan() requires a single column, got {}", cols());
    return do_make_exchange_plan(idxs);
  }

  // Throws if the data of the multivector cannot be used, i.e. if a host view of its storage is open. For
  // implementations of operations that take multivectors as arguments (e.g. LocalMat::spmm())
  void check_usable(std::string_view op) const
  {
    DDM_CHECK(root().host_view_holder_ == nullptr, "multivector: {} called while a host view of the multivector is open", op);
  }

protected:
  MultiVec(Index rows, Index cols)
      : rows_(rows)
      , cols_(cols)
  {
    DDM_CHECK(rows >= 0 && cols >= 0, "multivector: negative size {}x{}", rows, cols);
  }

  // For the implementations of do_columns(): a view of the columns [first, first + count) of parent
  MultiVec(const MultiVec& parent, Index first, Index count)
      : rows_(parent.rows_)
      , cols_(count)
      , owner_(&parent.root())
      , col_offset_(parent.col_offset_ + first)
  {
    ++owner_->live_views_;
  }

private:
  template <class>
  friend class MultiVecHostView;

  // Returns a view of the columns [first, first + count), created with the view constructor of MultiVec. The range is
  // valid. const, because views can be created of const multivectors, columns() restores the constness.
  virtual std::unique_ptr<MultiVec> do_columns(Index first, Index count) const = 0;
  virtual std::unique_ptr<MultiVec> do_create_multivector(Index rows, Index cols) const = 0;
  virtual void do_fill(T value) = 0;
  virtual void do_scale(T alpha) = 0;
  virtual void do_copy_from(const MultiVec& X) = 0;
  virtual void do_axpy(T alpha, const MultiVec& X) = 0;
  virtual void do_pointwise_mult(const MultiVec& Z, MultiVec& Y) const = 0;
  virtual void do_column_dots(const MultiVec& Y, std::span<T> result) const = 0;
  virtual void do_column_norms(std::span<real_type> result) const = 0;
  virtual void do_copy_rows_from(const MultiVec& src, Index n) = 0;
  virtual void do_dot(const MultiVec& Y, MultiVec& C) const = 0;
  virtual void do_mult(const MultiVec& M, MultiVec& Y, T alpha, T beta) const = 0;
  virtual void do_pack_rows(std::span<const Index> idx, MultiVec& buffer) const = 0;
  virtual void do_unpack_rows(const MultiVec& buffer, std::span<const Index> idx, Index first_col) = 0;
  virtual void do_scale_rows(const MultiVec& d) = 0;

  // The default works through host views, backends that keep their data elsewhere can override this to avoid the
  // copies
  virtual std::unique_ptr<ExchangePlan<T>> do_make_exchange_plan(const CommunicationPattern::IndexMap& idxs) const
  {
    return std::make_unique<HostExchangePlan<T>>(idxs);
  }

  // Returns the data in host memory, column by column. For Access::write, the contents may be arbitrary. Every
  // acquire_host() is followed by a release_host() with the same mode, and at most one access to the storage (shared
  // by an owner and its views) is active at a time.
  virtual std::span<T> acquire_host(Access mode) = 0;

  // Ends the access started by acquire_host(). For Access::write and Access::read_write, the data written to the span
  // must be part of the multivector afterwards.
  virtual void release_host(Access mode) = 0;

  // The multivector that owns the storage
  const MultiVec& root() const
  {
    return owner_ != nullptr ? *owner_ : *this;
  }

  // Whether *this and other are the same columns of the same storage
  bool same_columns(const MultiVec& other) const
  {
    return &root() == &other.root() && col_offset_ == other.col_offset_ && cols_ == other.cols_;
  }

  void check_columns(Index first, Index count) const
  {
    DDM_CHECK(0 <= first && 0 <= count && first + count <= cols_, "multivector: columns() of the range [{}, {}) out of range for {} columns", first, first + count, cols_);
  }

  template <class Ptr>
  Ptr checked_view(Ptr view, Index first, Index count) const
  {
    DDM_ASSERT(view != nullptr && view->owner_ == &root() && view->col_offset_ == col_offset_ + first && view->rows_ == rows_ && view->cols_ == count,
               "multivector: do_columns() did not return a view of the requested columns");
    return view;
  }

  // Checks the arguments of an operation that works entry by entry on *this and X: same size and backend, no host
  // view open, and X must not overlap *this unless it is the same columns. If X is only read (may_overlap), any
  // overlap is fine.
  void check_same_shape(const MultiVec& X, std::string_view op, bool may_overlap = false) const
  {
    check_usable(op);
    X.check_usable(op);
    DDM_CHECK(may_overlap || !overlaps(X) || same_columns(X), "multivector: {} requires distinct multivectors", op);
    DDM_CHECK(rows() == X.rows() && cols() == X.cols(), "multivector: {} size mismatch, {}x{} and {}x{}", op, rows(), cols(), X.rows(), X.cols());
    DDM_CHECK(backend() == X.backend(), "multivector: {} backend mismatch, {} and {}", op, to_string(backend()), to_string(X.backend()));
  }

  // const, because read views can be opened on const multivectors
  std::span<T> open_view(Access mode) const
  {
    DDM_CHECK(root().host_view_holder_ == nullptr, "multivector: host_view() called while another host view of the multivector is open");
    auto data = const_cast<MultiVec*>(this)->acquire_host(mode);
    DDM_CHECK(data.size() == static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_), "multivector: acquire_host() returned {} entries for a {}x{} multivector", data.size(), rows_, cols_);
    root().host_view_holder_ = this;
    return data;
  }

  void release_host_view(Access mode) const
  {
    DDM_ASSERT(root().host_view_holder_ == this, "multivector: releasing a host view that is not open");
    root().host_view_holder_ = nullptr;
    const_cast<MultiVec*>(this)->release_host(mode);
  }

  Index rows_;
  Index cols_;

  const MultiVec* owner_ = nullptr; ///< the owner of the storage if this is a view, nullptr otherwise
  Index col_offset_ = 0;            ///< the first column of this multivector in the storage of the owner

  // Bookkeeping of the owner, for itself and all its views
  mutable Index live_views_ = 0;
  mutable const MultiVec* host_view_holder_ = nullptr; ///< the multivector whose host view is open, if any
};
} // namespace ddm
