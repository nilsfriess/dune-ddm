#pragma once

#include "dune/ddm/helpers.hh"

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <memory>
#include <span>
#include <sycl/sycl.hpp>

namespace ddm::Sycl {
template <class Scalar, class Index>
class Vec;

template <class Scalar, class Index = std::uint_least32_t>
class Mat {
public:
  using block_type = Scalar;
  using index_type = Index;
  using allocator_type = std::allocator<Scalar>;

  Mat(const sycl::queue& q_, Index rows_, Index cols_, std::span<const Index> host_r, std::span<const Index> host_c, std::span<const block_type> host_a)
      : q(q_)
      , rows(rows_)
      , cols(cols_)
      , r(sycl::malloc_device<Index>(host_r.size(), q))
      , c(sycl::malloc_device<Index>(host_c.size(), q))
      , a(sycl::malloc_device<block_type>(host_a.size(), q))
  {
    DDM_CHECK(host_c.size() == host_a.size(), "Invalid CSR data (column index array size {} and data array size {} do not match)", host_c.size(), host_a.size());
    DDM_CHECK(host_r.size() == rows + 1, "Invalid CSR data (row offsets array size {} does not match the provided number of rows {})", host_r.size(), rows + 1);

    q.memcpy(r, host_r.data(), host_r.size_bytes());
    q.memcpy(c, host_c.data(), host_c.size_bytes());
    q.memcpy(a, host_a.data(), host_a.size_bytes());
    // The copies are asynchronous and read from the caller's host buffers, so we must
    // not return before they have completed (the caller can then free the host buffers)
    q.wait();
  }

  Mat(const Mat&) = delete;
  Mat& operator=(const Mat&) = delete;

  Mat(Mat&& other) noexcept
      : q(other.q)
      , rows(other.rows)
      , cols(other.cols)
      , r(other.r)
      , c(other.c)
      , a(other.a)
  {
    other.rows = 0;
    other.cols = 0;
    other.r = nullptr;
    other.c = nullptr;
    other.a = nullptr;
  }

  Mat& operator=(Mat&& other) noexcept
  {
    if (this != &other) {
      q = other.q;
      rows = other.rows;
      cols = other.cols;
      r = other.r;
      c = other.c;
      a = other.a;

      other.rows = 0;
      other.cols = 0;
      other.r = nullptr;
      other.c = nullptr;
      other.a = nullptr;
    }
    return *this;
  }

  ~Mat()
  {
    if (r || c || a) q.wait(); // no kernel may still be reading r/c/a when we free them
    if (r) sycl::free(r, q);
    if (c) sycl::free(c, q);
    if (a) sycl::free(a, q);
  }

  template <int d, class Allocator>
  static Mat from_bcrs(sycl::queue& q, const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, d, d>, Allocator>& A)
  {
    Index nnz = 0;
    const auto [fr, fc] = flatMatrixForEach(A, [&](auto&&, auto&&, auto&&) { ++nnz; });
    Index N = fr;

    std::vector<Index> row_ptr(N + 1, 0);
    std::vector<Index> col_ind(nnz, 0);
    std::vector<Scalar> values(nnz, 0);

    // Count number of entries per row and accumulate
    flatMatrixForEach(A, [&](auto&&, auto&& row, auto&&) { row_ptr[row + 1] += 1; });
    for (Index i = 0; i < N; ++i) row_ptr[i + 1] += row_ptr[i];

    assert(row_ptr[N] == nnz);

    // Now fill the other two arrays
    std::vector<Index> row_pos(N, 0); // position counter in each row
    flatMatrixForEach(A, [&](auto&& entry, auto&& row, auto&& col) {
      auto row_start = row_ptr[row];

      col_ind[row_start + row_pos[row]] = col;
      values[row_start + row_pos[row]] = entry;

      row_pos[row] += 1;
    });

    return Mat(q, N, N, {row_ptr.data(), row_ptr.size()}, {col_ind.data(), col_ind.size()}, {values.data(), values.size()});
  }

  Index N() const { return rows; }

  void usmv(Scalar alpha, const Vec<Scalar, Index>& x, Vec<Scalar, Index>& y) const
  {
    auto* y_data = y.data();
    const auto* const x_data = x.data();

    const auto* rr = r;
    const auto* cc = c;
    const auto* aa = a;

    q.parallel_for(sycl::range<1>(rows), [=](auto idx) {
      const auto row = idx[0];
      const auto row_start = rr[row];
      const auto row_end = rr[row + 1];

      block_type sum{0};
      for (auto k = row_start; k < row_end; ++k) sum += alpha * aa[k] * x_data[cc[k]];
      y_data[row] += sum;
    });
  }

  void mv(const Vec<Scalar, Index>& x, Vec<Scalar, Index>& y) const
  {
    auto* y_data = y.data();
    const auto* const x_data = x.data();

    const auto* rr = r;
    const auto* cc = c;
    const auto* aa = a;

    q.parallel_for(sycl::range<1>(rows), [=](auto idx) {
      const auto row = idx[0];
      const auto row_start = rr[row];
      const auto row_end = rr[row + 1];

      block_type sum{0};
      for (auto k = row_start; k < row_end; ++k) sum += aa[k] * x_data[cc[k]];
      y_data[row] = sum;
    });
  }

  Vec<Scalar, Index> getdiag() const
  {
    DDM_CHECK(rows == cols, "getdiag only for square matrices");
    Vec<Scalar, Index> diag(q, rows);

    const auto* rr = r;
    const auto* cc = c;
    const auto* aa = a;
    // Must be a plain pointer: capturing `diag` itself would copy the whole Vec
    // (and thus submit to `q`) from inside the submission of this kernel.
    auto* dd = diag.data();

    q.parallel_for(sycl::range<1>(rows), [=](auto idx) {
       const auto row = idx[0];
       const auto row_start = rr[row];
       const auto row_end = rr[row + 1];

       block_type d{0};
       for (auto k = row_start; k < row_end; ++k)
         if (cc[k] == row) d = aa[k];
       dd[row] = d;
     }).wait();
    return diag;
  }

  sycl::queue queue() const { return q; }

private:
  mutable sycl::queue q; // q.parallel_for is not const, but this->mv needs to be const

  Index rows;
  Index cols;

  Index* r;
  Index* c;
  block_type* a;
};
} // namespace ddm::Sycl
