#include "dune/ddm/helpers.hh"

#include <cstdint>
#include <dune/common/exceptions.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/solvers.hh>
#include <iostream>
#include <memory>
#include <span>
#include <sycl/sycl.hpp>
#include <utility>

template <class Scalar, class Index>
class SYCLVec;

template <class Vec>
class Jacobi : public Dune::Preconditioner<Vec, Vec> {
public:
  template <class Mat>
  Jacobi(const Mat& A)
      : diag(A.getdiag())
  {
    auto* d = diag.data;
    A.queue().parallel_for(sycl::range<1>(A.N()), [=](auto idx) { d[idx] = 1. / d[idx]; });
  }

  void pre(Vec&, Vec&) override {}
  void post(Vec&) override {}

  void apply(Vec& x, const Vec& d) override
  {
    DDM_CHECK(x.size() == d.size(), "x and d must have the same size");
    DDM_CHECK(x.size() == diag.size(), "x and diag must have the same size");

    auto* dd = diag.data;
    auto* bd = d.data;
    auto* xd = x.data;

    x.queue().parallel_for(sycl::range<1>(diag.size()), [=](auto idx) { xd[idx] = bd[idx] * dd[idx]; });
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::sequential; }

private:
  Vec diag;
};

template <class Scalar, class Index = std::uint_least32_t>
class SYCLMat {
public:
  using block_type = Scalar;
  using index_type = Index;
  using allocator_type = std::allocator<Scalar>;

  SYCLMat(const sycl::queue& q_, Index rows_, Index cols_, std::span<const Index> host_r, std::span<const Index> host_c, std::span<const block_type> host_a)
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

  SYCLMat(const SYCLMat&) = delete;
  SYCLMat& operator=(const SYCLMat&) = delete;

  ~SYCLMat()
  {
    q.wait(); // no kernel may still be reading r/c/a when we free them
    sycl::free(r, q);
    sycl::free(c, q);
    sycl::free(a, q);
  }

  Index N() const { return rows; }

  template <class Vec>
  void usmv(Scalar alpha, const Vec& x, Vec& y) const
  {
    auto* y_data = y.data;
    const auto* const x_data = x.data;

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

  template <class Vec>
  void mv(const Vec& x, Vec& y) const
  {
    auto* y_data = y.data;
    const auto* const x_data = x.data;

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

  SYCLVec<Scalar, Index> getdiag() const
  {
    DDM_CHECK(rows == cols, "getdiag only for square matrices");
    SYCLVec<Scalar, Index> diag(q, rows);

    const auto* rr = r;
    const auto* cc = c;
    const auto* aa = a;
    // Must be a plain pointer: capturing `diag` itself would copy the whole SYCLVec
    // (and thus submit to `q`) from inside the submission of this kernel.
    auto* dd = diag.data;

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

template <class Scalar, class Index = std::uint_least32_t>
struct SYCLVec {
  using field_type = Scalar;

  SYCLVec(const sycl::queue q_, Index n_)
      : q(q_)
      , n(n_)
      , data(sycl::malloc_device<field_type>(n, q))
      , red_dev(sycl::malloc_device<field_type>(1, q))
      , red_host(sycl::malloc_host<field_type>(1, q))
  {
  }

  SYCLVec(const SYCLVec& other)
      : q(other.q)
      , n(other.n)
      , data(sycl::malloc_device<field_type>(other.n, q))
      , red_dev(sycl::malloc_device<field_type>(1, q))
      , red_host(sycl::malloc_host<field_type>(1, q))
  {
    q.memcpy(data, other.data, n * sizeof(field_type));
  }

  SYCLVec(SYCLVec&& other) noexcept
      : q(other.q)
      , n(std::exchange(other.n, 0))
      , data(std::exchange(other.data, nullptr))
      , red_dev(std::exchange(other.red_dev, nullptr))
      , red_host(std::exchange(other.red_host, nullptr))
  {
  }

  SYCLVec& operator=(const SYCLVec& other)
  {
    if (this == &other) return *this;

    if (n != other.n) {
      release();
      q = other.q;
      n = other.n;
      data = sycl::malloc_device<field_type>(n, q);
      red_dev = sycl::malloc_device<field_type>(1, q);
      red_host = sycl::malloc_host<field_type>(1, q);
    }
    q.memcpy(data, other.data, n * sizeof(field_type));
    return *this;
  }

  SYCLVec& operator=(SYCLVec&& other) noexcept
  {
    if (this == &other) return *this;

    release();
    q = other.q;
    n = std::exchange(other.n, 0);
    data = std::exchange(other.data, nullptr);
    red_dev = std::exchange(other.red_dev, nullptr);
    red_host = std::exchange(other.red_host, nullptr);
    return *this;
  }

  ~SYCLVec() { release(); }

  Index size() const { return n; }

  field_type dot(const SYCLVec& y) const
  {
    const auto* v = data;
    const auto* w = y.data;
    auto* sum = red_dev;

    q.submit([&](sycl::handler& h) {
      auto red = sycl::reduction(sum, sycl::plus<>(), sycl::property::reduction::initialize_to_identity{});
      h.parallel_for(sycl::range<1>(n), red, [=](auto idx, auto& tmp) { tmp += v[idx] * w[idx]; });
    });
    q.memcpy(red_host, sum, sizeof(field_type)).wait();

    return *red_host;
  }

  field_type two_norm() const
  {
    using std::sqrt;
    return sqrt(two_norm2());
  }

  field_type two_norm2() const
  {
    const auto* v = data;
    auto* sum = red_dev;

    q.submit([&](sycl::handler& h) {
      // Without initialize_to_identity the (uninitialised) contents of *sum are used as
      // the initial value of the reduction.
      auto red = sycl::reduction(sum, sycl::plus<>(), sycl::property::reduction::initialize_to_identity{});
      h.parallel_for(sycl::range<1>(n), red, [=](auto idx, auto& tmp) { tmp += v[idx] * v[idx]; });
    });
    q.memcpy(red_host, sum, sizeof(field_type)).wait();

    return *red_host;
  }

  SYCLVec& operator=(field_type a)
  {
    q.fill(data, a, n);
    return *this;
  }

  SYCLVec& operator*=(field_type a)
  {
    if (a == field_type(1)) return *this;

    auto* v = data;
    q.parallel_for(sycl::range<1>(n), [=](auto idx) { v[idx] *= a; });
    return *this;
  }

  SYCLVec& operator+=(const SYCLVec& other)
  {
    DDM_CHECK(n == other.n, "Size mismatch in operator+= ({} vs {})", n, other.n);
    auto* v = data;
    const auto* w = other.data;
    q.parallel_for(sycl::range<1>(n), [=](auto idx) { v[idx] += w[idx]; });
    return *this;
  }

  SYCLVec& operator-=(const SYCLVec& other)
  {
    DDM_CHECK(n == other.n, "Size mismatch in operator-= ({} vs {})", n, other.n);
    auto* v = data;
    const auto* w = other.data;
    q.parallel_for(sycl::range<1>(n), [=](auto idx) { v[idx] -= w[idx]; });
    return *this;
  }

  void axpy(field_type a, const SYCLVec& y)
  {
    DDM_CHECK(n == y.n, "Size mismatch in axpy ({} vs {})", n, y.n);
    auto* v = data;
    const auto* w = y.data;
    q.parallel_for(sycl::range<1>(n), [=](auto idx) { v[idx] += a * w[idx]; });
  }

  sycl::queue queue() const { return q; }

private:
  friend class SYCLMat<Scalar, Index>;
  friend class Jacobi<SYCLVec>;

  void release()
  {
    if (data == nullptr) return;
    q.wait(); // make sure all kernels are done when we try to free the memory
    sycl::free(data, q);
    sycl::free(red_dev, q);
    sycl::free(red_host, q);
  }

  mutable sycl::queue q;
  Index n;
  field_type* data;

  // Scratch for the parallel reductions in dot/two_norm
  field_type* red_dev;
  field_type* red_host;
};

int main(int argc, char** argv)
{
  sycl::queue q{sycl::property::queue::in_order{}};

  Dune::ParameterTree ptree;
  ptree["reduction"] = "1e-8";
  ptree["restart"] = "30";
  ptree["maxit"] = "10000";
  ptree["verbose"] = "4";
  Dune::ParameterTreeParser parser;
  parser.readOptions(argc, argv, ptree);

  using Vec = SYCLVec<double>;
  using Mat = SYCLMat<double>;
  using Index = typename Mat::index_type;

  using Op = Dune::MatrixAdapter<Mat, Vec, Vec>;
  Index gridsize = 512;
  auto n = gridsize * gridsize;
  auto nnz = 5 * n;

  std::shared_ptr<Mat> A;
  {
    std::vector<Index> r;
    std::vector<Index> c;
    std::vector<double> a;
    r.reserve(n + 1);
    c.reserve(nnz);
    a.reserve(nnz);
    r.push_back(0);

    const auto to_linear = [&](auto i, auto j) { return i * gridsize + j; };

    for (Index i = 0; i < gridsize; ++i) {
      for (Index j = 0; j < gridsize; ++j) {
        if (i > 0) {
          c.push_back(to_linear(i - 1, j));
          a.push_back(-1);
        }

        if (i < gridsize - 1) {
          c.push_back(to_linear(i + 1, j));
          a.push_back(-1);
        }

        if (j > 0) {
          c.push_back(to_linear(i, j - 1));
          a.push_back(-1);
        }

        if (j < gridsize - 1) {
          c.push_back(to_linear(i, j + 1));
          a.push_back(-1);
        }

        c.push_back(to_linear(i, j));
        a.push_back(4);

        r.push_back(c.size());
      }
    }

    A = std::make_shared<Mat>(q, n, n, std::span{r.data(), r.size()}, std::span{c.data(), c.size()}, std::span{a.data(), a.size()});
  }
  
  auto op = std::make_shared<Op>(A);
  auto prec = std::make_shared<Jacobi<Vec>>(*A);

  // malloc_device does not zero-initialise, so both vectors must be set explicitly
  Vec x(q, n);
  x = 0.0;
  Vec b(q, n);
  b = 1.0;

  // Dune::RestartedGMResSolver<Vec> solver(op, prec, ptree);
  Dune::CGSolver<Vec> solver(op, prec, ptree);
  Dune::InverseOperatorResult res;
  solver.apply(x, b, res);
}
