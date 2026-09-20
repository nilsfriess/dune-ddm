#include "dune/ddm/backend/host/backend.hh"
#include "dune/ddm/backend/sycl/backend.hh"
#include "dune/ddm/multivector.hh"
#include "dune/ddm/sycl/vec.hh"
#include "tests/test_utils.hh"

#include <cmath>
#include <cstdint>
#include <dune/common/fvector.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/istl/bvector.hh>
#include <sycl/sycl.hpp>
#include <vector>

namespace {
template <class Scalar, class Backend>
void run(Dune::TestSuite& t, typename Backend::context_type ctx)
{
  std::uint_least32_t rows = 1000;
  std::uint_least32_t cols = 10;

  ddm::MultiVector<Scalar, Backend> v(ctx, rows, cols);
  t.check(v.rows() == rows, "multivector-rows") << "Multivector didn't return the correct number of rows";
  t.check(v.cols() == cols, "multivector-columns") << "Multivector didn't return the correct number of columns";
  t.check(v.data() != nullptr, "multivector-data") << "Multivector data() is nullptr";

  // Test the const overloads
  ([&](const ddm::MultiVector<Scalar, Backend>& vc) {
    t.check(v.rows() == rows, "multivector-rows") << "Multivector didn't return the correct number of rows";
    t.check(v.cols() == cols, "multivector-columns") << "Multivector didn't return the correct number of columns";
    t.check(v.data() != nullptr, "multivector-data") << "Multivector data() is nullptr";
  })(v);
}

using HostVector = Dune::BlockVector<Dune::FieldVector<double, 1>>;

HostVector filled(std::size_t n, double shift = 0.0)
{
  HostVector v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = std::sin(static_cast<double>(i)) + shift;
  return v;
}

template <class V1, class V2>
void check_equal(Dune::TestSuite& t, const V1& a, const V2& b, const char* label)
{
  t.check(a.size() == b.size(), "round trip preserves the size") << label;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    t.check(std::abs(a[i] - b[i]) <= 1e-13, "round trip preserves the entries") << label << ": i=" << i << " a=" << a[i] << " b=" << b[i];
}

/// Host vector -> column -> host vector, for a host multivector
void run_host_copy_test(Dune::TestSuite& t)
{
  constexpr std::uint_least32_t rows = 64;
  ddm::MultiVector<double, ddm::backend::HostBackend> mv({}, rows, 3);
  mv.zero();

  const auto v = filled(rows);
  ddm::copy_into_column(v, mv, 1u);
  HostVector w(rows);
  ddm::copy_from_column(mv, 1u, w);
  check_equal(t, v, w, "host block vector round trip");

  // A plain scalar host vector works the same way
  std::vector<double> s(rows);
  for (std::size_t i = 0; i < rows; ++i) s[i] = static_cast<double>(i) * 0.5;
  ddm::copy_into_column(s, mv, 2u);
  std::vector<double> s2(rows);
  ddm::copy_from_column(mv, 2u, s2);
  check_equal(t, s, s2, "std::vector round trip");

  // A shorter vector writes only its leading entries; the rest of the column is untouched
  const auto short_v = filled(10, 42.0);
  ddm::copy_into_column(short_v, mv, 0u);
  HostVector column(rows);
  ddm::copy_from_column(mv, 0u, column);
  HostVector leading(10);
  for (std::size_t i = 0; i < 10; ++i) leading[i] = column[i];
  check_equal(t, short_v, leading, "short vector writes its leading entries");
  for (std::size_t i = 10; i < rows; ++i) t.check(column[i] == 0.0, "entries past the short vector are untouched") << "i=" << i;
}

/// Host vector -> device column -> host vector, plus the device-to-device raw copy via Sycl::Vec
void run_sycl_copy_test(Dune::TestSuite& t, sycl::queue& q)
{
  constexpr std::uint_least32_t rows = 64;
  ddm::MultiVector<double, ddm::backend::SyclBackend> mv(q, rows, 3);
  mv.zero();

  const auto v = filled(rows);
  ddm::copy_into_column(v, mv, 0u);
  HostVector w(rows);
  ddm::copy_from_column(mv, 0u, w);
  check_equal(t, v, w, "host -> device -> host round trip");

  // A shorter vector writes only its leading entries; the rest of the column stays zero
  const auto short_v = filled(10, 42.0);
  ddm::copy_into_column(short_v, mv, 1u);
  HostVector column(rows);
  ddm::copy_from_column(mv, 1u, column);
  HostVector leading(10);
  for (std::size_t i = 0; i < 10; ++i) leading[i] = column[i];
  check_equal(t, short_v, leading, "short vector writes its leading entries");
  for (std::size_t i = 10; i < rows; ++i) t.check(column[i] == 0.0, "entries past the short vector are untouched") << "i=" << i;

  // Device vector -> device column -> device vector is the raw one-scalar-per-entry copy
  const auto v2 = filled(rows, 7.0);
  std::vector<double> v2_flat(rows);
  for (std::size_t i = 0; i < rows; ++i) v2_flat[i] = v2[i][0];
  const auto dv = ddm::Sycl::Vec<double>::from_host_vector(q, v2_flat);
  ddm::copy_into_column(dv, mv, 2u);
  ddm::Sycl::Vec<double> dw(q, rows);
  ddm::copy_from_column(mv, 2u, dw);
  check_equal(t, v2_flat, dw.to_host_vector(), "device vector round trip");
}
} // namespace

int main(int argc, char** argv)
{
  Dune::MPIHelper::instance(argc, argv);
  return ddmtest::runParallelTest("test_multivector", [&](Dune::TestSuite& t) {
    run<double, ddm::backend::HostBackend>(t, {});
    run<float, ddm::backend::HostBackend>(t, {});
    run_host_copy_test(t);

    sycl::queue q{sycl::property::queue::in_order{}};
    run_sycl_copy_test(t, q);
  });
}
