// Unit tests of Vec: value semantics, the arithmetic ISTL's solvers rely on and host views, checked against Eigen for
// float, double and complex<double>

#include "test_helpers.hh"

#include "ddm/vec/vec.hh"

#include <complex>
#include <dune/common/exceptions.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/test/testsuite.hh>
#include <format>
#include <string_view>
#include <utility>

namespace {
using namespace ddm;
using namespace ddm::test;
using Dune::TestSuite;
using Error = Dune::InvalidStateException; // thrown by DDM_CHECK

template <class T>
T scalar(double re, double im)
{
  if constexpr (std::is_same_v<T, Real<T>>) return T(re);
  else return T(re, im);
}

template <class T>
std::string name(std::string_view test)
{
  return std::format("{}<{}>", test, type_name<T>());
}

template <class T>
Vec<T> make_vec(Index n)
{
  return create_vec<T>(Dune::ParameterTree{}, n);
}

template <class T>
Vec<T> random_vec(Index n, unsigned seed)
{
  auto v = make_vec<T>(n);
  from_eigen(v, Vector<T>(random_matrix<T>(n, 1, seed)));
  return v;
}

template <class T>
TestSuite test_value_semantics()
{
  TestSuite t(name<T>("value semantics"));
  const Vector<T> a = random_matrix<T>(6, 1, 1);

  auto v = make_vec<T>(6);
  t.check(v.size() == 6 && v.backend() == BackendId::istl, "size and backend");
  t.check(to_eigen(v) == Vector<T>::Zero(6), "zero-initialized");
  from_eigen(v, a);

  Vec<T> w(v);
  w *= T(2);
  t.check(to_eigen(v) == a && to_eigen(w) == T(2) * a, "copies are deep");

  auto u = make_vec<T>(6);
  u = v;
  v *= T(3);
  t.check(to_eigen(u) == a, "copy assignment copies the values");
  from_eigen(v, a);

  auto s = make_vec<T>(5);
  t.checkThrow<Error>([&] { s = v; }, "copy assignment of a vector of a different size");
  t.checkNoThrow([&] { v = std::as_const(v); }, "self assignment");
  t.check(to_eigen(v) == a, "self assignment keeps the values");

  Vec<T> m(std::move(w));
  t.check(to_eigen(m) == T(2) * a, "move construction");
  t.checkThrow<Error>([&] { (void)w.size(); }, "using a moved-from vector");
  w = v;
  t.check(to_eigen(w) == a, "copy assignment to a moved-from vector");

  auto x = make_vec<T>(6);
  x = std::move(m);
  t.check(to_eigen(x) == T(2) * a, "move assignment");

  v = scalar<T>(3, -1);
  t.check(to_eigen(v) == Vector<T>::Constant(6, scalar<T>(3, -1)), "assignment of a scalar");
  return t;
}

template <class T>
TestSuite test_arithmetic()
{
  TestSuite t(name<T>("arithmetic"));
  const Vector<T> a = random_matrix<T>(7, 1, 2);
  const Vector<T> b = random_matrix<T>(7, 1, 3);
  const auto alpha = scalar<T>(0.5, -2);
  const auto x = random_vec<T>(7, 2);
  const auto y = random_vec<T>(7, 3);

  auto z = x;
  z.zero();
  t.check(to_eigen(z) == Vector<T>::Zero(7), "zero()");
  z.fill(alpha);
  t.check(to_eigen(z) == Vector<T>::Constant(7, alpha), "fill()");

  z = x;
  z *= alpha;
  t.check(approx(to_eigen(z), Vector<T>(alpha * a)), "operator*=");
  z = x;
  z += y;
  t.check(approx(to_eigen(z), Vector<T>(a + b)), "operator+=");
  z = x;
  z -= y;
  t.check(approx(to_eigen(z), Vector<T>(a - b)), "operator-=");
  z = x;
  z.axpy(alpha, y);
  t.check(approx(to_eigen(z), Vector<T>(a + alpha * b)), "axpy()");

  const Real<T> tol = std::is_same_v<Real<T>, float> ? Real<T>(1e-5) : Real<T>(1e-13);
  t.check(std::abs(x.dot(y) - a.dot(b)) <= tol * a.norm() * b.norm(), "dot() conjugates the left argument");
  t.check(std::abs(x.dot(x) - T(a.squaredNorm())) <= tol * a.squaredNorm(), "dot() with itself");
  t.check(std::abs(x.two_norm() - a.norm()) <= tol * a.norm(), "two_norm()");

  x.pointwise_mult(y, z);
  t.check(approx(to_eigen(z), Vector<T>(a.cwiseProduct(b))), "pointwise_mult()");

  auto big = random_vec<T>(9, 4);
  const auto big_before = to_eigen(big);
  big.copy_n_from(x, 7);
  Vector<T> expected = big_before;
  expected.head(7) = a;
  t.check(to_eigen(big) == expected, "copy_n_from() a shorter vector");
  auto small = make_vec<T>(4);
  small.copy_n_from(big, 3);
  expected = Vector<T>::Zero(4);
  expected.head(3) = a.head(3);
  t.check(to_eigen(small) == expected, "copy_n_from() a longer vector");

  t.checkThrow<Error>([&] { small.copy_n_from(big, 5); }, "copy_n_from() more entries than the vector has");
  t.checkThrow<Error>([&] { small.copy_n_from(big, -1); }, "copy_n_from() a negative number of entries");
  t.checkThrow<Error>([&] { big.copy_n_from(big, 3); }, "copy_n_from() itself");
  t.checkThrow<Error>([&] { small.axpy(T(1), big); }, "axpy() size mismatch");
  t.checkThrow<Error>([&] { (void)small.dot(big); }, "dot() size mismatch");
  return t;
}

template <class T>
TestSuite test_host_views()
{
  TestSuite t(name<T>("host views"));
  const Vector<T> a = random_matrix<T>(5, 1, 5);
  auto v = random_vec<T>(5, 5);
  auto w = random_vec<T>(5, 6);

  {
    auto view = v.host_view(ddm::read_write);
    view[2] = T(42);
    t.check(view.size() == 5 && view.span().size() == 5 && view.data() == &view[0], "size(), span() and data()");

    T sum{0};
    for (const auto& value : view) sum += value;
    t.check(std::abs(sum - (a.sum() - a(2) + T(42))) <= Real<T>(1e-4) * (1 + std::abs(sum)), "iterating over a host view");

    t.checkThrow<Error>([&] { auto second = std::as_const(v).host_view(ddm::read); }, "second host view");
    t.checkThrow<Error>([&] { (void)v.dot(w); }, "dot() with an open host view");
    t.checkThrow<Error>([&] { w.axpy(T(1), v); }, "axpy() with an open host view of the argument");
    t.checkNoThrow([&] { (void)v.size(); }, "size() with an open host view");
    t.checkNoThrow([&] { (void)v.backend(); }, "backend() with an open host view");

    view.release();
    t.checkNoThrow([&] { (void)v.dot(w); }, "dot() after release()");
  }
  t.check(to_eigen(v)(2) == T(42), "read_write host view");
  return t;
}

template <class T>
TestSuite test_column_views()
{
  TestSuite t(name<T>("column views"));
  const auto M = random_matrix<T>(6, 3, 7);
  const Vector<T> a = random_matrix<T>(6, 1, 8);
  auto X = make_multivec<T>(6, 3);
  from_eigen(*X, M);

  auto c = column(*X, 1);
  t.check(c.size() == 6 && c.multivec().is_view(), "a column view is a view");
  t.check(to_eigen(c) == M.col(1), "a column view shows the column");
  c *= T(2);
  Matrix<T> expected = M;
  expected.col(1) *= T(2);
  t.check(to_eigen(*X) == expected, "writes to a column view change the column");

  Vec<T> copy(c);
  copy.zero();
  t.check(!copy.multivec().is_view() && to_eigen(*X) == expected, "copies of a column view own their data");

  const auto v = random_vec<T>(6, 8);
  c = v;
  expected.col(1) = a;
  t.check(c.multivec().is_view() && to_eigen(*X) == expected, "assignment to a column view writes to the column");

  c = make_vec<T>(6);
  expected.col(1).setZero();
  t.check(c.multivec().is_view() && to_eigen(*X) == expected, "move assignment to a column view writes to the column");

  auto o = make_vec<T>(6);
  o = column(*X, 2);
  t.check(!o.multivec().is_view() && to_eigen(o) == M.col(2), "move assignment of a column view copies the values");

  auto c_again = column(*X, 1);
  c_again = c;
  t.check(to_eigen(*X) == expected, "assignment between two views of the same column");

  const Real<T> tol = std::is_same_v<Real<T>, float> ? Real<T>(1e-5) : Real<T>(1e-13);
  t.check(std::abs(column(*X, 0).dot(column(*X, 0)) - T(M.col(0).squaredNorm())) <= tol * M.col(0).squaredNorm(), "dot() of two views of the same column");
  t.checkThrow<Error>([&] { c.copy_n_from(c_again, 2); }, "copy_n_from() of a view of the same column");

  // The column views and their multivector share one storage, of which only one host view can be open
  {
    const auto view = std::as_const(*X).host_view(ddm::read);
    t.checkThrow<Error>([&] { c *= T(1); }, "using a column view while a host view of its multivector is open");
  }
  {
    auto view = c.host_view(ddm::read_write);
    view[0] = T(5);
    t.checkThrow<Error>([&] { X->scale(T(1)); }, "using the multivector while a host view of a column view is open");
  }
  t.check(to_eigen(*X)(0, 1) == T(5), "host view of a column view");
  return t;
}

template <class T>
TestSuite test_all()
{
  TestSuite t(name<T>("vec"));
  t.subTest(test_value_semantics<T>());
  t.subTest(test_arithmetic<T>());
  t.subTest(test_host_views<T>());
  t.subTest(test_column_views<T>());
  return t;
}
} // namespace

int main(int argc, char** argv)
{
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;
  t.subTest(test_all<float>());
  t.subTest(test_all<double>());
  t.subTest(test_all<std::complex<double>>());
  return t.exit();
}
