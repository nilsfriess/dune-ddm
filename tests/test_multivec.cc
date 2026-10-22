// Unit tests of the MultiVec interface: host views, views of columns, the block operations and the operations with
// vectors, checked against Eigen for float, double and complex<double>

#include "test_helpers.hh"

#include "ddm/impl/mat/istlmat.hh"
#include "ddm/mat/pattern.hh"

#include <complex>
#include <dune/common/exceptions.hh>
#include <dune/common/math.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/test/testsuite.hh>
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace ddm;
using namespace ddm::test;
using Dune::TestSuite;
using Error = Dune::InvalidStateException; // thrown by DDM_CHECK

// re + i im for complex T, re otherwise, so that the complex tests also use complex coefficients
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

// An operation is run on the owner itself and on a view of some of its columns. For the view, the columns outside of
// it must not change.
struct Case {
  Index first;
  Index count;
  bool view;
};

constexpr Index case_cols = 6;
constexpr Case cases[] = {{0, case_cols, false}, {2, 3, true}};

// The multivector an operation is applied to in case c: the owner, or a view that is stored in `view`
template <class T>
MultiVec<T>& select(MultiVec<T>& owner, const Case& c, std::unique_ptr<MultiVec<T>>& view)
{
  if (!c.view) return owner;
  view = owner.columns(c.first, c.count);
  return *view;
}

template <class T>
TestSuite test_basics()
{
  TestSuite t(name<T>("basics"));
  auto X = make_multivec<T>(5, 3);
  t.check(X->rows() == 5 && X->cols() == 3, "size");
  t.check(X->backend() == BackendId::istl, "backend");
  t.check(!X->is_view(), "an owning multivector is no view");
  t.check(to_eigen(*X) == Matrix<T>::Zero(5, 3), "zero-initialized");

  const auto M = random_matrix<T>(5, 3, 1);
  from_eigen(*X, M);
  t.check(to_eigen(*X) == M, "write and read through host views");
  {
    auto view = X->host_view(ddm::read_write);
    view(4, 2) = T(42);
  }
  t.check(to_eigen(*X)(4, 2) == T(42), "read_write host view");

  // While a host view is open, the multivector cannot be used otherwise
  auto Y = make_multivec<T>(5, 3);
  {
    const auto view = std::as_const(*X).host_view(ddm::read);
    t.checkThrow<Error>([&] { auto second = std::as_const(*X).host_view(ddm::read); }, "second host view");
    t.checkThrow<Error>([&] { X->fill(T(1)); }, "fill() with an open host view");
    t.checkThrow<Error>([&] { Y->copy_from(*X); }, "copy_from() with an open host view of the argument");
  }
  t.checkNoThrow([&] { X->fill(T(1)); }, "fill() after the host view was released");

  auto Z = X->create_multivector(2, 4);
  t.check(Z->rows() == 2 && Z->cols() == 4 && Z->backend() == X->backend() && to_eigen(*Z) == Matrix<T>::Zero(2, 4), "create_multivector()");
  return t;
}

template <class T>
TestSuite test_views()
{
  TestSuite t(name<T>("views"));
  const auto M = random_matrix<T>(6, 5, 2);
  auto X = make_multivec<T>(6, 5);
  from_eigen(*X, M);

  auto V = X->columns(1, 3);
  t.check(V->is_view() && V->rows() == 6 && V->cols() == 3 && V->backend() == X->backend(), "shape of a view");
  t.check(to_eigen(*V) == M.middleCols(1, 3), "a view shows the columns of its owner");

  V->fill(T(7));
  Matrix<T> expected = M;
  expected.middleCols(1, 3).setConstant(T(7));
  t.check(to_eigen(*X) == expected, "writes through a view change only its columns of the owner");

  X->scale(T(2));
  t.check(to_eigen(*V) == Matrix<T>::Constant(6, 3, T(14)), "writes to the owner are visible in the view");

  auto W = V->columns(1, 2);
  t.check(W->is_view() && to_eigen(*W) == to_eigen(*X).middleCols(2, 2), "view of a view");
  const auto R = random_matrix<T>(6, 2, 3);
  from_eigen(*W, R);
  t.check(to_eigen(*X).middleCols(2, 2) == R, "a host view of a view of a view writes to the owner");

  const auto& X_const = *X;
  std::unique_ptr<const MultiVec<T>> C = X_const.columns(3, 2);
  t.check(to_eigen(*C) == to_eigen(*X).middleCols(3, 2), "view of a const multivector");

  auto E = X->columns(5, 0);
  t.check(E->rows() == 6 && E->cols() == 0, "empty view at the end");
  t.checkThrow<Error>([&] { X->columns(4, 2); }, "range past the end");
  t.checkThrow<Error>([&] { X->columns(-1, 1); }, "negative first column");
  t.checkThrow<Error>([&] { X->columns(0, -1); }, "negative count");

  auto Y = make_multivec<T>(6, 5);
  t.check(X->overlaps(*X), "a multivector overlaps itself");
  t.check(!X->overlaps(*Y) && !Y->overlaps(*X), "different multivectors do not overlap");
  t.check(X->overlaps(*V) && V->overlaps(*X), "a view overlaps its owner");
  t.check(V->overlaps(*W) && W->overlaps(*V), "a view overlaps the view it was created from");
  t.check(!X->columns(0, 2)->overlaps(*X->columns(2, 2)), "adjacent column ranges do not overlap");
  t.check(X->columns(0, 3)->overlaps(*X->columns(2, 2)), "intersecting column ranges overlap");
  t.check(!E->overlaps(*X) && !X->overlaps(*E), "an empty view overlaps nothing");

  // The owner and its views share one storage, of which only one host view can be open
  const auto V_expected = to_eigen(*V);
  {
    const auto view = std::as_const(*V).host_view(ddm::read);
    t.check(view(5, 2) == V_expected(5, 2), "host view of a view");
    t.checkThrow<Error>([&] { auto second = std::as_const(*X).host_view(ddm::read); }, "host view of the owner while one of a view is open");
    t.checkThrow<Error>([&] { X->scale(T(1)); }, "using the owner while a host view of a view is open");
    t.checkThrow<Error>([&] { W->scale(T(1)); }, "using another view while a host view of a view is open");
  }
  t.checkNoThrow([&] { X->scale(T(1)); }, "using the owner after the host view of a view was released");
  return t;
}

template <class T>
TestSuite test_level1()
{
  TestSuite t(name<T>("fill, scale, copy_from, axpy"));
  const auto Mx = random_matrix<T>(7, case_cols, 4);
  const auto My = random_matrix<T>(7, case_cols, 5);
  const auto alpha = scalar<T>(-0.5, 2);

  // Every operation is tested twice (see `cases`): once on a whole 7x6 multivector, and once on a view of its columns
  // 2, 3 and 4. X and Y are what the operations work on: either the owners X_owner and Y_owner themselves, or views of
  // them.
  //
  // run_and_compare() resets X_owner to the random matrix Mx and then runs the operation on X. Afterwards, the columns
  // of X must contain `expected` (computed with Eigen), and all other columns of X_owner must still be as in Mx. For the
  // view, the second part checks that the operation did not write outside of the view.
  for (const auto& c : cases) {
    auto X_owner = make_multivec<T>(7, case_cols);
    auto Y_owner = make_multivec<T>(7, case_cols);
    from_eigen(*Y_owner, My);
    std::unique_ptr<MultiVec<T>> X_view;
    std::unique_ptr<MultiVec<T>> Y_view;
    auto& X = select(*X_owner, c, X_view);
    const auto& Y = select(*Y_owner, c, Y_view);

    const auto run_and_compare = [&](auto op, const Matrix<T>& expected, std::string_view what) {
      from_eigen(*X_owner, Mx);
      op();
      Matrix<T> expected_owner = Mx;
      expected_owner.middleCols(c.first, c.count) = expected;
      t.check(approx(to_eigen(*X_owner), expected_owner), std::format("{}, {}", what, c.view ? "view" : "owner"));
    };

    // The columns of Mx and My that X and Y contain
    const auto x = Mx.middleCols(c.first, c.count);
    const auto y = My.middleCols(c.first, c.count);
    run_and_compare([&] { X.fill(alpha); }, Matrix<T>::Constant(7, c.count, alpha), "fill()");
    run_and_compare([&] { X.scale(alpha); }, alpha * x, "scale()");
    run_and_compare([&] { X.copy_from(Y); }, y, "copy_from()");
    run_and_compare([&] { X.axpy(alpha, Y); }, x + alpha * y, "axpy()");
  }
  return t;
}

template <class T>
TestSuite test_dot()
{
  TestSuite t(name<T>("dot"));
  const auto Mx = random_matrix<T>(9, 5, 6);
  const auto My = random_matrix<T>(9, 6, 7);
  auto X = make_multivec<T>(9, 5);
  auto Y = make_multivec<T>(9, 6);
  from_eigen(*X, Mx);
  from_eigen(*Y, My);

  auto C = make_multivec<T>(5, 6);
  X->dot(*Y, *C);
  t.check(approx(to_eigen(*C), Matrix<T>(Mx.adjoint() * My)), "C = X^H Y");

  auto G = make_multivec<T>(5, 5);
  X->dot(*X, *G);
  t.check(approx(to_eigen(*G), Matrix<T>(Mx.adjoint() * Mx)), "Gram matrix X^H X");

  // Views as all three arguments, the result is written to some columns of a larger multivector
  auto C_owner = make_multivec<T>(3, 8);
  X->columns(1, 3)->dot(*Y->columns(2, 4), *C_owner->columns(3, 4));
  Matrix<T> expected = Matrix<T>::Zero(3, 8);
  expected.middleCols(3, 4) = Mx.middleCols(1, 3).adjoint() * My.middleCols(2, 4);
  t.check(approx(to_eigen(*C_owner), expected), "views of X, Y and C");
  return t;
}

template <class T>
TestSuite test_mult()
{
  TestSuite t(name<T>("mult"));
  const auto Mx = random_matrix<T>(8, 4, 8);
  const auto Mm = random_matrix<T>(4, 3, 9);
  const auto My = random_matrix<T>(8, 3, 10);
  auto X = make_multivec<T>(8, 4);
  auto M = make_multivec<T>(4, 3);
  auto Y = make_multivec<T>(8, 3);
  from_eigen(*X, Mx);
  from_eigen(*M, Mm);

  const std::pair<T, T> coefficients[] = {{T(1), T(0)}, {scalar<T>(2, 1), T(0)}, {T(-1), T(1)}, {T(0.5), scalar<T>(-2, 0.5)}, {T(0), T(1)}, {T(0), T(0)}};
  for (std::size_t k = 0; k < std::size(coefficients); ++k) {
    const auto [alpha, beta] = coefficients[k];
    from_eigen(*Y, My);
    X->mult(*M, *Y, alpha, beta);
    t.check(approx(to_eigen(*Y), Matrix<T>(alpha * Mx * Mm + beta * My)), std::format("Y = alpha X M + beta Y for coefficient pair {}", k));
  }

  // beta = 0: Y is not read, so NaNs in Y must not show up in the result
  {
    const Real<T> nan = std::numeric_limits<Real<T>>::quiet_NaN();
    Y->fill(scalar<T>(nan, nan));
    X->mult(*M, *Y);
    const auto result = to_eigen(*Y);
    t.check(result.allFinite() && approx(result, Matrix<T>(Mx * Mm)), "beta = 0 ignores NaNs in Y");
  }

  // Views as all three arguments
  {
    const auto Mx_big = random_matrix<T>(8, 7, 11);
    const auto Mm_big = random_matrix<T>(4, 5, 12);
    const auto My_big = random_matrix<T>(8, 6, 13);
    auto X_big = make_multivec<T>(8, 7);
    auto M_big = make_multivec<T>(4, 5);
    auto Y_big = make_multivec<T>(8, 6);
    from_eigen(*X_big, Mx_big);
    from_eigen(*M_big, Mm_big);
    from_eigen(*Y_big, My_big);

    X_big->columns(2, 4)->mult(*M_big->columns(1, 3), *Y_big->columns(3, 3), T(-1), T(1));
    Matrix<T> expected = My_big;
    expected.middleCols(3, 3) -= Mx_big.middleCols(2, 4) * Mm_big.middleCols(1, 3);
    t.check(approx(to_eigen(*Y_big), expected), "views of X, M and Y");

    // X and Y may be views of the same multivector if they do not overlap (as in a restart that writes into the
    // trailing columns of the basis)
    from_eigen(*X_big, Mx_big);
    X_big->columns(0, 4)->mult(*M_big->columns(0, 3), *X_big->columns(4, 3));
    expected = Mx_big;
    expected.middleCols(4, 3) = Mx_big.leftCols(4) * Mm_big.leftCols(3);
    t.check(approx(to_eigen(*X_big), expected), "X and Y distinct views of the same multivector");
  }
  return t;
}

template <class T>
TestSuite test_columnwise()
{
  TestSuite t(name<T>("column_dots, column_norms, pointwise_mult, copy_rows_from"));
  const auto Mx = random_matrix<T>(7, case_cols, 25);
  const auto My = random_matrix<T>(7, case_cols, 26);

  for (const auto& c : cases) {
    const auto where = c.view ? "view" : "owner";
    auto X_owner = make_multivec<T>(7, case_cols);
    auto Y_owner = make_multivec<T>(7, case_cols);
    auto Z_owner = make_multivec<T>(7, case_cols);
    from_eigen(*X_owner, Mx);
    from_eigen(*Y_owner, My);
    std::unique_ptr<MultiVec<T>> X_view;
    std::unique_ptr<MultiVec<T>> Y_view;
    std::unique_ptr<MultiVec<T>> Z_view;
    const auto& X = select(*X_owner, c, X_view);
    const auto& Y = select(*Y_owner, c, Y_view);
    auto& Z = select(*Z_owner, c, Z_view);
    const Matrix<T> x = Mx.middleCols(c.first, c.count);
    const Matrix<T> y = My.middleCols(c.first, c.count);

    std::vector<T> dots(c.count);
    X.column_dots(Y, dots);
    t.check(approx(Eigen::Map<const Vector<T>>(dots.data(), c.count), Vector<T>((x.adjoint() * y).diagonal())), std::format("column_dots(), {}", where));
    X.column_dots(X, dots);
    t.check(approx(Eigen::Map<const Vector<T>>(dots.data(), c.count), Vector<T>((x.adjoint() * x).diagonal())), std::format("column_dots() with itself, {}", where));

    std::vector<Real<T>> norms(c.count);
    X.column_norms(norms);
    t.check(approx(Eigen::Map<const Vector<Real<T>>>(norms.data(), c.count), Vector<Real<T>>(x.colwise().norm().transpose())), std::format("column_norms(), {}", where));

    X.pointwise_mult(Y, Z);
    Matrix<T> expected = Matrix<T>::Zero(7, case_cols);
    expected.middleCols(c.first, c.count) = x.cwiseProduct(y);
    t.check(approx(to_eigen(*Z_owner), expected), std::format("pointwise_mult(), {}", where));

    // To more and to fewer rows
    const auto Mw = random_matrix<T>(9, c.count, 27);
    auto W = make_multivec<T>(9, c.count);
    from_eigen(*W, Mw);
    W->copy_rows_from(X, 7);
    expected = Mw;
    expected.topRows(7) = x;
    t.check(to_eigen(*W) == expected, std::format("copy_rows_from() to more rows, {}", where));
    auto S = make_multivec<T>(4, c.count);
    S->copy_rows_from(X, 3);
    expected = Matrix<T>::Zero(4, c.count);
    expected.topRows(3) = x.topRows(3);
    t.check(to_eigen(*S) == expected, std::format("copy_rows_from() to fewer rows, {}", where));
    t.checkThrow<Error>([&] { S->copy_rows_from(X, 5); }, std::format("copy_rows_from() of more rows than there are, {}", where));
  }

  auto X = make_multivec<T>(3, 2);
  std::vector<T> too_short(1);
  t.checkThrow<Error>([&] { X->column_dots(*X, too_short); }, "column_dots() with the wrong number of results");
  t.checkThrow<Error>([&] { X->columns(0, 1)->copy_rows_from(*X->columns(0, 1), 2); }, "copy_rows_from() of itself");
  return t;
}

// The operations with vectors (multivectors with one column, and columns as vectors) and the row operations, on the
// owner and on a view
template <class T>
TestSuite test_vector_ops()
{
  TestSuite t(name<T>("operations with vectors and rows"));
  const auto Mx = random_matrix<T>(7, case_cols, 14);
  const Vector<T> y = random_matrix<T>(7, 1, 15);
  const Vector<T> d = random_matrix<T>(7, 1, 16);

  for (const auto& c : cases) {
    const auto where = c.view ? "view" : "owner";
    auto X_owner = make_multivec<T>(7, case_cols);
    from_eigen(*X_owner, Mx);
    std::unique_ptr<MultiVec<T>> X_view;
    auto& X = select(*X_owner, c, X_view);
    const Matrix<T> x = Mx.middleCols(c.first, c.count);

    auto y_vec = create_vector(X, 7);
    from_eigen(y_vec, y);
    auto c_vec = create_vector(X, c.count);
    X.dot(y_vec.multivec(), c_vec.multivec());
    t.check(approx(to_eigen(c_vec), Vector<T>(x.adjoint() * y)), std::format("dot() with a vector, {}", where));

    const Vector<T> coeffs = random_matrix<T>(c.count, 1, 17);
    from_eigen(c_vec, coeffs);
    X.mult(c_vec.multivec(), y_vec.multivec());
    t.check(approx(to_eigen(y_vec), Vector<T>(x * coeffs)), std::format("mult() with a vector, {}", where));

    y_vec = column(X, 1);
    t.check(to_eigen(y_vec) == x.col(1), std::format("assigning a column to a vector, {}", where));

    from_eigen(y_vec, y);
    column(X, 2) = y_vec;
    Matrix<T> expected = Mx;
    expected.col(c.first + 2) = y;
    t.check(to_eigen(*X_owner) == expected, std::format("assigning a vector to a column, {}", where));

    from_eigen(*X_owner, Mx);
    auto d_vec = create_vector(X, 7);
    from_eigen(d_vec, d);
    X.scale_rows(d_vec.multivec());
    expected = Mx;
    expected.middleCols(c.first, c.count) = d.asDiagonal() * x;
    t.check(approx(to_eigen(*X_owner), expected), std::format("scale_rows(), {}", where));

    from_eigen(*X_owner, Mx);
    const std::vector<Index> idx{5, 0, 3};
    auto buffer = X.create_multivector(3, c.count);
    X.pack_rows(idx, *buffer);
    Matrix<T> packed(3, c.count);
    for (std::size_t k = 0; k < idx.size(); ++k) packed.row(k) = x.row(idx[k]);
    t.check(to_eigen(*buffer) == packed, std::format("pack_rows(), {}", where));

    // Unpack into the columns [1, 1 + c.count) of a zero multivector with one column more than the buffer
    auto Z_owner = make_multivec<T>(7, c.count + 1);
    std::unique_ptr<MultiVec<T>> Z_view;
    if (c.view) Z_view = Z_owner->columns(0, c.count + 1);
    auto& Z = c.view ? *Z_view : *Z_owner;
    Z.unpack_rows(*buffer, idx, 1);
    expected = Matrix<T>::Zero(7, c.count + 1);
    for (std::size_t k = 0; k < idx.size(); ++k) expected.row(idx[k]).tail(c.count) = packed.row(k);
    t.check(to_eigen(*Z_owner) == expected, std::format("unpack_rows(), {}", where));
  }
  return t;
}

// The tridiagonal n x n matrix with a on the diagonal and b, conj(b) below and above it, and the same matrix as a dense
// Eigen matrix
template <class T>
std::pair<std::shared_ptr<LocalMat<T>>, Matrix<T>> tridiagonal(Index n, T a, T b)
{
  Pattern pattern(n, n, 3);
  for (Index i = 0; i < n; ++i)
    for (Index j = std::max<Index>(0, i - 1); j <= std::min<Index>(n - 1, i + 1); ++j) pattern.add(i, j);
  pattern.finalize();

  auto A = std::make_shared<IstlMat<T>>(Dune::ParameterTree{}, pattern);
  Matrix<T> dense = Matrix<T>::Zero(n, n);
  for (Index i = 0; i < n; ++i) dense(i, i) = a;
  for (Index i = 0; i + 1 < n; ++i) {
    dense(i + 1, i) = b;
    dense(i, i + 1) = Dune::conjugateComplex(b);
  }
  for (Index i = 0; i < n; ++i)
    for (Index j = std::max<Index>(0, i - 1); j <= std::min<Index>(n - 1, i + 1); ++j) {
      const T v = dense(i, j);
      A->add_values(std::span(&i, 1), std::span(&j, 1), std::span(&v, 1));
    }
  A->assemble();
  return {A, dense};
}

template <class T>
TestSuite test_spmm()
{
  TestSuite t(name<T>("spmm"));
  const auto [A, A_dense] = tridiagonal<T>(8, scalar<T>(2, 1), scalar<T>(-1, 0.5));
  const auto Mx = random_matrix<T>(8, 5, 18);
  auto X = make_multivec<T>(8, 5);
  auto Y = make_multivec<T>(8, 5);
  from_eigen(*X, Mx);

  A->spmm(*X, *Y);
  t.check(approx(to_eigen(*Y), Matrix<T>(A_dense * Mx)), "Y = A X");

  Y->fill(T(0));
  A->spmm(*X->columns(1, 2), *Y->columns(3, 2));
  Matrix<T> expected = Matrix<T>::Zero(8, 5);
  expected.middleCols(3, 2) = A_dense * Mx.middleCols(1, 2);
  t.check(approx(to_eigen(*Y), expected), "views of X and Y");

  A->spmm(*X->columns(0, 2), *X->columns(2, 2));
  expected = Mx;
  expected.middleCols(2, 2) = A_dense * Mx.leftCols(2);
  t.check(approx(to_eigen(*X), expected), "X and Y distinct views of the same multivector");

  t.checkThrow<Error>([&] { A->spmm(*X->columns(0, 2), *X->columns(1, 2)); }, "overlapping X and Y");

  // mv(), usmv() and get_diag() with column views as vectors
  from_eigen(*X, Mx);
  Y->fill(T(0));
  const auto alpha = scalar<T>(0.5, 1);
  {
    auto x0 = column(*X, 0);
    auto y1 = column(*Y, 1);
    A->mv(x0, y1);
    A->usmv(alpha, x0, y1);
    auto d = column(*Y, 4);
    A->get_diag(d);
    t.checkThrow<Error>([&] { auto x0_again = column(*X, 0); A->mv(x0, x0_again); }, "mv() with x and y views of the same column");
  }
  expected = Matrix<T>::Zero(8, 5);
  expected.col(1) = (T(1) + alpha) * A_dense * Mx.col(0);
  expected.col(4) = A_dense.diagonal();
  t.check(approx(to_eigen(*Y), expected), "mv(), usmv() and get_diag() on column views");

  // mv() overwrites the entries of empty rows as well
  Pattern pattern(3, 3);
  pattern.add(0, 0);
  pattern.add(2, 2);
  pattern.finalize();
  const auto B = std::make_shared<IstlMat<T>>(Dune::ParameterTree{}, pattern);
  B->assemble();
  Vec<T> u(make_multivec<T>(3, 1));
  Vec<T> v(make_multivec<T>(3, 1));
  u.fill(T(1));
  v.fill(T(7));
  B->mv(u, v);
  t.check(to_eigen(v) == Vector<T>::Zero(3), "mv() with an empty row");
  return t;
}

template <class T>
TestSuite test_errors()
{
  TestSuite t(name<T>("argument checks"));
  auto X = random_multivec<T>(6, 4, 19);
  auto Y = random_multivec<T>(6, 3, 20);
  auto M = random_multivec<T>(4, 3, 21);

  t.checkThrow<Error>([&] { X->copy_from(*Y); }, "copy_from() size mismatch");
  t.checkThrow<Error>([&] { X->axpy(T(1), *Y); }, "axpy() size mismatch");
  // Operations that work entry by entry accept the same columns as input and output, but no other overlap
  const auto Mx = to_eigen(*X);
  X->copy_from(*X);
  t.check(to_eigen(*X) == Mx, "copy_from() of itself does nothing");
  X->axpy(T(1), *X);
  t.check(approx(to_eigen(*X), Matrix<T>(T(2) * Mx)), "axpy() of itself");
  X->columns(1, 2)->pointwise_mult(*Y->columns(0, 2), *X->columns(1, 2));
  t.checkThrow<Error>([&] { X->columns(0, 2)->axpy(T(1), *X->columns(1, 2)); }, "axpy() of an overlapping view");
  t.checkThrow<Error>([&] { X->columns(0, 2)->pointwise_mult(*Y->columns(0, 2), *X->columns(1, 2)); }, "pointwise_mult() into an overlapping view");
  t.checkThrow<Error>([&] { X->columns(0, 2)->copy_from(*X->columns(1, 2)); }, "copy_from() of an overlapping view");
  t.checkNoThrow([&] { X->columns(0, 2)->copy_from(*X->columns(2, 2)); }, "copy_from() of a distinct view of the same multivector");

  auto C = make_multivec<T>(4, 4);
  t.checkThrow<Error>([&] { X->dot(*Y, *C); }, "dot() with C of the wrong size");
  auto S = random_multivec<T>(6, 8, 22);
  auto Z = random_multivec<T>(6, 2, 23);
  t.checkThrow<Error>([&] { S->columns(0, 6)->dot(*Z, *S->columns(5, 2)); }, "dot() with C overlapping X");
  t.checkNoThrow([&] { S->columns(0, 6)->dot(*Z, *S->columns(6, 2)); }, "dot() with C a distinct view of the same multivector as X");

  auto Y_wrong = make_multivec<T>(6, 2);
  t.checkThrow<Error>([&] { X->mult(*M, *Y_wrong); }, "mult() with Y of the wrong size");
  auto M_wrong = make_multivec<T>(3, 3);
  t.checkThrow<Error>([&] { X->mult(*M_wrong, *Y); }, "mult() with M of the wrong size");
  auto M32 = make_multivec<T>(3, 2);
  t.checkThrow<Error>([&] { S->columns(0, 3)->mult(*M32, *S->columns(2, 2)); }, "mult() with Y overlapping X");
  auto Q = random_multivec<T>(4, 6, 24);
  t.checkThrow<Error>([&] { Q->columns(0, 4)->mult(*Q->columns(4, 2), *Q->columns(4, 2)); }, "mult() with Y overlapping M");

  auto D = make_multivec<T>(6, 2);
  t.checkThrow<Error>([&] { X->scale_rows(*D); }, "scale_rows() with more than one column of scaling factors");

  const std::vector<Index> bad_idx{0, 6};
  auto buffer = make_multivec<T>(2, 4);
  t.checkThrow<Error>([&] { X->pack_rows(bad_idx, *buffer); }, "pack_rows() with a row out of range");
  t.checkThrow<Error>([&] { X->columns(0, 2)->unpack_rows(*X->columns(1, 2), std::vector<Index>(6, 0), 0); }, "unpack_rows() with an overlapping buffer");
  return t;
}

template <class T>
TestSuite test_all()
{
  TestSuite t(name<T>("multivec"));
  t.subTest(test_basics<T>());
  t.subTest(test_views<T>());
  t.subTest(test_level1<T>());
  t.subTest(test_columnwise<T>());
  t.subTest(test_dot<T>());
  t.subTest(test_mult<T>());
  t.subTest(test_vector_ops<T>());
  t.subTest(test_spmm<T>());
  t.subTest(test_errors<T>());
  return t;
}
} // namespace

int main()
{
  TestSuite t;
  t.subTest(test_all<float>());
  t.subTest(test_all<double>());
  t.subTest(test_all<std::complex<double>>());
  return t.exit();
}
