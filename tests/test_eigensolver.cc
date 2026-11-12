// Unit tests of the Lanczos eigensolver: the eigenpairs of small problems against a dense reference, against Spectra
// with a projection, multiple eigenvalues (which need the replacement of dependent directions), a threshold instead of a
// fixed number of eigenpairs, and invalid configs.
// For double and complex<double>.

#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/impl/mat/istlmat.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/pattern.hh"
#include "test_helpers.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <dune/common/exceptions.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/test/testsuite.hh>
#include <format>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
using namespace ddm;
using namespace ddm::test;
using Dune::TestSuite;
using Error = Dune::InvalidStateException; // thrown by DDM_CHECK

template <class T>
std::string name(std::string_view test)
{
  return std::format("{}<{}>", test, type_name<T>());
}

// The sparse matrix with the nonzero entries of dense
template <class T>
std::shared_ptr<const LocalMat<T>> sparse(const Matrix<T>& dense)
{
  const Index n = dense.rows();
  Pattern pattern(n, n);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j)
      if (dense(i, j) != T(0) || i == j) pattern.add(i, j);
  pattern.finalize();

  auto M = std::make_shared<IstlMat<T>>(Dune::ParameterTree{}, pattern);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j)
      if (dense(i, j) != T(0) || i == j) {
        const T v = dense(i, j);
        M->add_values(std::span(&i, 1), std::span(&j, 1), std::span(&v, 1));
      }
  M->assemble();
  return M;
}

// A x = lambda B x with A the 1D Laplacian (Hermitian with complex entries for complex T, with the same eigenvalues)
// and B diagonal with random entries in [0.5, 1.5]. If zeros > 0, the first `zeros` entries of B are zero, so that B is
// only semidefinite (like B = D A D with a partition of unity D).
template <class T>
struct Problem {
  Matrix<T> A;
  Matrix<T> B;
};

template <class T>
Problem<T> laplace_problem(Index n, Index zeros)
{
  Problem<T> p{Matrix<T>::Zero(n, n), Matrix<T>::Zero(n, n)};
  T off = -1;
  if constexpr (!std::is_same_v<T, Real<T>>) off *= std::polar(Real<T>(1), Real<T>(0.3));
  for (Index i = 0; i < n; ++i) p.A(i, i) = 2;
  for (Index i = 0; i + 1 < n; ++i) {
    p.A(i + 1, i) = off;
    p.A(i, i + 1) = Dune::conjugateComplex(off);
  }
  const auto random = random_matrix<Real<T>>(n, 1, 1);
  for (Index i = 0; i < n; ++i) p.B(i, i) = i < zeros ? Real<T>(0) : Real<T>(1) + random(i, 0) / 2;
  return p;
}

// The nev smallest eigenvalues of A x = lambda B x for positive definite A and positive semidefinite B, from the dense
// problem B x = mu A x with mu = 1 / lambda
template <class T>
std::vector<Real<T>> reference_eigenvalues(const Problem<T>& p, Index nev)
{
  const Eigen::GeneralizedSelfAdjointEigenSolver<Matrix<T>> eig(p.B, p.A);
  std::vector<Real<T>> lambda;
  for (Index j = 0; j < nev; ++j) lambda.push_back(1 / eig.eigenvalues()(p.A.rows() - 1 - j)); // mu in increasing order
  return lambda;
}

Dune::ParameterTree lanczos_config(int nev, int blocksize)
{
  Dune::ParameterTree config;
  config["type"] = "lanczos";
  config["nev"] = std::to_string(nev);
  config["blocksize"] = std::to_string(blocksize);
  config["tolerance"] = "1e-10";
  return config;
}

// Whether the eigenvalues agree up to a relative error of 1e-8
template <class R>
bool same_eigenvalues(const std::vector<R>& a, const std::vector<R>& b)
{
  if (a.size() != b.size()) return false;
  for (std::size_t j = 0; j < a.size(); ++j)
    if (std::abs(a[j] - b[j]) > 1e-8 * std::abs(b[j])) return false;
  return true;
}

template <class T>
std::vector<Real<T>> real_values(const Eigensolver<T>& solver)
{
  std::vector<Real<T>> values;
  for (auto v : solver.values()) values.push_back(std::real(v));
  return values;
}

// B-orthonormal eigenvectors with A x = lambda B x. With a semidefinite B, the eigenvectors are only determined up to the
// kernel of B, so only B X is compared.
template <class T>
void check_eigenvectors(TestSuite& t, const Problem<T>& p, const Eigensolver<T>& solver, std::string_view where)
{
  const auto X = to_eigen(*solver.vectors());
  const Index nev = X.cols();
  Matrix<T> Lambda = Matrix<T>::Zero(nev, nev);
  for (Index j = 0; j < nev; ++j) Lambda(j, j) = solver.values()[j];
  t.check(approx(Matrix<T>(X.adjoint() * p.B * X), Matrix<T>(Matrix<T>::Identity(nev, nev))), std::format("B-orthonormal eigenvectors, {}", where));
  const bool definite = p.B.diagonal().real().minCoeff() > 0;
  const Matrix<T> residual = definite ? Matrix<T>(p.A * X - p.B * X * Lambda) : Matrix<T>(p.B * (p.A.inverse() * p.B * X * Lambda - X));
  t.check(residual.norm() < 1e-7, std::format("eigenvector residual {}, {}", residual.norm(), where));
}

// Eigenvalues and eigenvectors against the dense reference, for several block sizes (including ones that divide neither
// nev nor ncv) and for a definite and a semidefinite B
template <class T>
TestSuite test_against_dense()
{
  TestSuite t(name<T>("lanczos against dense reference"));
  const Index n = 120;
  const int nev = 6;
  for (const Index zeros : {Index(0), Index(10)}) {
    const auto p = laplace_problem<T>(n, zeros);
    const auto A = sparse(p.A);
    const auto B = sparse(p.B);
    const auto expected = reference_eigenvalues(p, nev);

    for (const int blocksize : {1, 2, 4, 5}) {
      const auto where = std::format("blocksize {}, {} zeros in B", blocksize, zeros);
      auto solver = create_eigensolver<T>(lanczos_config(nev, blocksize), A, B);
      solver->solve();
      t.check(same_eigenvalues(real_values(*solver), expected), std::format("eigenvalues, {}", where));
      check_eigenvectors(t, p, *solver, where);
    }
  }
  return t;
}

// With a two-sided projection, Lanczos and Spectra solve the same problem P (A - sigma B)^{-1} P^T B
template <class T>
TestSuite test_projection_against_spectra()
{
  TestSuite t(name<T>("lanczos against spectra with a projection"));
#ifdef DUNE_DDM_HAVE_SPECTRA
  const Index n = 100;
  const int nev = 8;
  const auto p = laplace_problem<T>(n, 5);
  const auto A = sparse(p.A);
  const auto B = sparse(p.B);

  // P zeroes the entries from 80 on, it is its own transpose
  const auto P = [](const MultiVec<T>& X, MultiVec<T>& Y) {
    Y.copy_from(X);
    auto view = Y.host_view(read_write);
    for (Index j = 0; j < Y.cols(); ++j)
      for (Index i = 80; i < Y.rows(); ++i) view(i, j) = 0;
  };

  auto spectra_config = lanczos_config(nev, 1);
  spectra_config["type"] = "spectra";
  auto spectra = create_eigensolver<T>(spectra_config, A, B);
  spectra->set_projection(P, P);
  spectra->solve();

  for (const int blocksize : {1, 3}) {
    auto lanczos = create_eigensolver<T>(lanczos_config(nev, blocksize), A, B);
    lanczos->set_projection(P, P);
    lanczos->solve();
    t.check(same_eigenvalues(real_values(*lanczos), real_values(*spectra)), std::format("eigenvalues, blocksize {}", blocksize));

    const auto X = to_eigen(*lanczos->vectors());
    t.check(X.bottomRows(n - 80).norm() == 0, std::format("eigenvectors in the range of P, blocksize {}", blocksize));
  }
#endif
  return t;
}

// Eigenvalues with a multiplicity larger than the block size. The Krylov space of a random block contains only blocksize
// vectors of each eigenspace, so it becomes invariant after a few steps, and the dependent directions of the next block
// are replaced by random vectors. Those bring in further vectors of the eigenspaces. Like any Krylov method, the solver
// cannot know whether it has found all copies of an eigenvalue, so only check that the result consists of eigenpairs and
// contains more copies of the smallest eigenvalue than the block size.
template <class T>
TestSuite test_multiple_eigenvalues()
{
  TestSuite t(name<T>("lanczos with multiple eigenvalues"));
  const Index n = 60;
  const int nev = 5;
  const int blocksize = 2;
  Problem<T> p{Matrix<T>::Zero(n, n), Matrix<T>::Identity(n, n)};
  for (Index i = 0; i < n; ++i) p.A(i, i) = 1 + i % 6; // the eigenvalues 1, ..., 6, each 10 times
  const auto A = sparse(p.A);
  const auto B = sparse(p.B);

  auto config = lanczos_config(nev, blocksize);
  config["ncv"] = "16";
  auto solver = create_eigensolver<T>(config, A, B);
  solver->solve();

  const auto values = real_values(*solver);
  t.check(std::ranges::all_of(values, [](auto v) { return std::abs(v - std::round(v)) < 1e-8; }), "only eigenvalues");
  t.check(std::ranges::is_sorted(values), "sorted");
  t.check(std::ranges::count_if(values, [](auto v) { return std::abs(v - 1) < 1e-8; }) > blocksize, "more copies of the eigenvalue 1 than the block size");

  const auto X = to_eigen(*solver->vectors());
  Matrix<T> Lambda = Matrix<T>::Zero(nev, nev);
  for (int j = 0; j < nev; ++j) Lambda(j, j) = solver->values()[j];
  t.check(approx(Matrix<T>(X.adjoint() * X), Matrix<T>(Matrix<T>::Identity(nev, nev))), "orthonormal eigenvectors");
  t.check((p.A * X - X * Lambda).norm() < 1e-7, "eigenvector residual");
  return t;
}

// With a threshold: all eigenvalues below it, but at least nev and at most nev_max. Starting from nev 4 and the default
// ncv 8, the solver has to grow the basis several times to hold the 40 eigenpairs below the threshold.
template <class T>
TestSuite test_threshold()
{
  TestSuite t(name<T>("lanczos with a threshold"));
  const Index n = 120;
  const int nev = 4;
  for (const Index zeros : {Index(0), Index(10)}) {
    const auto p = laplace_problem<T>(n, zeros);
    const auto A = sparse(p.A);
    const auto B = sparse(p.B);
    const auto reference = reference_eigenvalues(p, 60);
    const auto smallest = [&](Index count) {
      return std::vector(reference.begin(), reference.begin() + count);
    };
    const auto between = [&](Index j) {
      return std::format("{}", (reference[j - 1] + reference[j]) / 2);
    }; // the j-th and the next

    for (const int blocksize : {1, 4}) {
      const auto where = std::format("blocksize {}, {} zeros in B", blocksize, zeros);
      const auto solve = [&](const std::string& threshold, int nev_max) {
        auto config = lanczos_config(nev, blocksize);
        config["threshold"] = threshold;
        config["nev_max"] = std::to_string(nev_max);
        auto solver = create_eigensolver<T>(config, A, B);
        solver->solve();
        return solver;
      };

      const auto all = solve(between(40), 50);
      t.check(same_eigenvalues(real_values(*all), smallest(40)), std::format("the eigenvalues below the threshold, {}", where));
      check_eigenvectors(t, p, *all, where);

      const auto minimum = solve(std::format("{}", reference[0] / 2), 50);
      t.check(same_eigenvalues(real_values(*minimum), smallest(nev)), std::format("nev eigenvalues if none is below the threshold, {}", where));

      const auto capped = solve(between(40), 20);
      t.check(same_eigenvalues(real_values(*capped), smallest(20)), std::format("nev_max eigenvalues if more are below the threshold, {}", where));
    }
  }
  return t;
}

template <class T>
TestSuite test_errors()
{
  TestSuite t(name<T>("lanczos errors"));
  const auto p = laplace_problem<T>(20, 0);
  const auto A = sparse(p.A);
  const auto B = sparse(p.B);

  auto config = lanczos_config(4, 0);
  t.checkThrow<Error>([&] { create_eigensolver<T>(config, A, B); }, "blocksize 0");
  config = lanczos_config(4, 2);
  config["keep"] = "1";
  t.checkThrow<Error>([&] { create_eigensolver<T>(config, A, B); }, "keep 1");
  config = lanczos_config(4, 4);
  config["ncv"] = "17"; // ncv + blocksize > 20
  t.checkThrow<Error>([&] { create_eigensolver<T>(config, A, B)->solve(); }, "ncv + blocksize larger than the problem");
  config = lanczos_config(4, 2);
  config["threshold"] = "-1"; // below the default shift
  t.checkThrow<Error>([&] { create_eigensolver<T>(config, A, B); }, "threshold below the shift");
  config = lanczos_config(4, 2);
  config["threshold"] = "1";
  config["nev_max"] = "3";
  t.checkThrow<Error>([&] { create_eigensolver<T>(config, A, B); }, "nev_max smaller than nev");
#ifdef DUNE_DDM_HAVE_SPECTRA
  config = lanczos_config(4, 2);
  config["type"] = "spectra";
  config["threshold"] = "1";
  t.checkThrow<Error>([&] { create_eigensolver<T>(config, A, B)->solve(); }, "threshold with spectra");
#endif
  return t;
}

template <class T>
TestSuite test_all()
{
  TestSuite t(name<T>("eigensolver"));
  t.subTest(test_against_dense<T>());
  t.subTest(test_projection_against_spectra<T>());
  t.subTest(test_multiple_eigenvalues<T>());
  t.subTest(test_threshold<T>());
  t.subTest(test_errors<T>());
  return t;
}
} // namespace

int main(int argc, char** argv)
{
  Dune::MPIHelper::instance(argc, argv);
  TestSuite t;
  t.subTest(test_all<double>());
  t.subTest(test_all<std::complex<double>>());
  return t.exit();
}
