#pragma once

// Helpers shared by the unit tests: the backends the tests run on, conversion to Eigen for the reference results, and
// random data

#include "ddm/impl/multivec/istlmultivec.hh"
#include "ddm/index.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/vec/vec.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <memory>
#include <random>
#include <string_view>
#include <type_traits>

namespace ddm::test {
template <class T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>; // column-major, like a multivector

template <class T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <class T>
using Real = typename Vec<T>::real_type;

template <class T>
constexpr std::string_view type_name()
{
  if constexpr (std::is_same_v<T, float>) return "float";
  else if constexpr (std::is_same_v<T, double>) return "double";
  else return "complex<double>";
}

// The backend the tests run on. TODO: Run the tests for every backend once there is more than one
template <class T>
std::unique_ptr<MultiVec<T>> make_multivec(Index rows, Index cols)
{
  return std::make_unique<IstlMultiVec<T>>(rows, cols);
}

template <class T>
Matrix<T> to_eigen(const MultiVec<T>& X)
{
  const auto view = X.host_view(read);
  return Eigen::Map<const Matrix<T>>(view.data(), X.rows(), X.cols());
}

template <class T>
void from_eigen(MultiVec<T>& X, const Matrix<T>& M)
{
  auto view = X.host_view(write);
  Eigen::Map<Matrix<T>>(view.data(), X.rows(), X.cols()) = M;
}

template <class T>
Vector<T> to_eigen(const Vec<T>& v)
{
  const auto view = v.host_view(read);
  return Eigen::Map<const Vector<T>>(view.data(), v.size());
}

template <class T>
void from_eigen(Vec<T>& v, const Vector<T>& x)
{
  auto view = v.host_view(write);
  Eigen::Map<Vector<T>>(view.data(), v.size()) = x;
}

// Entries uniformly distributed in [-1, 1] (real and imaginary part for complex T), reproducible through the seed
template <class T>
Matrix<T> random_matrix(Index rows, Index cols, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Real<T>> dist(-1, 1);
  Matrix<T> M(rows, cols);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) {
      if constexpr (std::is_same_v<T, Real<T>>) M(i, j) = dist(rng);
      else M(i, j) = T(dist(rng), dist(rng));
    }
  return M;
}

// A multivector of the test backend with random entries
template <class T>
std::unique_ptr<MultiVec<T>> random_multivec(Index rows, Index cols, unsigned seed)
{
  auto X = make_multivec<T>(rows, cols);
  from_eigen(*X, random_matrix<T>(rows, cols, seed));
  return X;
}

// Whether a and b agree up to rounding errors, relative to the size of b
template <class A, class B>
bool approx(const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b)
{
  using R = typename A::RealScalar;
  const R tol = std::is_same_v<R, float> ? R(1e-5) : R(1e-13);
  return a.rows() == b.rows() && a.cols() == b.cols() && (a - b).norm() <= tol * std::max(R(1), b.norm());
}
} // namespace ddm::test
