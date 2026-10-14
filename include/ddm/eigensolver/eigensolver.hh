#pragma once

#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/registry.hh"

#include <dune/common/parametertree.hh>
#include <functional>
namespace ddm {
/** Solver for the generalised eigenproblem A x = lambda B x, computing the eigenpairs with the smallest eigenvalues.
 *
 *  Config:
 *  - nev:       number of eigenpairs to compute (default 16)
 *  - ncv:       number of Lanczos vectors (default 2 * nev)
 *  - maxit:     maximum number of iterations (default 1000)
 *  - tolerance: convergence tolerance (default 1e-5)
 *  - shift:     shift sigma of the shift-and-invert operator (A - sigma B)^{-1} (default 1e-3)
 */
template <class T>
class Eigensolver {
public:
  static constexpr int default_nev = 16;

  Eigensolver(const Dune::ParameterTree& config, std::shared_ptr<const LocalMat<T>> A, std::shared_ptr<const LocalMat<T>> B)
      : A_(std::move(A))
      , B_(std::move(B))
      , nev_(config.get("nev", default_nev))
      , ncv_(config.get("ncv", 2 * nev_))
      , maxit_(config.get("maxit", 1000))
      , tolerance_(config.get("tolerance", T(1e-5)))
      , shift_(config.get("shift", T(1e-3)))
  {
    DDM_CHECK(A_->rows() == B_->rows(), "Eigensolver: size mismatch, A has {} rows, B has {}", A_->rows(), B_->rows());
    DDM_CHECK(A_->cols() == B_->cols(), "Eigensolver: size mismatch, A has {} cols, B has {}", A_->cols(), B_->cols());
    DDM_CHECK(nev_ >= 1, "Eigensolver: nev must be at least 1, got {}", nev_);
    DDM_CHECK(ncv_ > nev_, "Eigensolver: ncv must be larger than nev, got ncv = {} and nev = {}", ncv_, nev_);
  }

  virtual ~Eigensolver() = default;

  // Restricts the iteration to the range of P, which must be a projection (P P = P). P is applied after every
  // shift-invert solve and to the starting vector. If its transpose P_transposed is given as well, it is applied
  // before every shift-invert solve, so that the operator becomes P (A - sigma B)^{-1} P^T B, which is self-adjoint in
  // the B inner product for every shift.
  using Projection = std::function<void(const Vec<T>& x, Vec<T>& y)>;
  void set_projection(Projection P, Projection P_transposed = {})
  {
    projection_ = std::move(P);
    projection_transposed_ = std::move(P_transposed);
  }

  virtual void solve() = 0;
  virtual const std::vector<T>& values() const = 0;
  virtual std::shared_ptr<MultiVec<T>> vectors() const = 0;

  const std::shared_ptr<const LocalMat<T>>& A() const { return A_; }
  const std::shared_ptr<const LocalMat<T>>& B() const { return B_; }
  int nev() const { return nev_; }
  int ncv() const { return ncv_; }
  int maxit() const { return maxit_; }
  T tolerance() const { return tolerance_; }
  T shift() const { return shift_; }

protected:
  const Projection& projection() const { return projection_; }                       // empty if not set
  const Projection& projection_transposed() const { return projection_transposed_; } // empty if not set

private:
  std::shared_ptr<const LocalMat<T>> A_;
  std::shared_ptr<const LocalMat<T>> B_;

  int nev_;
  int ncv_;
  int maxit_;
  T tolerance_;
  T shift_;

  Projection projection_;
  Projection projection_transposed_;
};

template <class T>
using EigensolverRegistry = Registry<std::shared_ptr<Eigensolver<T>>, std::shared_ptr<const LocalMat<T>>, std::shared_ptr<const LocalMat<T>>>;

template <class T>
void register_eigensolver(std::string name, typename EigensolverRegistry<T>::Factory factory)
{
  EigensolverRegistry<T>::instance().add(std::move(name), std::move(factory));
}

template <class T>
std::shared_ptr<Eigensolver<T>> create_eigensolver(const Dune::ParameterTree& config, std::shared_ptr<const LocalMat<T>> A, std::shared_ptr<const LocalMat<T>> B)
{
  initialize();
  DDM_CHECK(A != nullptr, "eigensolver: matrix A is nullptr");
  DDM_CHECK(B != nullptr, "eigensolver: matrix B is nullptr");
  DDM_CHECK(A->backend() == B->backend(), "eigensolver: backend mismatch, A has {}, B has {}", to_string(A->backend()), to_string(B->backend()));
  const auto backend = A->backend(); // before A is moved into the argument list
  return EigensolverRegistry<T>::instance().create_for_backend("eigensolver", config, "spectra", backend, std::move(A), std::move(B));
}
} // namespace ddm
