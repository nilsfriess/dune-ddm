#pragma once

#include "ddm/logger.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/registry.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <functional>
#include <memory>
#include <optional>

namespace ddm {
/** Solver for the generalised eigenproblem A x = lambda B x, computing the eigenpairs with the smallest eigenvalues.
 *
 *  Config:
 *  - type:      spectra (default) | lanczos, see SpectraEigensolver and LanczosEigensolver
 *  - nev:       number of eigenpairs to compute (default 16), the minimum number if threshold is set
 *  - ncv:       number of Lanczos vectors (default 2 * nev), the initial number if threshold is set
 *  - maxit:     maximum number of iterations (default 1000)
 *  - tolerance: convergence tolerance (default 1e-5)
 *  - shift:     shift sigma of the shift-and-invert operator (A - sigma B)^{-1} (default -1e-3). Negative, so that for
 *               positive semi-definite A and B without a common kernel A - sigma B is positive definite and all
 *               eigenvalues lie above the shift
 *  - threshold: if set, the number of eigenpairs is not fixed: all eigenpairs with eigenvalues below the threshold tau
 *               are computed, but at least nev and at most nev_max. More precisely, these are the eigenvalues with
 *               |lambda - sigma| < |tau - sigma|, the same if the shift lies below the spectrum. Must lie above the
 *               shift. Only supported by the lanczos eigensolver, which adds Lanczos vectors as needed, so a ncv above
 *               the expected number of eigenpairs saves reallocations
 *  - nev_max:   the maximum number of eigenpairs with a threshold (default 4 * ncv)
 *  - print_eigenvalues: print the computed eigenvalues on every rank, at every log level except off (default false)
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
      , nev_max_(config.get("nev_max", 4 * ncv_))
      , maxit_(config.get("maxit", 1000))
      , tolerance_(config.get("tolerance", typename Vec<T>::real_type(1e-5)))
      , shift_(config.get("shift", typename Vec<T>::real_type(-1e-3)))
      , threshold_(config.hasKey("threshold") ? std::optional(config.get<typename Vec<T>::real_type>("threshold")) : std::nullopt)
      , print_eigenvalues_(config.get("print_eigenvalues", false))
  {
    DDM_CHECK(A_->rows() == B_->rows(), "Eigensolver: size mismatch, A has {} rows, B has {}", A_->rows(), B_->rows());
    DDM_CHECK(A_->cols() == B_->cols(), "Eigensolver: size mismatch, A has {} cols, B has {}", A_->cols(), B_->cols());
    DDM_CHECK(nev_ >= 1, "Eigensolver: nev must be at least 1, got {}", nev_);
    DDM_CHECK(ncv_ > nev_, "Eigensolver: ncv must be larger than nev, got ncv = {} and nev = {}", ncv_, nev_);
    if (threshold_) {
      DDM_CHECK(nev_max_ >= nev_, "Eigensolver: nev_max must not be smaller than nev, got nev_max = {} and nev = {}", nev_max_, nev_);
      DDM_CHECK(*threshold_ > shift_, "Eigensolver: the threshold must lie above the shift, got threshold {} and shift {}", *threshold_, shift_);
    }
  }

  virtual ~Eigensolver() = default;

  // Restricts the iteration to the range of P, which must be a projection (P P = P). P is applied after every
  // shift-invert solve and to the starting vector. If its transpose P_transposed is given as well, it is applied
  // before every shift-invert solve, so that the operator becomes P (A - sigma B)^{-1} P^T B, which is self-adjoint in
  // the B inner product for every shift. Y = P X is applied to all columns of X at once, X and Y are distinct.
  using Projection = std::function<void(const MultiVec<T>& X, MultiVec<T>& Y)>;
  void set_projection(Projection P, Projection P_transposed = {})
  {
    projection_ = std::move(P);
    projection_transposed_ = std::move(P_transposed);
  }

  // Replaces (A - sigma B)^{-1} in the shift-invert operator: make(sigma) is called once with the shift before the
  // iteration and returns the function that is applied instead. A is then only used for its size. The replacement S
  // must be self-adjoint, so that the operator P S P^T B stays self-adjoint in the B inner product. X = S B is applied
  // to all columns of B at once, B and X are distinct.
  using ShiftInvert = std::function<void(const MultiVec<T>& B, MultiVec<T>& X)>;
  using ShiftInvertFactory = std::function<ShiftInvert(typename Vec<T>::real_type sigma)>;
  void set_shift_invert(ShiftInvertFactory make)
  {
    shift_invert_ = std::move(make);
  }

  // Computes the eigenpairs, see values() and vectors()
  void solve()
  {
    do_solve();
    if (print_eigenvalues_) logger::print_all("Eigenvalues: {}", logger::join(values()));
  }

  virtual const std::vector<T>& values() const = 0;
  virtual std::shared_ptr<MultiVec<T>> vectors() const = 0;

  const std::shared_ptr<const LocalMat<T>>& A() const
  {
    return A_;
  }
  const std::shared_ptr<const LocalMat<T>>& B() const
  {
    return B_;
  }
  int nev() const
  {
    return nev_;
  }
  int ncv() const
  {
    return ncv_;
  }
  int nev_max() const
  {
    return nev_max_;
  }
  int maxit() const
  {
    return maxit_;
  }
  typename Vec<T>::real_type tolerance() const
  {
    return tolerance_;
  }
  typename Vec<T>::real_type shift() const
  {
    return shift_;
  }
  std::optional<typename Vec<T>::real_type> threshold() const
  {
    return threshold_;
  } // empty if not set

protected:
  const Projection& projection() const
  {
    return projection_;
  } // empty if not set
  const Projection& projection_transposed() const
  {
    return projection_transposed_;
  } // empty if not set

  // Sets up S = (A - sigma B)^{-1}, or its replacement if one was set with set_shift_invert(), and returns it. The solver
  // for A - sigma B is created with create_subproblem_solver() from the subtrees "solver" and "prec" of config. Called
  // once per solve(), S lives as long as the returned function. Timed as the event "Eigensolver"/"factorize" (which
  // lies inside "Eigensolver"/"solve").
  ShiftInvert make_shift_invert(const Dune::ParameterTree& config, typename Vec<T>::real_type sigma) const
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Eigensolver", "factorize")};
    if (shift_invert_) return shift_invert_(sigma);
    auto shifted = std::make_shared<const Mat<T>>(Dune::ParameterTree{}, add(*A_, T(-sigma), *B_), nullptr);
    std::shared_ptr<Solver<T>> solver = create_subproblem_solver<T>(config.sub("solver"), config.sub("prec"), std::move(shifted));
    return [solver](const MultiVec<T>& b, MultiVec<T>& x) {
      x.zero();
      solver->solve(b, x);
    };
  }

private:
  virtual void do_solve() = 0;

  std::shared_ptr<const LocalMat<T>> A_;
  std::shared_ptr<const LocalMat<T>> B_;

  int nev_;
  int ncv_;
  int nev_max_;
  int maxit_;
  typename Vec<T>::real_type tolerance_;
  typename Vec<T>::real_type shift_;
  std::optional<typename Vec<T>::real_type> threshold_;
  bool print_eigenvalues_;

  Projection projection_;
  Projection projection_transposed_;
  ShiftInvertFactory shift_invert_;
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
