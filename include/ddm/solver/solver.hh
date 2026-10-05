#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/registry.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <memory>
#include <string>
#include <utility>

namespace ddm {
template <class T>
class Solver {
public:
  using value_type = T;

  virtual ~Solver() = default;
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  // Solves A x = b, x is used as the initial guess
  void solve(const Vec<T>& b, Vec<T>& x)
  {
    DDM_CHECK(&b != &x, "solver: solve() requires distinct vectors b and x");
    DDM_CHECK(A_->is_assembled(), "solver: solve() called with unassembled matrix");
    DDM_CHECK(b.size() == A_->rows() && x.size() == A_->cols(), "solver: solve() size mismatch, matrix is {}x{}, but b has size {} and x has size {}", A_->rows(), A_->cols(), b.size(), x.size());
    DDM_CHECK(b.backend() == A_->backend() && x.backend() == A_->backend(), "solver: solve() backend mismatch, matrix is {}, b is {}, x is {}", to_string(A_->backend()), to_string(b.backend()),
              to_string(x.backend()));
    do_solve(b, x);
  }

  const std::shared_ptr<const Mat<T>>& mat() const { return A_; }

  // nullptr if the solver is unpreconditioned
  const std::shared_ptr<Prec<T>>& prec() const { return P_; }

  // Default values or config entry for common values the solvers need
  double reduction(const Dune::ParameterTree& config) const { return config.get("reduction", 1e-8); }
  int maxit(const Dune::ParameterTree& config) const { return config.get("maxit", 5000); }
  int verbosity(const Dune::ParameterTree& config) const { return config.get("verbose", 0); }

protected:
  Solver(std::shared_ptr<const Mat<T>> A, std::shared_ptr<Prec<T>> P)
      : A_(std::move(A))
      , P_(std::move(P))
  {
    DDM_CHECK(A_ != nullptr, "solver: matrix is nullptr");
    DDM_CHECK(A_->rows() == A_->cols(), "solver: matrix must be square, but is {}x{}", A_->rows(), A_->cols());
    if (P_) DDM_CHECK(P_->mat()->rows() == A_->rows(), "solver: preconditioner size {} does not match matrix size {}", P_->mat()->rows(), A_->rows());
  }

private:
  virtual void do_solve(const Vec<T>& b, Vec<T>& x) = 0;

  std::shared_ptr<const Mat<T>> A_;
  std::shared_ptr<Prec<T>> P_;
};

template <class T>
using SolverRegistry = Registry<std::shared_ptr<Solver<T>>, std::shared_ptr<const Mat<T>>, std::shared_ptr<Prec<T>>>;

// Registers a solver that works for any backend
template <class T>
void register_solver(std::string name, typename SolverRegistry<T>::Factory factory)
{
  SolverRegistry<T>::instance().add(std::move(name), std::move(factory));
}

// Registers a solver that is only used for matrices of the given backend. It takes precedence over one with the same
// name that is registered for any backend.
template <class T>
void register_solver(std::string name, BackendId backend, typename SolverRegistry<T>::Factory factory)
{
  SolverRegistry<T>::instance().add(std::move(name), backend, std::move(factory));
}

// P may be nullptr for an unpreconditioned solver
template <class T>
std::shared_ptr<Solver<T>> create_solver(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, std::shared_ptr<Prec<T>> P = nullptr)
{
  initialize();
  DDM_CHECK(A != nullptr, "solver: matrix is nullptr");
  const auto backend = A->backend(); // before A is moved into the argument list
  return SolverRegistry<T>::instance().create_for_backend("solver", config, "gmres", backend, std::move(A), std::move(P));
}
} // namespace ddm
