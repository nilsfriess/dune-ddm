#pragma once

#include "ddm/check.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/solver/solver.hh"

#include <memory>

namespace ddm {
// A solver that applies one step of a preconditioner. Usually that preconditioner is a direct solver
template <class T>
class DirectSolver : public Solver<T> {
public:
  DirectSolver(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, std::shared_ptr<Prec<T>> P)
      : Solver<T>(config, std::move(A), std::move(P))
  {
    DDM_CHECK(this->prec() != nullptr, "The DirectSolver class must be used with a preconditioner");
  }

private:
  void do_solve(const Vec<T>& b, Vec<T>& x) override { this->prec()->apply(x, b); }

  void do_info() const override { logger::info("Direct solver (applies the preconditioner once)"); }
};
} // namespace ddm
