#pragma once

#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/solver/solver.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/solvers.hh>
#include <memory>
#include <utility>

namespace ddm {
template <class T>
class GMResSolver final : public Solver<T> {
  using DuneSolver = Dune::RestartedGMResSolver<Vec<T>, Vec<T>>;

public:
  GMResSolver(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, std::shared_ptr<Prec<T>> P)
      : Solver<T>(std::move(A), std::move(P))
  {
    auto sp = std::make_shared<Dune::ScalarProduct<Vec<T>>>();

    auto reduction = this->reduction(config);
    auto restart = config.get("restart", 30);
    auto maxit = this->maxit(config);
    auto verbose = this->verbosity(config);
    solver = std::make_unique<DuneSolver>(this->mat(), sp, this->prec(), reduction, restart, maxit, verbose);
  }

private:
  void do_solve(const Vec<T>& b, Vec<T>& x) override
  {
    Vec<T> bcopy = b; // Dune solvers don't take b as const, so we have to copy here...
    Dune::InverseOperatorResult res;
    solver->apply(x, bcopy, res);
  }

  std::unique_ptr<DuneSolver> solver;
};
} // namespace ddm
