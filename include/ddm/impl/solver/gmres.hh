#pragma once

#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/solver/solver.hh"
#include "dune/ddm/logger.hh"
#include "scalar_product.hh"

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
      : Solver<T>(config, std::move(A), std::move(P))
  {
    std::shared_ptr<Dune::ScalarProduct<Vec<T>>> sp;
    if (this->mat()->sequential()) sp = std::make_shared<Dune::ScalarProduct<Vec<T>>>();
    else sp = std::make_shared<ConsistentScalarProduct<T>>(this->mat()->communication());

    restart_ = config.get("restart", 30);
    solver = std::make_unique<DuneSolver>(this->mat(), sp, this->prec(), this->reduction(), restart_, this->maxit(), this->verbosity());
  }

private:
  void do_solve(const Vec<T>& b, Vec<T>& x) override
  {
    Vec<T> bcopy = b; // Dune solvers don't take b as const, so we have to copy here...
    Dune::InverseOperatorResult res;
    solver->apply(x, bcopy, res);
  }

  void do_info() const override
  {
    logger::info("GMRes solver (using DUNE ISTL's RestartedGMResSolver)");
    logger::info("Restart: {}", restart_);
  }

  std::unique_ptr<DuneSolver> solver;
  int restart_;
};
} // namespace ddm
