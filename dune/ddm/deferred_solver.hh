#pragma once

#include <dune/common/parametertree.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solvercategory.hh>
#include <dune/istl/solverfactory.hh>
#include <memory>

namespace ddm {
template <class Operator>
class DeferredSolver : public Dune::InverseOperator<typename Operator::domain_type, typename Operator::range_type> {
  using Solver = Dune::InverseOperator<typename Operator::domain_type, typename Operator::range_type>;

public:
  DeferredSolver(std::shared_ptr<Operator> op_, Dune::ParameterTree config_)
      : op(std::move(op_))
      , config(std::move(config_))
  {
  }

  Dune::SolverCategory::Category category() const override
  {
    // TODO: we shouldn't just hard-code this but I don't know what would be a better solution
    return Dune::SolverCategory::sequential;
  }

  void apply(typename Solver::domain_type& x, typename Solver::range_type& b, Dune::InverseOperatorResult& res) override
  {
    if (!solver) init_solver();
    solver->apply(x, b, res);
  }

  void apply(typename Solver::domain_type& x, typename Solver::range_type& b, double reduction, Dune::InverseOperatorResult& res) override
  {
    if (!solver) init_solver();
    solver->apply(x, b, reduction, res);
  }

  std::shared_ptr<Solver> get_solver()
  {
    if (!solver) init_solver();
    return solver;
  }

private:
  void init_solver()
  {
    solver = Dune::getSolverFromFactory(op, config);
    op.reset(); // op is not needed anymore by us
  }

  std::shared_ptr<Operator> op;
  Dune::ParameterTree config;

  std::shared_ptr<Solver> solver = nullptr;
};

/** @brief Creates a solver using Dune's solver factory but defers the actual setup to the first call to apply
 *
 *  The purpose of this function is to be able to create a solver that might never actually be used
 *  so a potentially expensive setup (e.g. a matrix factorisation) is only ever executed if it is
 *  really needed.
 *
 *  Only solvers without preconditioners are supported for now.
 */
template <class Operator>
std::shared_ptr<Dune::InverseOperator<typename Operator::domain_type, typename Operator::range_type>> getDeferredSolverFromFactory(std::shared_ptr<Operator> op, const Dune::ParameterTree& config)
{
  return std::make_shared<DeferredSolver<Operator>>(op, config);
}
} // namespace ddm
