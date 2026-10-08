#pragma once

#include "ddm/check.hh"
#include "ddm/mat/mat.hh"
#include "ddm/overlap.hh"
#include "ddm/prec/prec.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"
#include "dune/ddm/logger.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/solvercategory.hh>
#include <memory>
#include <optional>
#include <utility>

namespace ddm {
/** Additive Schwarz preconditioner: z = sum_i R_i^T A_i^{-1} R_i r, where R_i restricts to the overlapping subdomain
 *  of rank i and A_i = R_i A R_i^T.
 *
 *  The overlapping subdomains are built from the index set of the matrix's communication, extended along the pattern
 *  of the local matrix. A_i^{-1} is applied by a sequential solver, so it is inexact unless that solver is exact.
 *
 *  Config:
 *  - overlap:          number of layers added to the index set of the matrix (default 1)
 *  - subdomain_mat:    config of the subdomain matrix A_i (see create_local_mat_like(), default: the type of A's local
 *                      matrix)
 *  - subdomain_solver: config of the solver for A_i (see create_subproblem_solver(), default: exact solve)
 *  - subdomain_prec:   config of the preconditioner of that solver (see create_subproblem_solver())
 */
template <class T>
class SchwarzPrec final : public Prec<T> {
public:
  // Collective
  SchwarzPrec(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
      , config_(config)
      , layers_(config.get("overlap", 1))
      , ovlp_(std::make_shared<const Overlap>(extend_overlap(communication(), this->mat()->local().pattern(), layers_)))
      , apply_event_(Logger::get().registerOrGetEvent("Schwarz", "apply"))
      , solve_event_(Logger::get().registerOrGetEvent("Schwarz", "subdomain solve"))
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Schwarz", "setup")};
    do_update();
  }

  // On a single rank, the matrix is sequential, and so is the preconditioner
  Dune::SolverCategory::Category category() const override { return this->mat()->category(); }

  // Get the overlap object, containing the overlapping communication and the layer information
  std::shared_ptr<const Overlap> get_overlap() const { return ovlp_; }

  // Get the overlapping matrix
  std::shared_ptr<const Mat<T>> get_overlapping_mat() const { return A_sub_; }

private:
  // Called while the members are initialized, so the check has to happen here and not in the constructor body
  const Communication& communication() const
  {
    DDM_CHECK(this->mat()->communication() != nullptr, "schwarz: the matrix has no communication");
    return *this->mat()->communication();
  }

  // Collective
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    Logger::ScopedLog sl{apply_event_};

    // r is consistent, so the owners of the overlap indices have the right values
    d_->copy_n_from(r, ovlp_->n_original);
    ovlp_->comm->broadcast(*d_);

    Logger::get().startEvent(solve_event_);
    x_->zero();
    solver_->solve(*d_, *x_);
    Logger::get().endEvent(solve_event_);

    // Sums the subdomain corrections over all subdomains containing an index, which makes the result consistent
    ovlp_->comm->reduce(*x_);
    z.copy_n_from(*x_, ovlp_->n_original);
  }

  // Collective. The overlapping index set is kept, the subdomain matrix and its solver are rebuilt
  void do_update() override { setup_subdomain(); }

  void do_info() const override
  {
    logger::info("Overlapping Schwarz preconditioner");
    logger::info("Overlap: {}", layers_);
    {
      logger::increase_indent();
      logger::info("Subdomain solver info");
      {
        logger::increase_indent();
        solver_->info();
        logger::decrease_indent();
      }
      logger::decrease_indent();
    }
  }

  void setup_subdomain()
  {
    A_sub_ = std::make_shared<const Mat<T>>(overlapping_matrix(config_.sub("subdomain_mat"), *this->mat(), *ovlp_), nullptr);
    solver_ = create_subproblem_solver<T>(config_.sub("subdomain_solver"), config_.sub("subdomain_prec"), A_sub_);
    d_.emplace(A_sub_->create_range_vector());
    x_.emplace(A_sub_->create_domain_vector());
  }

  Dune::ParameterTree config_;
  int layers_;
  std::shared_ptr<const Overlap> ovlp_;

  std::shared_ptr<const Mat<T>> A_sub_;
  std::shared_ptr<Solver<T>> solver_;
  std::optional<Vec<T>> d_; ///< defect on the overlapping subdomain
  std::optional<Vec<T>> x_; ///< correction on the overlapping subdomain

  Logger::Event* apply_event_{nullptr};
  Logger::Event* solve_event_{nullptr};
};
} // namespace ddm
