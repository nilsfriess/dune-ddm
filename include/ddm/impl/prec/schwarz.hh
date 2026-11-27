#pragma once

#include "ddm/check.hh"
#include "ddm/debug/vec_sink.hh"
#include "ddm/logger.hh"
#include "ddm/mat/mat.hh"
#include "ddm/overlap.hh"
#include "ddm/pou.hh"
#include "ddm/prec/prec.hh"
#include "ddm/region.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/solvercategory.hh>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ddm {
/** Additive Schwarz preconditioner: z = sum_i R_i^T A_i^{-1} R_i r, where R_i restricts to the overlapping subdomain
 *  of rank i and A_i = R_i A R_i^T. The restricted variant (RAS) weights the subdomain corrections with the partition
 *  of unity: z = sum_i R_i^T D_i A_i^{-1} R_i r.
 *
 *  The overlapping subdomains are built from the index set of the matrix's communication, extended along the pattern
 *  of the local matrix. A_i^{-1} is applied by a sequential solver, so it is inexact unless that solver is exact. The
 *  solver is set up (e.g. A_i is factorised) in the constructor, or with defer_factorization on the first call of
 *  setup_solvers(), get_subdomain_solvers() or apply(). The owner can then choose when the factorisation lives in memory:
 *  TwoLevelSchwarzPrec only sets it up after a coarse space that does not need it (e.g. GenEO), so that it does not add
 *  to the peak memory of the eigensolver. The setup is timed as the event "Schwarz"/"factorize", and also counts as
 *  "Schwarz"/"setup", which therefore always contains the factorisation.
 *
 *  With eliminate_boundary, the subdomain problems have homogeneous Dirichlet conditions on the outermost layer of the
 *  overlapping index set: A_i gets identity rows there (only the rows, so A_i can also be used for harmonic extensions)
 *  and the defect is set to zero there, so the subdomain correction vanishes on that layer.
 *
 *  Config:
 *  - variant:            additive | restricted (default additive)
 *  - overlap:            number of layers added to the index set of the matrix (default 1, restricted needs at least 1)
 *  - local_subdomains:   number of subdomains per MPI rank (default 1), see partition_matrix_with_overlap(). Only 1 is
 *                        supported by TwoLevelSchwarzPrec
 *  - eliminate_boundary: eliminate the outermost layer, see above (default true, only used with overlap >= 1)
 *  - pou:                config of the partition of unity (see partition_of_unity()), only used with overlap >= 1
 *  - subdomain_mat:      config of the subdomain matrix A_i (see create_local_mat_like(), default: the type of A's local
 *                        matrix)
 *  - subdomain_solver:   config of the solver for A_i (see create_subproblem_solver(), default: exact solve)
 *  - subdomain_prec:     config of the preconditioner of that solver (see create_subproblem_solver())
 */
template <class T>
class SchwarzPrec final : public Prec<T> {
public:
  // Collective
  SchwarzPrec(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, bool defer_factorization = false)
      : Prec<T>(std::move(A))
      , config_(config)
      , defer_factorization_(defer_factorization)
      , restricted_(parse_variant(config.get("variant", "additive")))
      , layers_(config.get("overlap", 1))
      , eliminate_boundary_(config.get("eliminate_boundary", true) && layers_ >= 1)
      , local_subdomains_(config.get("local_subdomains", 1))
      , ovlp_(std::make_shared<const Overlap>(extend_overlap(communication(), this->mat()->local().pattern(), layers_)))
      , apply_event_(Logger::get().registerOrGetEvent("Schwarz", "apply"))
      , solve_event_(Logger::get().registerOrGetEvent("Schwarz", "subdomain solve"))
      , setup_event_(Logger::get().registerOrGetEvent("Schwarz", "setup"))
      , factorize_event_(Logger::get().registerOrGetEvent("Schwarz", "factorize"))
  {
    DDM_CHECK(!restricted_ || layers_ >= 1, "schwarz: the restricted variant needs an overlap of at least 1, got {}", layers_);
    do_update();
  }

  // On a single rank, the matrix is sequential, and so is the preconditioner
  Dune::SolverCategory::Category category() const override
  {
    return this->mat()->category();
  }

  // Get the overlap object, containing the overlapping communication and the layer information
  std::shared_ptr<const Overlap> get_overlap() const
  {
    return ovlp_;
  }

  // Get the overlapping matrices
  std::vector<std::shared_ptr<const Mat<T>>> get_overlapping_mats() const
  {
    return A_subs_;
  }

  // Get the matrix on the overlapping index set of the rank, i.e. on the union of the local subdomains. With a single
  // local subdomain, this is the matrix of that subdomain (so with eliminate_boundary, it has identity rows on the
  // outermost layer)
  std::shared_ptr<const LocalMat<T>> get_full_overlapping_mat() const
  {
    return A_full_;
  }

  // Get the local subdomains (see partition_matrix_with_overlap())
  const std::vector<LocalSubdomain>& get_subdomains() const
  {
    return subdomains_;
  }

  // Get the partitions of unity on the local subdomains (empty with overlap 0)
  std::vector<std::shared_ptr<const Vec<T>>> get_pous() const
  {
    return pous_;
  }

  // Get the number of local subdomains per MPI rank
  int get_local_subdomains() const
  {
    return local_subdomains_;
  }

  // Get the subdomain solvers, sets them up if that has not happened yet (see setup_solvers())
  std::vector<std::shared_ptr<Solver<T>>> get_subdomain_solvers()
  {
    setup_solvers();
    return solvers_;
  }

  // Sets up the subdomain solvers (e.g. factorises A_i) if that has not happened yet. Not collective, the subdomain
  // solver is sequential
  void setup_solvers()
  {
    if (not solvers_.empty()) return;
    Logger::ScopedLog setup{setup_event_};
    Logger::ScopedLog factorize{factorize_event_};
    for (Index i = 0; i < local_subdomains_; ++i) solvers_.emplace_back(create_subproblem_solver<T>(config_.sub("subdomain_solver"), config_.sub("subdomain_prec"), A_subs_[i]));
  }

  // True if the outermost layer is eliminated (see eliminate_boundary), i.e. the subdomain solver also computes harmonic
  // extensions from that layer
  bool eliminates_boundary() const
  {
    return eliminate_boundary_;
  }

private:
  static bool parse_variant(const std::string& variant)
  {
    DDM_CHECK(variant == "additive" || variant == "restricted", "schwarz: unknown variant '{}', expected 'additive' or 'restricted'", variant);
    return variant == "restricted";
  }

  // Called while the members are initialized, so the check has to happen here and not in the constructor body
  const Communication& communication() const
  {
    DDM_CHECK(this->mat()->communication() != nullptr, "schwarz: the matrix has no communication");
    return *this->mat()->communication();
  }

  // Collective
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    setup_solvers(); // before the apply event starts, so a deferred setup is not counted as apply time
    Logger::ScopedLog sl{apply_event_};

    // r is consistent, so the owners of the overlap indices have the right values
    d_full_->copy_n_from(r, ovlp_->n_original);
    ovlp_->comm->broadcast(*d_full_);
    x_full_->zero();

    if (local_subdomains_ == 1) {
      // A single local subdomain is the whole overlapping subdomain, so the solve works on d_full_ and x_full_ directly
      if (eliminate_boundary_) boundary_masks_[0].pointwise_mult(*d_full_, *d_full_);
      Logger::get().startEvent(solve_event_);
      solvers_[0]->solve(*d_full_, *x_full_);
      Logger::get().endEvent(solve_event_);
      if (restricted_) pous_[0]->pointwise_mult(*x_full_, *x_full_);
    }
    else {
      // Restrict the defect to each local subdomain, solve there and add up the corrections in x_full_
      for (std::size_t j = 0; j < solvers_.size(); ++j) {
        const auto indices = subdomains_[j].region.indices();
        d_full_->multivec().pack_rows(indices, ds_[j].multivec());
        if (eliminate_boundary_) boundary_masks_[j].pointwise_mult(ds_[j], ds_[j]);
        xs_[j].zero();
        Logger::get().startEvent(solve_event_);
        solvers_[j]->solve(ds_[j], xs_[j]);
        Logger::get().endEvent(solve_event_);
        if (restricted_) pous_[j]->pointwise_mult(xs_[j], xs_[j]);
        x_full_->multivec().add_rows(xs_[j].multivec(), indices);
      }
    }

    // Sums the subdomain corrections over all subdomains containing an index, which makes the result consistent
    ovlp_->comm->reduce(*x_full_);
    z.copy_n_from(*x_full_, ovlp_->n_original);
  }

  // Collective. The overlapping index set is kept, the subdomain matrix is rebuilt and its solver is set up again (with
  // defer_factorization only on its next use)
  void do_update() override
  {
    {
      Logger::ScopedLog sl{setup_event_};
      setup_subdomains();
    }
    if (!defer_factorization_) setup_solvers();
  }

  void do_info() const override
  {
    logger::info("Overlapping Schwarz preconditioner ({})", restricted_ ? "restricted" : "additive");
    logger::info("Overlap: {}", layers_);
    logger::info("Eliminate boundary: {}", eliminate_boundary_);
    logger::info("Local subdomains: {}", local_subdomains_);
    {
      logger::increase_indent();
      logger::info("Subdomain solver info");
      {
        logger::increase_indent();
        logger::info("Only for local subdomain 0");
        if (not solvers_.empty()) solvers_[0]->info();
        else logger::info("Not set up yet");
        logger::decrease_indent();
      }
      logger::decrease_indent();
    }
  }

  void setup_subdomains()
  {
    auto A_full = overlapping_matrix(config_.sub("subdomain_mat"), *this->mat(), *ovlp_);
    // The local subdomains only depend on the pattern, so like the overlapping index set they are kept on update
    if (subdomains_.empty()) subdomains_ = partition_matrix_with_overlap(*A_full, *ovlp_, local_subdomains_);

    A_subs_.clear();
    boundary_masks_.clear();
    ds_.clear();
    xs_.clear();
    pous_.clear();
    solvers_.clear(); // belongs to the old subdomain matrices, see setup_solvers()
    for (const auto& sd : subdomains_) {
      std::shared_ptr<LocalMat<T>> A_sub;
      if (sd.region.is_subdomain()) A_sub = A_full; // a single local subdomain is the whole overlapping subdomain
      else A_sub = submatrix(*A_full, sd.region);
      if (eliminate_boundary_) {
        std::vector<Index> boundary;
        for (Index k = 0; k < std::ssize(sd.layer); ++k)
          if (sd.layer[k] == layers_) boundary.push_back(k);
        A_sub->zero_rows(boundary);

        // 1 everywhere, 0 on the eliminated layer, to zero the defect there in apply()
        auto& mask = boundary_masks_.emplace_back(A_sub->create_range_vector());
        mask = T{1};
        {
          auto view = mask.host_view(read_write);
          for (auto k : boundary) view[k] = T{0};
        }
      }
      A_subs_.emplace_back(std::make_shared<const Mat<T>>(config_.sub("subdomain_mat"), std::move(A_sub), nullptr));
      if (local_subdomains_ > 1) {
        ds_.emplace_back(A_subs_.back()->create_range_vector());
        xs_.emplace_back(A_subs_.back()->create_domain_vector());
      }
    }
    // TODO(20261127-134424): Log subdomain dof numbers for all subdomains
    Logger::get().set_stat("Schwarz/subdomain dofs", static_cast<double>(A_subs_[0]->rows()));
    A_full_ = A_full;
    d_full_.emplace(A_full->create_range_vector());
    x_full_.emplace(A_full->create_domain_vector());

    if (layers_ >= 1) {
      for (auto& pou : partition_of_unity(config_.sub("pou"), *ovlp_, subdomains_, *A_full)) pous_.push_back(std::make_shared<const Vec<T>>(std::move(pou)));

      // Write the POU vector to the debug sink: the sum over the local subdomains of the debug rank
      if (debug::has_vec_sink<T>()) {
        auto pou_ovlp = A_full->create_domain_vector();
        if (debug::vec_sink_debug_rank<T>() == this->mat()->communication()->rank())
          for (std::size_t j = 0; j < subdomains_.size(); ++j) pou_ovlp.multivec().add_rows(pous_[j]->multivec(), subdomains_[j].region.indices());
        ovlp_->comm->reduce(pou_ovlp);
        auto pou_novlp = this->mat()->create_domain_vector();
        pou_novlp.copy_n_from(pou_ovlp, pou_novlp.size());
        debug::add_vec_to_sink("pou", pou_novlp);
      }
    }
  }

  Dune::ParameterTree config_;
  bool defer_factorization_;
  bool restricted_;
  int layers_;
  bool eliminate_boundary_;
  int local_subdomains_;
  std::shared_ptr<const Overlap> ovlp_;

  std::vector<LocalSubdomain> subdomains_;            ///< the local subdomains, see partition_matrix_with_overlap()
  std::shared_ptr<const LocalMat<T>> A_full_;         ///< the matrix on the overlapping index set of the rank (with one local subdomain the same object as A_subs_[0]'s)
  std::vector<std::shared_ptr<const Mat<T>>> A_subs_; ///< A_subs_[j] is the matrix of local subdomain j
  std::vector<std::shared_ptr<Solver<T>>> solvers_;   ///< solvers_[j] is the solver for A_subs_[j], empty until setup_solvers()
  std::vector<std::shared_ptr<const Vec<T>>> pous_;   ///< pous_[j] is the partition of unity on local subdomain j (empty with overlap 0)
  std::optional<Vec<T>> d_full_;                      ///< defect on the overlapping index set of the rank
  std::optional<Vec<T>> x_full_;                      ///< correction on the overlapping index set of the rank
  std::vector<Vec<T>> ds_;                            ///< defects on the local subdomains (only with more than one)
  std::vector<Vec<T>> xs_;                            ///< corrections on the local subdomains (only with more than one)
  std::vector<Vec<T>> boundary_masks_;                ///< per local subdomain: 0 on the eliminated layer, 1 elsewhere (only with eliminate_boundary)

  Logger::Event* apply_event_{nullptr};
  Logger::Event* solve_event_{nullptr};
  Logger::Event* setup_event_{nullptr};
  Logger::Event* factorize_event_{nullptr};
};
} // namespace ddm
