#pragma once

#include "ddm/check.hh"
#include "ddm/coarse_space.hh"
#include "ddm/impl/prec/coarse_spaces/geneo.hh"
#include "ddm/impl/prec/coarse_spaces/msgfem.hh"
#include "ddm/impl/prec/coarse_spaces/nicolaides.hh"
#include "ddm/impl/prec/coarse_spaces/ring.hh"
#include "ddm/impl/prec/galerkin.hh"
#include "ddm/impl/prec/schwarz.hh"
#include "ddm/logger.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/solvercategory.hh>
#include <memory>
#include <string>
#include <utility>

namespace ddm {
/** Two-level Schwarz preconditioner. With M^{-1} the fine level (see SchwarzPrec) and Q = R_0^T A_0^{-1} R_0 the
 *  coarse correction (see CoarseCorrection), it applies
 *  - additive: z = M^{-1} r + Q r
 *  - deflated: y = M^{-1} r, z = y + Q (r - A y)
 *
 *  Config:
 *  - coarse_mode:       additive | deflated (default additive)
 *  - coarse_space_type: nicolaides | geneo | msgfem | geneo_ring | msgfem_ring (default nicolaides). With several local
 *                       subdomains per rank (fine.local_subdomains), only nicolaides and geneo are supported so far
 *  - overlap:           overlap of the fine level, takes precedence over fine.overlap
 *  - pou:               partition of unity of both levels, takes precedence over fine.pou
 *  - fine:              config of the fine level (see SchwarzPrec)
 *  - coarse:            config of the coarse correction (see CoarseCorrection)
 *  - coarse_space:      config of the coarse space (see geneo_coarse_space(), msgfem_coarse_space(),
 *                       ring_coarse_space())
 */
template <class T>
class TwoLevelSchwarzPrec final : public Prec<T> {
public:
  // Collective
  TwoLevelSchwarzPrec(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
      , setup_event_(start_setup_event())
      , deflated_(parse_coarse_mode(config.get("coarse_mode", "additive")))
      , schwarz_(fine_config(config), this->mat(), /*defer_factorization=*/true)
      , tmp_(this->mat()->create_domain_vector())
      , res_(this->mat()->create_range_vector())
      , apply_event_(Logger::get().registerOrGetEvent("Two-level Schwarz", "apply"))
  {
    // The fine level is set up except for the factorisation of the subdomain matrices. It is done before the coarse
    // space if the coarse space uses the subdomain solver (harmonic extensions from the outermost layer), otherwise
    // only after the coarse correction: then the factorisation does not add to the peak memory of the coarse space
    // (e.g. the factorisations of GenEO's or the saddle point method's eigensolver), as if it were done on the first
    // apply, but is still timed as part of the setup. With phase barriers (see Logger::set_phase_barriers()), the
    // factorisation, the coarse space and the coarse correction start on all ranks at the same time.
    auto& logger = Logger::get();
    const MPI_Comm comm = schwarz_.get_overlap()->comm->communication_pattern().communicator();
    logger.phase_barrier(comm);

    const auto coarse_space_type = config.get("coarse_space_type", "nicolaides");
    DDM_ASSERT(schwarz_.get_local_subdomains() == 1 || coarse_space_type == "nicolaides" || coarse_space_type == "geneo",
               "two-level schwarz: several local subdomains per rank ({}) are only supported with the nicolaides and geneo coarse spaces", schwarz_.get_local_subdomains());
    const bool needs_fine_solver = schwarz_.eliminates_boundary() && coarse_space_uses_fine_solver(coarse_space_type, config.sub("coarse_space"));
    if (needs_fine_solver) {
      schwarz_.setup_solvers();
      logger.phase_barrier(comm);
    }
    const auto fine_solver = needs_fine_solver ? schwarz_.get_subdomain_solvers()[0] : nullptr;

    logger.startEvent(logger.registerOrGetEvent("Two-level Schwarz", "coarse space"));
    if (coarse_space_type == "nicolaides") {
      const auto pous = schwarz_.get_pous();
      coarse_space_ = nicolaides_coarse_space<T>(schwarz_.get_overlap(), schwarz_.get_subdomains(), pous, *this->mat(), *schwarz_.get_full_overlapping_mat());
    }
    else if (coarse_space_type == "geneo") {
      const auto pous = schwarz_.get_pous();
      coarse_space_ = geneo_coarse_space<T>(config.sub("coarse_space"), schwarz_.get_overlap(), schwarz_.get_subdomains(), pous, this->mat(), *schwarz_.get_full_overlapping_mat());
    }
    else if (coarse_space_type == "msgfem")
      coarse_space_ = msgfem_coarse_space<T>(config.sub("coarse_space"), schwarz_.get_overlap(), *schwarz_.get_pous()[0], this->mat(), schwarz_.get_overlapping_mats()[0], fine_solver);
    else if (coarse_space_type == "geneo_ring" || coarse_space_type == "msgfem_ring")
      coarse_space_ = ring_coarse_space<T>(config.sub("coarse_space"), schwarz_.get_overlap(), *schwarz_.get_pous()[0], this->mat(), schwarz_.get_overlapping_mats()[0], fine_solver,
                                           coarse_space_type == "msgfem_ring");
    else DDM_CHECK(false, "TwoLevelSchwarzPrec: unknown coarse space type {}", coarse_space_type);
    logger.endEvent(logger.registerOrGetEvent("Two-level Schwarz", "coarse space"));
    logger.set_stat("Coarse space/basis size", static_cast<double>(coarse_space_->size()));
    logger.phase_barrier(comm);

    // The matrix of every block of the coarse space: the matrix on the whole overlapping index set for a single block
    // that covers all indices, otherwise the matrix of the block's local subdomain
    std::vector<const LocalMat<T>*> A_blocks;
    if (coarse_space_->blocks.size() == 1 && coarse_space_->blocks[0].covers_all()) A_blocks.push_back(schwarz_.get_full_overlapping_mat().get());
    else
      for (const auto& A_sub : schwarz_.get_overlapping_mats()) A_blocks.push_back(&A_sub->local());
    coarse_correction_ = std::make_unique<CoarseCorrection<T>>(config.sub("coarse"), this->mat(), A_blocks, coarse_space_);
    logger.phase_barrier(comm);

    if (!needs_fine_solver) {
      schwarz_.setup_solvers();
      logger.phase_barrier(comm);
    }
    logger.endEvent(setup_event_);
  }

  // On a single rank, the matrix is sequential, and so is the preconditioner
  Dune::SolverCategory::Category category() const override
  {
    return this->mat()->category();
  }

private:
  // Registers and starts the setup event. Called in the member initializer list, so the event also covers the
  // construction of the fine level
  static Logger::Event* start_setup_event()
  {
    auto* event = Logger::get().registerOrGetEvent("Two-level Schwarz", "setup");
    Logger::get().startEvent(event);
    return event;
  }

  // True if the coarse space uses the fine-level subdomain solver (with eliminate_boundary it computes the harmonic
  // extensions from the outermost layer): MS-GFEM with every method except saddle_point, and the ring coarse spaces
  // with the extension from_boundary. Must match the uses in msgfem_coarse_space() and ring_coarse_space().
  static bool coarse_space_uses_fine_solver(const std::string& type, const Dune::ParameterTree& config)
  {
    if (type == "msgfem") return config.get("method", std::string("saddle_point")) != "saddle_point";
    if (type == "geneo_ring" || type == "msgfem_ring") return config.get("extension", std::string("from_boundary")) == "from_boundary";
    return false;
  }

  static bool parse_coarse_mode(const std::string& mode)
  {
    DDM_CHECK(mode == "additive" || mode == "deflated", "two-level schwarz: unknown coarse_mode '{}', expected 'additive' or 'deflated'", mode);
    return mode == "deflated";
  }

  // The config of the fine level: the subtree "fine", but the overlap and the partition of unity can also be set on the
  // two-level method itself (key "overlap", subtree "pou"), which then take precedence. Unlike for the one-level method, overlap 0 is not allowed: the
  // coarse space needs a partition of unity that vanishes on the outermost layer.
  static Dune::ParameterTree fine_config(const Dune::ParameterTree& config)
  {
    auto fine = config.sub("fine");
    if (config.hasKey("overlap")) fine["overlap"] = config["overlap"];
    if (config.hasSub("pou")) {
      const auto& pou = config.sub("pou");
      for (const auto& key : pou.getValueKeys()) fine["pou." + key] = pou[key];
    }
    const int overlap = fine.get("overlap", 1);
    DDM_CHECK(overlap >= 1, "two-level schwarz: the overlap must be at least 1, got {}", overlap);
    return fine;
  }

  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    Logger::ScopedLog sl{apply_event_};

    // Both apply() overwrite their output completely
    schwarz_.apply(z, r);
    if (deflated_) {
      // res = r - A z, consistent
      res_ = r;
      this->mat()->applyscaleadd(T{-1}, z, res_);
      coarse_correction_->apply(tmp_, res_);
    }
    else coarse_correction_->apply(tmp_, r);
    z += tmp_;
  }

  void do_update() override
  {
    TODO("do_update");
  }

  void do_info() const override
  {
    logger::info("Two-level Schwarz preconditioner ({} coarse correction)", deflated_ ? "deflated" : "additive");
    {
      logger::increase_indent();
      logger::info("Fine-level preconditioner");
      {
        logger::increase_indent();
        schwarz_.info();
        logger::decrease_indent();
      }
      logger::decrease_indent();
    }
    {
      logger::increase_indent();
      logger::info("Coarse-level preconditioner");
      {
        logger::increase_indent();
        coarse_correction_->info();
        logger::decrease_indent();
      }
      logger::decrease_indent();
    }
  }

  // The coarse level is built from the overlapping index set and the overlapping matrix of the fine level, so the
  // fine level must be constructed first: do not change the order of the members. The setup event is started in the
  // member initializer list, so it comes first to cover the whole construction.
  Logger::Event* setup_event_;
  bool deflated_;
  SchwarzPrec<T> schwarz_;
  std::shared_ptr<CoarseSpace<T>> coarse_space_;
  std::unique_ptr<CoarseCorrection<T>> coarse_correction_;

  Vec<T> tmp_; ///< the coarse correction in apply()
  Vec<T> res_; ///< the residual of the fine-level correction in apply() (deflated only)

  Logger::Event* apply_event_;
};
} // namespace ddm
