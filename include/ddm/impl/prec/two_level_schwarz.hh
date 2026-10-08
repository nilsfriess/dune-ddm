#pragma once

#include "ddm/check.hh"
#include "ddm/coarse_space.hh"
#include "ddm/impl/prec/coarse_spaces/nicolaides.hh"
#include "ddm/impl/prec/galerkin.hh"
#include "ddm/impl/prec/schwarz.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"
#include "dune/ddm/logger.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/solvercategory.hh>
#include <memory>
#include <string>
#include <utility>

namespace ddm {
/** Two-level Schwarz preconditioner with a Nicolaides coarse space. With M^{-1} the fine level (see SchwarzPrec) and
 *  Q = R_0^T A_0^{-1} R_0 the coarse correction (see CoarseCorrection), it applies
 *  - additive: z = M^{-1} r + Q r
 *  - deflated: y = M^{-1} r, z = y + Q (r - A y)
 *
 *  Config:
 *  - coarse_mode: additive | deflated (default additive)
 *  - overlap:     overlap of the fine level, takes precedence over fine.overlap
 *  - pou:         partition of unity of both levels, takes precedence over fine.pou
 *  - fine:        config of the fine level (see SchwarzPrec)
 *  - coarse:      config of the coarse correction (see CoarseCorrection)
 */
template <class T>
class TwoLevelSchwarzPrec final : public Prec<T> {
public:
  // Collective
  TwoLevelSchwarzPrec(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
      , deflated_(parse_coarse_mode(config.get("coarse_mode", "additive")))
      , schwarz_(fine_config(config), this->mat())
      , coarse_space_(nicolaides_coarse_space<T>(schwarz_.get_overlap(), *schwarz_.get_pou(), schwarz_.get_overlapping_mat()->local()))
      , coarse_correction_(config.sub("coarse"), this->mat(), schwarz_.get_overlapping_mat()->local(), coarse_space_)
      , tmp_(this->mat()->create_domain_vector())
      , res_(this->mat()->create_range_vector())
  {
  }

  // On a single rank, the matrix is sequential, and so is the preconditioner
  Dune::SolverCategory::Category category() const override { return this->mat()->category(); }

private:
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
    // Both apply() overwrite their output completely
    schwarz_.apply(z, r);
    if (deflated_) {
      // res = r - A z, consistent
      res_ = r;
      this->mat()->applyscaleadd(T{-1}, z, res_);
      coarse_correction_.apply(tmp_, res_);
    }
    else coarse_correction_.apply(tmp_, r);
    z += tmp_;
  }

  void do_update() override { TODO("do_update"); }

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
        coarse_correction_.info();
        logger::decrease_indent();
      }
      logger::decrease_indent();
    }
  }

  // The coarse level is built from the overlapping index set and the overlapping matrix of the fine level, so the
  // fine level must be constructed first: do not change the order of the members.
  bool deflated_;
  SchwarzPrec<T> schwarz_;
  std::shared_ptr<CoarseSpace<T>> coarse_space_;
  CoarseCorrection<T> coarse_correction_;

  Vec<T> tmp_; ///< the coarse correction in apply()
  Vec<T> res_; ///< the residual of the fine-level correction in apply() (deflated only)
};
} // namespace ddm
