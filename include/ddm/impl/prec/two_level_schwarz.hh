#pragma once

#include "ddm/check.hh"
#include "ddm/coarse_space.hh"
#include "ddm/impl/prec/coarse_spaces/nicolaides.hh"
#include "ddm/impl/prec/galerkin.hh"
#include "ddm/impl/prec/schwarz.hh"
#include "ddm/mat/mat.hh"
#include "ddm/pou.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"
#include "dune/ddm/logger.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/solvercategory.hh>
#include <memory>
#include <optional>
#include <utility>

namespace ddm {
template <class T>
class TwoLevelSchwarzPrec final : public Prec<T> {
public:
  // Collective
  TwoLevelSchwarzPrec(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
      , schwarz_(fine_config(config), this->mat())
      , pou_(partition_of_unity(config.sub("pou"), *schwarz_.get_overlap(), schwarz_.get_overlapping_mat()->local()))
      , coarse_space_(nicolaides_coarse_space<T>(schwarz_.get_overlap(), pou_, schwarz_.get_overlapping_mat()->local()))
      , coarse_correction_(config.sub("coarse"), this->mat(), schwarz_.get_overlapping_mat()->local(), coarse_space_)
      , tmp_(this->mat()->create_domain_vector())
  {
  }

  // On a single rank, the matrix is sequential, and so is the preconditioner
  Dune::SolverCategory::Category category() const override { return this->mat()->category(); }

private:
  // The config of the fine level: the subtree "fine", but the overlap can also be set on the two-level method itself
  // (key "overlap"), which then takes precedence. Unlike for the one-level method, overlap 0 is not allowed: the
  // coarse space needs a partition of unity that vanishes on the outermost layer.
  static Dune::ParameterTree fine_config(const Dune::ParameterTree& config)
  {
    auto fine = config.sub("fine");
    if (config.hasKey("overlap")) fine["overlap"] = config["overlap"];
    const int overlap = fine.get("overlap", 1);
    DDM_CHECK(overlap >= 1, "two-level schwarz: the overlap must be at least 1, got {}", overlap);
    return fine;
  }

  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    // Both apply() overwrite their output completely
    schwarz_.apply(z, r);
    coarse_correction_.apply(tmp_, r);
    z += tmp_;
  }

  void do_update() override { TODO("do_update"); }

  void do_info() const override
  {
    logger::info("Two-level Schwarz preconditioner");
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
  SchwarzPrec<T> schwarz_;
  Vec<T> pou_;
  std::shared_ptr<CoarseSpace<T>> coarse_space_;
  CoarseCorrection<T> coarse_correction_;

  Vec<T> tmp_; ///< the coarse correction in apply()
};
} // namespace ddm
