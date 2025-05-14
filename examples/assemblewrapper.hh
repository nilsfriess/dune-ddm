#pragma once

#include "helpers.hh"
#include "spdlog/spdlog.h"

#include <cstdlib>
#include <dune/common/exceptions.hh>
#include <dune/pdelab/gridfunctionspace/lfsindexcache.hh>
#include <dune/pdelab/localoperator/callswitch.hh>
#include <map>

// TODO: Implement the remaining missing functions
template <class LocalOperator>
class AssembleWrapper {
public:
  enum { doSkipEntity = LocalOperator::doSkipEntity };
  enum { doSkipIntersection = LocalOperator::doSkipIntersection };

  enum { doPatternVolume = LocalOperator::doPatternVolume };
  enum { doPatternVolumePostSkeleton = LocalOperator::doPatternVolumePostSkeleton };
  enum { doPatternSkeleton = LocalOperator::doPatternSkeleton };
  enum { doPatternBoundary = LocalOperator::doPatternBoundary };

  enum { doAlphaVolume = LocalOperator::doAlphaVolume };
  enum { doAlphaVolumePostSkeleton = LocalOperator::doAlphaVolumePostSkeleton };
  enum { doAlphaSkeleton = LocalOperator::doAlphaSkeleton };
  enum { doAlphaBoundary = LocalOperator::doAlphaBoundary };

  enum { doLambdaVolume = LocalOperator::doLambdaVolume };
  enum { doLambdaVolumePostSkeleton = LocalOperator::doLambdaVolumePostSkeleton };
  enum { doLambdaSkeleton = LocalOperator::doLambdaSkeleton };
  enum { doLambdaBoundary = LocalOperator::doLambdaBoundary };

  enum { doSkeletonTwoSided = LocalOperator::doSkeletonTwoSided };

  enum { isLinear = LocalOperator::isLinear };

  explicit AssembleWrapper(LocalOperator *lop_) : lop{lop_} {}

  template <typename EG>
  bool skip_entity(const EG &eg) const
  {
    return Dune::PDELab::LocalOperatorApply::skipEntity(*lop, eg);
  }

  template <typename IG>
  bool skip_intersection(const IG &ig) const
  {
    return Dune::PDELab::LocalOperatorApply::skipIntersection(*lop, ig);
  }

  template <typename LFSU, typename LFSV, typename LocalPattern>
  void pattern_volume(const LFSU &lfsu, const LFSV &lfsv, LocalPattern &pattern) const
  {
    spdlog::trace("Called pattern_volume");
    Dune::PDELab::LocalOperatorApply::patternVolume(*lop, lfsu, lfsv, pattern);
  }

  template <typename LFSU, typename LFSV, typename LocalPattern>
  void pattern_volume_post_skeleton(const LFSU &lfsu, const LFSV &lfsv, LocalPattern &pattern) const
  {
    spdlog::trace("Called pattern_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::patternVolumePostSkeleton(*lop, lfsu, lfsv, pattern);
  }

  template <typename LFSU, typename LFSV, typename LocalPattern>
  void pattern_skeleton(const LFSU &lfsu_s, const LFSV &lfsv_s, const LFSU &lfsu_n, const LFSV &lfsv_n, LocalPattern &pattern_sn, LocalPattern &pattern_ns) const
  {
    spdlog::trace("Called pattern_skeleton");
    Dune::PDELab::LocalOperatorApply::patternSkeleton(*lop, lfsu_s, lfsv_s, lfsu_n, lfsv_n, pattern_sn, pattern_ns);
  }

  template <typename LFSU, typename LFSV, typename LocalPattern>
  void pattern_boundary(const LFSU &lfsu_s, const LFSV &lfsv_s, LocalPattern &pattern_ss) const
  {
    spdlog::trace("Called pattern_boundary");
    Dune::PDELab::LocalOperatorApply::patternBoundary(*lop, lfsu_s, lfsv_s, pattern_ss);
  }

  template <typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_volume(const EG &eg, const LFSU &lfsu, const X &x, const LFSV &lfsv, R &r) const
  {
    spdlog::trace("Called alpha_volume");
    Dune::PDELab::LocalOperatorApply::alphaVolume(*lop, eg, lfsu, x, lfsv, r);
  }

  template <typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_volume_post_skeleton(const EG &eg, const LFSU &lfsu, const X &x, const LFSV &lfsv, R &r) const
  {
    spdlog::trace("Called alpha_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::alphaVolumePostSkeleton(*lop, eg, lfsu, x, lfsv, r);
  }

  template <typename IG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_skeleton(const IG &ig, const LFSU &lfsu_s, const X &x_s, const LFSV &lfsv_s, const LFSU &lfsu_n, const X &x_n, const LFSV &lfsv_n, R &r_s, R &r_n) const
  {
    spdlog::trace("Called alpha_skeleton");
    Dune::PDELab::LocalOperatorApply::alphaSkeleton(*lop, ig, lfsu_s, x_s, lfsv_s, lfsu_n, x_n, lfsv_n, r_s, r_n);
  }

  template <typename IG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_boundary(const IG &ig, const LFSU &lfsu_s, const X &x_s, const LFSV &lfsv_s, R &r_s) const
  {
    spdlog::trace("Called alpha_boundary");
    Dune::PDELab::LocalOperatorApply::alphaBoundary(*lop, ig, lfsu_s, x_s, lfsv_s, r_s);
  }

  template <typename EG, typename LFSV, typename R>
  void lambda_volume(const EG &eg, const LFSV &lfsv, R &r) const
  {
    spdlog::trace("Called lambda_volume");
    Dune::PDELab::LocalOperatorApply::lambdaVolume(*lop, eg, lfsv, r);
  }

  template <typename EG, typename LFSV, typename R>
  void lambda_volume_post_skeleton(const EG &eg, const LFSV &lfsv, R &r) const
  {
    spdlog::trace("Called lambda_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::lambdaVolumePostSkeleton(*lop, eg, lfsv, r);
  }

  template <typename IG, typename LFSV, typename R>
  void lambda_skeleton(const IG &ig, const LFSV &lfsv_s, const LFSV &lfsv_n, R &r_s, R &r_n) const
  {
    spdlog::trace("Called lambda_skeleton");
    Dune::PDELab::LocalOperatorApply::lambdaSkeleton(*lop, ig, lfsv_s, lfsv_n, r_s, r_n);
  }

  template <typename IG, typename LFSV, typename R>
  void lambda_boundary(const IG &ig, const LFSV &lfsv, R &r) const
  {
    spdlog::trace("Called lambda_boundary");
    Dune::PDELab::LocalOperatorApply::lambdaBoundary(*lop, ig, lfsv, r);
  }

  template <typename EG, typename LFSU, typename X, typename Z, typename LFSV, typename Y>
  void jacobian_apply_volume(const EG &eg, const LFSU &lfsu, const X &x, const Z &z, const LFSV &lfsv, Y &y) const
  {
    spdlog::trace("Called jacobian_apply_volume");
    Dune::PDELab::LocalOperatorApply::jacobianApplyVolume(*lop, eg, lfsu, x, z, lfsv, y);
  }

  template <typename EG, typename LFSU, typename X, typename Z, typename LFSV, typename Y>
  void jacobian_apply_volume_post_skeleton(const EG &eg, const LFSU &lfsu, const X &x, const Z &z, const LFSV &lfsv, Y &y) const
  {
    spdlog::trace("Called jacobian_apply_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::jacobianApplyVolumePostSkeleton(*lop, eg, lfsu, x, z, lfsv, y);
  }

  template <typename IG, typename LFSU, typename X, typename Z, typename LFSV, typename Y>
  void jacobian_apply_skeleton(const IG &ig, const LFSU &lfsu_s, const X &x_s, const Z &z_s, const LFSV &lfsv_s, const LFSU &lfsu_n, const X &x_n, const Z &z_n, const LFSV &lfsv_n, Y &y_s,
                               Y &y_n) const
  {
    spdlog::trace("Called jacobian_apply_skeleton");
    Dune::PDELab::LocalOperatorApply::jacobianApplySkeleton(*lop, ig, lfsu_s, x_s, z_s, lfsv_s, lfsu_n, x_n, z_n, lfsv_n, y_s, y_n);
  }

  template <typename IG, typename LFSU, typename X, typename Z, typename LFSV, typename Y>
  void jacobian_apply_boundary(const IG &ig, const LFSU &lfsu_s, const X &x_s, const Z &z_s, const LFSV &lfsv_s, Y &y_s) const
  {
    spdlog::trace("Called jacobian_apply_boundary");
    Dune::PDELab::LocalOperatorApply::jacobianApplyBoundary(*lop, ig, lfsu_s, x_s, z_s, lfsv_s, y_s);
  }

  template <typename EG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_volume(const EG &eg, const LFSU &lfsu, const X &x, const LFSV &lfsv, M &mat) const
  {
    spdlog::trace("Called jacobian_volume");

    auto M_before = mat.container();
    Dune::PDELab::LocalOperatorApply::jacobianVolume(*lop, eg, lfsu, x, lfsv, mat);
    auto M_after = mat.container();

    Dune::PDELab::LFSIndexCache cache(lfsu);
    cache.update();

    for (std::size_t i = 0; i < lfsu.size(); ++i) {
      auto gi = cache.containerIndex(i)[0];
      for (std::size_t j = 0; j < lfsu.size(); ++j) { // TODO: We assume lfsu == lfsv
        auto gj = cache.containerIndex(j)[0];

        // TODO: Here and below we assume that an unordered_map zero initialises a double when inserting a new entry. Is this true?
        curr_neumann_corrections[gi][gj] += M_after(lfsu, i, lfsv, j) - M_before(lfsu, i, lfsv, j);
      }
    }

    // Corrections for other ranks
    for (const auto &[rank, mask] : *on_boundary_mask_for_rank) {
      bool hasDofAtBoundary = false;
      bool hasDofOutsideBoundary = false;
      for (std::size_t i = 0; i < cache.size(); ++i) {
        auto dofidx = cache.containerIndex(i)[0];

        if ((*on_boundary_mask_for_rank).at(rank)[dofidx]) {
          hasDofAtBoundary = true;
        }

        if ((*outside_boundary_mask_for_rank).at(rank)[dofidx]) {
          hasDofOutsideBoundary = true;
        }
      }

      if (hasDofAtBoundary and hasDofOutsideBoundary) {
        for (const auto &[row, cols] : curr_neumann_corrections) {
          for (const auto &[col, val] : cols) {
            if ((*on_boundary_mask_for_rank).at(rank)[row] and (*on_boundary_mask_for_rank).at(rank)[col]) {
              neumann_correction_matrices[rank][row][col] += val;
            }
          }
        }
      }
    }

    // Corrections for ourselves
    bool hasDofAtBoundary = false;
    bool hasDofOutsideBoundary = false;
    for (std::size_t i = 0; i < cache.size(); ++i) {
      auto dofidx = cache.containerIndex(i)[0];

      if ((*on_boundary_mask)[dofidx]) {
        hasDofAtBoundary = true;
      }

      if ((*outside_boundary_mask)[dofidx]) {
        hasDofOutsideBoundary = true;
      }
    }
    if (hasDofAtBoundary and hasDofOutsideBoundary) {
      for (const auto &[row, cols] : curr_neumann_corrections) {
        for (const auto &[col, val] : cols) {
          if ((*on_boundary_mask)[row] and (*on_boundary_mask)[col]) {
            neumann_correction_matrices[-1][row][col] += val; // -1 means our own corrections
          }
        }
      }
    }

    curr_neumann_corrections.clear();
  }

  template <typename EG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_volume_post_skeleton(const EG &eg, const LFSU &lfsu, const X &x, const LFSV &lfsv, M &mat) const
  {
    spdlog::trace("Called jacobian_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::jacobianVolumePostSkeleton(*lop, eg, lfsu, x, lfsv, mat);
  }

  template <typename IG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_skeleton(const IG &ig, const LFSU &lfsu_s, const X &x_s, const LFSV &lfsv_s, const LFSU &lfsu_n, const X &x_n, const LFSV &lfsv_n, M &mat_ss, M &mat_sn, M &mat_ns, M &mat_nn) const
  {
    spdlog::trace("Called jacobian_skeleton");
    Dune::PDELab::LocalOperatorApply::jacobianSkeleton(*lop, ig, lfsu_s, x_s, lfsv_s, lfsu_n, x_n, lfsv_n, mat_ss, mat_sn, mat_ns, mat_nn);
  }

  template <typename IG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_boundary(const IG &ig, const LFSU &lfsu_s, const X &x_s, const LFSV &lfsv_s, M &mat_ss) const
  {
    spdlog::trace("Called jacobian_boundary");
    Dune::PDELab::LocalOperatorApply::jacobianBoundary(*lop, ig, lfsu_s, x_s, lfsv_s, mat_ss);
  }

  template <class Mat>
  void setMasks(const Mat &A, const std::map<int, std::vector<bool>> *on_boundary_mask_for_rank, const std::map<int, std::vector<bool>> *outside_boundary_mask_for_rank,
                const std::vector<bool> *on_boundary_mask, const std::vector<bool> *outside_boundary_mask)
  {
    this->on_boundary_mask_for_rank = on_boundary_mask_for_rank;
    this->outside_boundary_mask_for_rank = outside_boundary_mask_for_rank;
    this->on_boundary_mask = on_boundary_mask;
    this->outside_boundary_mask = outside_boundary_mask;

    const auto avg = A.nonzeroes() / A.N();
    for (const auto &[rank, mask] : *on_boundary_mask_for_rank) {
      auto &An = neumann_correction_matrices[rank];
      An.setBuildMode(Dune::BCRSMatrix<double>::implicit);
      An.setImplicitBuildModeParameters(avg, 0.4);
      An.setSize(A.N(), A.M());

      for (auto ri = A.begin(); ri != A.end(); ++ri) {
        if (mask[ri.index()]) {
          for (auto ci = A[ri.index()].begin(); ci != A[ri.index()].end(); ++ci) {
            if (mask[ci.index()]) {
              An.entry(ri.index(), ci.index()) = 0.0;
            }
          }
        }
      }
      An.compress();
    }

    // Set up an additional correction matrix for our own local correction
    auto &An = neumann_correction_matrices[-1];
    An.setBuildMode(Dune::BCRSMatrix<double>::implicit);
    An.setImplicitBuildModeParameters(avg, 0.4);
    An.setSize(A.N(), A.M());

    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      if ((*on_boundary_mask)[ri.index()]) {
        for (auto ci = A[ri.index()].begin(); ci != A[ri.index()].end(); ++ci) {
          if ((*on_boundary_mask)[ci.index()]) {
            An.entry(ri.index(), ci.index()) = 0.0;
          }
        }
      }
    }
    An.compress();
  }

  template <class GLIS>
  std::unordered_map<int, std::vector<TripleWithRank>> get_correction_triples(const GLIS &glis) const
  {
    std::unordered_map<int, std::vector<TripleWithRank>> triples_for_rank;

    for (const auto &[rank, An] : neumann_correction_matrices) {
      triples_for_rank[rank].reserve(An.nonzeroes());
      if (rank >= 0) {
        for (auto ri = An.begin(); ri != An.end(); ++ri) {
          auto grow = glis.pair(ri.index())->global();
          for (auto ci = An[ri.index()].begin(); ci != An[ri.index()].end(); ++ci) {
            auto gcol = glis.pair(ci.index())->global();
            if (std::abs(*ci) > 0) {
              triples_for_rank[rank].emplace_back(TripleWithRank{
                  .rank = rank,
                  .row = grow,
                  .col = gcol,
                  .val = *ci,
              });
            }
          }
        }
      }
      else {
        for (auto ri = An.begin(); ri != An.end(); ++ri) {
          for (auto ci = An[ri.index()].begin(); ci != An[ri.index()].end(); ++ci) {
            if (std::abs(*ci) > 0) {
              triples_for_rank[rank].emplace_back(TripleWithRank{
                  .rank = rank,
                  .row = ri.index(),
                  .col = ci.index(),
                  .val = *ci,
              });
            }
          }
        }
      }
    }

    return triples_for_rank;
  }

private:
  const std::map<int, std::vector<bool>> *on_boundary_mask_for_rank{nullptr};
  const std::map<int, std::vector<bool>> *outside_boundary_mask_for_rank{nullptr};
  const std::vector<bool> *on_boundary_mask{nullptr}; // Masks for the "inner" corrections
  const std::vector<bool> *outside_boundary_mask{nullptr};

  mutable std::unordered_map<std::size_t, std::unordered_map<std::size_t, double>> curr_neumann_corrections; // maps from row -> col -> value

  mutable std::unordered_map<int, Dune::BCRSMatrix<double>> neumann_correction_matrices;

  LocalOperator *lop;
};
