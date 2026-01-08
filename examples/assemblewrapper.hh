#pragma once

#include "dune/ddm/helpers.hh"
#include "dune/ddm/logger.hh"

#include <cstdlib>
#include <dune/common/exceptions.hh>
#include <dune/pdelab/gridfunctionspace/lfsindexcache.hh>
#include <dune/pdelab/localoperator/callswitch.hh>
#include <map>
#include <unordered_map>
#include <vector>

// TODO: Implement the remaining missing functions
/**
 * @brief Wrapper for a PDELab LocalOperator to intercept assembly and compute Neumann corrections.
 *
 * This class wraps a LocalOperator and forwards all calls to it. However, during `jacobian_volume`,
 * it additionally computes the contribution of the current element to the global matrix.
 * If this contribution affects degrees of freedom on the boundary of a neighboring rank's subdomain
 * (identified via masks), it is stored as a "Neumann correction".
 *
 * These corrections are later exchanged between ranks to ensure that each rank's local matrix
 * correctly reflects the Neumann boundary conditions on the artificial interface in an
 * overlapping Schwarz method.
 */
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

  explicit AssembleWrapper(LocalOperator* lop_)
      : lop{lop_}
  {
  }

  template <typename EG>
  bool skip_entity(const EG& eg) const
  {
    return Dune::PDELab::LocalOperatorApply::skipEntity(*lop, eg);
  }

  template <typename IG>
  bool skip_intersection(const IG& ig) const
  {
    return Dune::PDELab::LocalOperatorApply::skipIntersection(*lop, ig);
  }

  template <typename LFSU, typename LFSV, typename LocalPattern>
  void pattern_volume(const LFSU& lfsu, const LFSV& lfsv, LocalPattern& pattern) const
  {
    logger::trace("Called pattern_volume");
    Dune::PDELab::LocalOperatorApply::patternVolume(*lop, lfsu, lfsv, pattern);
  }

  template <typename LFSU, typename LFSV, typename LocalPattern>
  void pattern_volume_post_skeleton(const LFSU& lfsu, const LFSV& lfsv, LocalPattern& pattern) const
  {
    logger::trace("Called pattern_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::patternVolumePostSkeleton(*lop, lfsu, lfsv, pattern);
  }

  template <typename LFSU, typename LFSV, typename LocalPattern>
  void pattern_skeleton(const LFSU& lfsu_s, const LFSV& lfsv_s, const LFSU& lfsu_n, const LFSV& lfsv_n, LocalPattern& pattern_sn, LocalPattern& pattern_ns) const
  {
    logger::trace("Called pattern_skeleton");
    Dune::PDELab::LocalOperatorApply::patternSkeleton(*lop, lfsu_s, lfsv_s, lfsu_n, lfsv_n, pattern_sn, pattern_ns);
  }

  template <typename LFSU, typename LFSV, typename LocalPattern>
  void pattern_boundary(const LFSU& lfsu_s, const LFSV& lfsv_s, LocalPattern& pattern_ss) const
  {
    logger::trace("Called pattern_boundary");
    Dune::PDELab::LocalOperatorApply::patternBoundary(*lop, lfsu_s, lfsv_s, pattern_ss);
  }

  template <typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_volume(const EG& eg, const LFSU& lfsu, const X& x, const LFSV& lfsv, R& r) const
  {
    logger::trace("Called alpha_volume");
    Dune::PDELab::LocalOperatorApply::alphaVolume(*lop, eg, lfsu, x, lfsv, r);
  }

  template <typename EG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_volume_post_skeleton(const EG& eg, const LFSU& lfsu, const X& x, const LFSV& lfsv, R& r) const
  {
    logger::trace("Called alpha_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::alphaVolumePostSkeleton(*lop, eg, lfsu, x, lfsv, r);
  }

  template <typename IG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_skeleton(const IG& ig, const LFSU& lfsu_s, const X& x_s, const LFSV& lfsv_s, const LFSU& lfsu_n, const X& x_n, const LFSV& lfsv_n, R& r_s, R& r_n) const
  {
    logger::trace("Called alpha_skeleton");
    Dune::PDELab::LocalOperatorApply::alphaSkeleton(*lop, ig, lfsu_s, x_s, lfsv_s, lfsu_n, x_n, lfsv_n, r_s, r_n);
  }

  template <typename IG, typename LFSU, typename X, typename LFSV, typename R>
  void alpha_boundary(const IG& ig, const LFSU& lfsu_s, const X& x_s, const LFSV& lfsv_s, R& r_s) const
  {
    logger::trace("Called alpha_boundary");
    Dune::PDELab::LocalOperatorApply::alphaBoundary(*lop, ig, lfsu_s, x_s, lfsv_s, r_s);
  }

  template <typename EG, typename LFSV, typename R>
  void lambda_volume(const EG& eg, const LFSV& lfsv, R& r) const
  {
    logger::trace("Called lambda_volume");
    Dune::PDELab::LocalOperatorApply::lambdaVolume(*lop, eg, lfsv, r);
  }

  template <typename EG, typename LFSV, typename R>
  void lambda_volume_post_skeleton(const EG& eg, const LFSV& lfsv, R& r) const
  {
    logger::trace("Called lambda_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::lambdaVolumePostSkeleton(*lop, eg, lfsv, r);
  }

  template <typename IG, typename LFSV, typename R>
  void lambda_skeleton(const IG& ig, const LFSV& lfsv_s, const LFSV& lfsv_n, R& r_s, R& r_n) const
  {
    logger::trace("Called lambda_skeleton");
    Dune::PDELab::LocalOperatorApply::lambdaSkeleton(*lop, ig, lfsv_s, lfsv_n, r_s, r_n);
  }

  template <typename IG, typename LFSV, typename R>
  void lambda_boundary(const IG& ig, const LFSV& lfsv, R& r) const
  {
    logger::trace("Called lambda_boundary");
    Dune::PDELab::LocalOperatorApply::lambdaBoundary(*lop, ig, lfsv, r);
  }

  template <typename EG, typename LFSU, typename X, typename Z, typename LFSV, typename Y>
  void jacobian_apply_volume(const EG& eg, const LFSU& lfsu, const X& x, const Z& z, const LFSV& lfsv, Y& y) const
  {
    logger::trace("Called jacobian_apply_volume");
    Dune::PDELab::LocalOperatorApply::jacobianApplyVolume(*lop, eg, lfsu, x, z, lfsv, y);
  }

  template <typename EG, typename LFSU, typename X, typename Z, typename LFSV, typename Y>
  void jacobian_apply_volume_post_skeleton(const EG& eg, const LFSU& lfsu, const X& x, const Z& z, const LFSV& lfsv, Y& y) const
  {
    logger::trace("Called jacobian_apply_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::jacobianApplyVolumePostSkeleton(*lop, eg, lfsu, x, z, lfsv, y);
  }

  template <typename IG, typename LFSU, typename X, typename Z, typename LFSV, typename Y>
  void jacobian_apply_skeleton(const IG& ig, const LFSU& lfsu_s, const X& x_s, const Z& z_s, const LFSV& lfsv_s, const LFSU& lfsu_n, const X& x_n, const Z& z_n, const LFSV& lfsv_n, Y& y_s,
                               Y& y_n) const
  {
    logger::trace("Called jacobian_apply_skeleton");
    Dune::PDELab::LocalOperatorApply::jacobianApplySkeleton(*lop, ig, lfsu_s, x_s, z_s, lfsv_s, lfsu_n, x_n, z_n, lfsv_n, y_s, y_n);
  }

  template <typename IG, typename LFSU, typename X, typename Z, typename LFSV, typename Y>
  void jacobian_apply_boundary(const IG& ig, const LFSU& lfsu_s, const X& x_s, const Z& z_s, const LFSV& lfsv_s, Y& y_s) const
  {
    logger::trace("Called jacobian_apply_boundary");
    Dune::PDELab::LocalOperatorApply::jacobianApplyBoundary(*lop, ig, lfsu_s, x_s, z_s, lfsv_s, y_s);
  }

  template <typename EG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_volume(const EG& eg, const LFSU& lfsu, const X& x, const LFSV& lfsv, M& mat) const
  {
    logger::trace("Called jacobian_volume");

    if (on_boundary_mask_for_rank == nullptr) {
      // Standard assembly if no masks are set (e.g. during initial setup)
      Dune::PDELab::LocalOperatorApply::jacobianVolume(*lop, eg, lfsu, x, lfsv, mat);
    }
    else {
      // Intercept assembly to compute corrections
      auto M_before = mat.container();
      Dune::PDELab::LocalOperatorApply::jacobianVolume(*lop, eg, lfsu, x, lfsv, mat);

      Dune::PDELab::LFSIndexCache cache(lfsu);
      cache.update();

      // Pre-calculate global indices to avoid repeated cache lookups
      std::vector<std::size_t> global_indices(lfsu.size());
      for (std::size_t i = 0; i < lfsu.size(); ++i) {
        global_indices[i] = cache.containerIndex(i)[0];
        assert(cache.containerIndex(i).size() == 1);
      }

      // Corrections for other ranks:
      // If an element contributes to a DOF on the boundary of another rank's subdomain,
      // we record that contribution.
      for (const auto& [rank, mask] : *on_boundary_mask_for_rank) {
        bool hasDofAtBoundary = false;
        bool hasDofInsideBoundary = false;
        const auto& inside_mask = (*inside_boundary_mask_for_rank).at(rank);

        bool hasDofOutside = false;
        for (auto gi : global_indices) {
          if (mask[gi]) hasDofAtBoundary = true;
          if (!mask[gi] && !inside_mask[gi]) hasDofOutside = true;
        }

        // We only care if the element touches the boundary but is NOT fully inside the other rank's domain.
        if (hasDofAtBoundary and hasDofOutside) {
          auto& An = neumann_correction_matrices[rank];
          for (std::size_t i = 0; i < lfsu.size(); ++i) {
            auto gi = global_indices[i];
            if (!mask[gi]) continue;

            for (std::size_t j = 0; j < lfsu.size(); ++j) { // TODO: We assume lfsu == lfsv
              auto gj = global_indices[j];
              if (!mask[gj]) continue;

              // Calculate the contribution of this element
              double val = mat.container()(lfsu, i, lfsv, j) - M_before(lfsu, i, lfsv, j);
              An.entry(gi, gj) += val;
            }
          }
        }
      }

      // Corrections for ourselves (local Neumann boundaries)
      bool hasDofAtBoundary = false;
      bool hasDofOutsideBoundary = false;
      for (auto gi : global_indices) {
        if ((*on_boundary_mask)[gi]) hasDofAtBoundary = true;
        if ((*outside_boundary_mask)[gi]) hasDofOutsideBoundary = true;
      }
      if (hasDofAtBoundary and hasDofOutsideBoundary) {
        auto& An = neumann_correction_matrices[-1]; // -1 means our own corrections
        for (std::size_t i = 0; i < lfsu.size(); ++i) {
          auto gi = global_indices[i];
          if (!(*on_boundary_mask)[gi]) continue;

          for (std::size_t j = 0; j < lfsu.size(); ++j) {
            auto gj = global_indices[j];
            if (!(*on_boundary_mask)[gj]) continue;

            double val = mat.container()(lfsu, i, lfsv, j) - M_before(lfsu, i, lfsv, j);
            An.entry(gi, gj) += val;
          }
        }
      }
    }
  }

  template <typename EG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_volume_post_skeleton(const EG& eg, const LFSU& lfsu, const X& x, const LFSV& lfsv, M& mat) const
  {
    logger::trace("Called jacobian_volume_post_skeleton");
    Dune::PDELab::LocalOperatorApply::jacobianVolumePostSkeleton(*lop, eg, lfsu, x, lfsv, mat);
  }

  template <typename IG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_skeleton(const IG& ig, const LFSU& lfsu_s, const X& x_s, const LFSV& lfsv_s, const LFSU& lfsu_n, const X& x_n, const LFSV& lfsv_n, M& mat_ss, M& mat_sn, M& mat_ns, M& mat_nn) const
  {
    logger::trace("Called jacobian_skeleton");

    if (on_boundary_mask_for_rank == nullptr) { Dune::PDELab::LocalOperatorApply::jacobianSkeleton(*lop, ig, lfsu_s, x_s, lfsv_s, lfsu_n, x_n, lfsv_n, mat_ss, mat_sn, mat_ns, mat_nn); }
    else {
      auto M_ss_before = mat_ss.container();
      auto M_nn_before = mat_nn.container();

      Dune::PDELab::LocalOperatorApply::jacobianSkeleton(*lop, ig, lfsu_s, x_s, lfsv_s, lfsu_n, x_n, lfsv_n, mat_ss, mat_sn, mat_ns, mat_nn);

      // For now, let's assume that this is called in a DG discretisation.
      // Then all dofs in an element have the same distance. This means we can do the following:
      // - Check if a dof on the outside element is marked in one of the outside_boundary masks
      // - Check if a dof on the inside element in marked in one of the on_boundary masks
      // If both checks are true, we have to assemble a Neumann correction

      Dune::PDELab::LFSIndexCache cache_inside(lfsu_s);
      std::vector<std::size_t> global_indices_inside(lfsu_s.size());
      cache_inside.update();
      for (std::size_t i = 0; i < lfsu_s.size(); ++i) {
        global_indices_inside[i] = cache_inside.containerIndex(i)[0];
        assert(cache_inside.containerIndex(i).size() == 1);
      }

      Dune::PDELab::LFSIndexCache cache_outside(lfsu_n);
      std::vector<std::size_t> global_indices_outside(lfsu_n.size());
      cache_outside.update();
      for (std::size_t i = 0; i < lfsu_n.size(); ++i) {
        global_indices_outside[i] = cache_outside.containerIndex(i)[0];
        assert(cache_outside.containerIndex(i).size() == 1);
      }

      for (const auto& [rank, mask] : *on_boundary_mask_for_rank) {
        const auto& inside_mask = (*inside_boundary_mask_for_rank).at(rank);

        // Case 1: Inside element is on boundary, outside element is outside boundary.
        // Then we need to save the contributions from the inside element as Neumann corrections.
        {
          bool inside_elem_is_on_boundary = mask[global_indices_inside[0]];
          bool outside_elem_is_outside_boundary = !mask[global_indices_outside[0]] && !inside_mask[global_indices_outside[0]];

          // If the outside element is a ghost element, we don't need to assemble Neumann corrections for it
          bool outside_elem_is_ghost = ig.outside().partitionType() == Dune::GhostEntity;

          // Sanity checks: In a DG method, if one dof of an element is on (outside) the subdomain boundary, all of them should be on (outside)
          if (inside_elem_is_on_boundary) assert(std::all_of(global_indices_inside.begin(), global_indices_inside.end(), [&](auto& gi) { return mask[gi]; }));
          if (outside_elem_is_outside_boundary) assert(std::all_of(global_indices_outside.begin(), global_indices_outside.end(), [&](auto& gi) { return !mask[gi] && !inside_mask[gi]; }));

          if (inside_elem_is_on_boundary and outside_elem_is_outside_boundary and !outside_elem_is_ghost) {
            auto& An = neumann_correction_matrices[rank];
            for (std::size_t i = 0; i < lfsu_s.size(); ++i) {
              auto gi = global_indices_inside[i];

              for (std::size_t j = 0; j < lfsu_s.size(); ++j) { // TODO: We assume lfsu == lfsv
                auto gj = global_indices_inside[j];

                // Calculate the contribution of this element
                double val = mat_ss.container()(lfsu_s, i, lfsv_s, j) - M_ss_before(lfsu_s, i, lfsv_s, j);
                An.entry(gi, gj) += val;
              }
            }
          }
        }

        // Case 2: Outside element is on boundary, inside element is outside boundary.
        // Then we need to save the contributions from the outside element as Neumann correction.
        {
          bool outside_elem_is_on_boundary = mask[global_indices_outside[0]];
          bool inside_elem_is_outside_boundary = !mask[global_indices_inside[0]] && !inside_mask[global_indices_inside[0]];

          // If the inside element is a ghost element, we don't need to assemble Neumann corrections for it
          bool inside_elem_is_ghost = ig.inside().partitionType() == Dune::GhostEntity;

          // Sanity checks: In a DG method, if one dof of an element is on (outside) the subdomain boundary, all of them should be on (outside)
          if (outside_elem_is_on_boundary) assert(std::all_of(global_indices_outside.begin(), global_indices_outside.end(), [&](auto& gi) { return mask[gi]; }));
          if (inside_elem_is_outside_boundary) assert(std::all_of(global_indices_inside.begin(), global_indices_inside.end(), [&](auto& gi) { return !mask[gi] && !inside_mask[gi]; }));

          if (outside_elem_is_on_boundary and inside_elem_is_outside_boundary and !inside_elem_is_ghost) {
            auto& An = neumann_correction_matrices[rank];
            for (std::size_t i = 0; i < lfsu_n.size(); ++i) {
              auto gi = global_indices_outside[i];

              for (std::size_t j = 0; j < lfsu_n.size(); ++j) { // TODO: We assume lfsu == lfsv
                auto gj = global_indices_outside[j];

                // Calculate the contribution of this element
                double val = mat_nn.container()(lfsu_n, i, lfsv_n, j) - M_nn_before(lfsu_n, i, lfsv_n, j);
                An.entry(gi, gj) += val;
              }
            }
          }
        }
      }
    }
  }

  template <typename IG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_boundary(const IG& ig, const LFSU& lfsu_s, const X& x_s, const LFSV& lfsv_s, M& mat_ss) const
  {
    logger::trace("Called jacobian_boundary");

    Dune::PDELab::LocalOperatorApply::jacobianBoundary(*lop, ig, lfsu_s, x_s, lfsv_s, mat_ss);
  }

  /**
   * @brief Sets the masks used to identify DOFs on the boundaries of overlapping subdomains.
   *
   * @param A The matrix structure (used to initialize correction matrices).
   * @param on_boundary_mask_for_rank Map from rank to a boolean mask indicating DOFs on the boundary of that rank's subdomain.
   * @param inside_boundary_mask_for_rank Map from rank to a boolean mask indicating DOFs strictly inside that rank's subdomain.
   * @param on_boundary_mask Mask for DOFs on the boundary of the local subdomain (for self-correction).
   * @param outside_boundary_mask Mask for DOFs outside the local subdomain (for self-correction).
   *
   * This method also initializes the `neumann_correction_matrices` for each relevant rank.
   */
  template <class Mat>
  void set_masks(const Mat& A, const std::map<int, std::vector<bool>>* on_boundary_mask_for_rank, const std::map<int, std::vector<bool>>* inside_boundary_mask_for_rank,
                 const std::vector<bool>* on_boundary_mask, const std::vector<bool>* outside_boundary_mask)
  {
    this->on_boundary_mask_for_rank = on_boundary_mask_for_rank;
    this->inside_boundary_mask_for_rank = inside_boundary_mask_for_rank;
    this->on_boundary_mask = on_boundary_mask;
    this->outside_boundary_mask = outside_boundary_mask;

    const auto avg = A.nonzeroes() / A.N();
    for (const auto& [rank, mask] : *on_boundary_mask_for_rank) {
      auto& An = neumann_correction_matrices[rank];
      An.setBuildMode(Dune::BCRSMatrix<double>::implicit);
      An.setImplicitBuildModeParameters(avg, 0.4);
      An.setSize(A.N(), A.M());

      for (auto ri = A.begin(); ri != A.end(); ++ri) {
        if (mask[ri.index()]) {
          for (auto ci = A[ri.index()].begin(); ci != A[ri.index()].end(); ++ci)
            if (mask[ci.index()]) An.entry(ri.index(), ci.index()) = 0.0;
        }
      }
      An.compress();
    }

    // Set up an additional correction matrix for our own local correction
    auto& An = neumann_correction_matrices[-1];
    An.setBuildMode(Dune::BCRSMatrix<double>::implicit);
    An.setImplicitBuildModeParameters(avg, 0.4);
    An.setSize(A.N(), A.M());

    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      if ((*on_boundary_mask)[ri.index()]) {
        for (auto ci = A[ri.index()].begin(); ci != A[ri.index()].end(); ++ci)
          if ((*on_boundary_mask)[ci.index()]) An.entry(ri.index(), ci.index()) = 0.0;
      }
    }
    An.compress();
  }

  void reset_masks()
  {
    on_boundary_mask_for_rank = nullptr;
    inside_boundary_mask_for_rank = nullptr;
    on_boundary_mask = nullptr;
    outside_boundary_mask = nullptr;
  }

  /**
   * @brief Extracts the computed Neumann corrections as a list of triples.
   *
   * @param glis Global lookup index set (to map local indices to global indices).
   * @return A map from rank to a vector of {row, col, value} triples representing the corrections to be sent to that rank.
   *         Rank -1 contains corrections for the local process.
   */
  template <class GLIS>
  std::unordered_map<int, std::vector<TripleWithRank>> get_correction_triples(const GLIS& glis) const
  {
    std::unordered_map<int, std::vector<TripleWithRank>> triples_for_rank;

    for (const auto& [rank, An] : neumann_correction_matrices) {
      triples_for_rank[rank].reserve(An.nonzeroes());
      if (rank >= 0) {
        for (auto ri = An.begin(); ri != An.end(); ++ri) {
          auto grow = glis.pair(ri.index())->global();
          for (auto ci = An[ri.index()].begin(); ci != An[ri.index()].end(); ++ci) {
            auto gcol = glis.pair(ci.index())->global();
            triples_for_rank[rank].emplace_back(TripleWithRank{
                .rank = rank,
                .row = grow,
                .col = gcol,
                .val = *ci,
            });
          }
        }
      }
      else {
        for (auto ri = An.begin(); ri != An.end(); ++ri) {
          for (auto ci = An[ri.index()].begin(); ci != An[ri.index()].end(); ++ci) {
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

    return triples_for_rank;
  }

private:
  const std::map<int, std::vector<bool>>* on_boundary_mask_for_rank{nullptr};
  const std::map<int, std::vector<bool>>* inside_boundary_mask_for_rank{nullptr};
  const std::vector<bool>* on_boundary_mask{nullptr}; // Masks for the "inner" corrections
  const std::vector<bool>* outside_boundary_mask{nullptr};

  mutable std::unordered_map<int, Dune::BCRSMatrix<double>> neumann_correction_matrices;

  LocalOperator* lop;
};
