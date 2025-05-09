#pragma once

#include "../helpers.hh"

template <class Mat, class Vec, class ParallelIndexSet>
Mat apply_neumann_corrections(const Mat &A, const std::vector<TripleWithRank> &remote_ncorr_triples, const Vec &dirichlet_mask_ovlp, const ParallelIndexSet &ovlp_paridxs)
{
  Mat Aneu = A;
  for (const auto &triple : remote_ncorr_triples) {
    // The triples use global indices, so we first have to convert them to local indices
    // on the overlapping subdomain. Also, we might have received some indices that are
    // outside of our overlapping subdomain, so we first have to check that.
    // TODO: Should we simply assume that we only received valid indices?
    if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
      auto lrow = ovlp_paridxs[triple.row].local();
      auto lcol = ovlp_paridxs[triple.col].local();

      Aneu[lrow][lcol] -= triple.val;
    }
    else {
      spdlog::debug("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
    }
  }

  // Make sure global Dirichlet conditions are correctly set. We have to eliminate symmetrically, because the eigensolver expects a symmetric problem.
  for (std::size_t i = 0; i < A.N(); ++i) {
    if (dirichlet_mask_ovlp[i] > 0) {
      for (auto ci = Aneu[i].begin(); ci != Aneu[i].end(); ++ci) {
        *ci = (ci.index() == i) ? 1.0 : 0.0;
      }
    }
    else {
      for (auto ci = Aneu[i].begin(); ci != Aneu[i].end(); ++ci) {
        if (dirichlet_mask_ovlp[ci.index()] > 0) {
          *ci = 0.0;
        }
      }
    }
  }
  return Aneu;
}
