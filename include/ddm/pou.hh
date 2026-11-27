#pragma once

#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/overlap.hh"
#include "ddm/region.hh"
#include "ddm/vec/host_view.hh"
#include "ddm/vec/vec.hh"

#include <algorithm>
#include <cstddef>
#include <dune/common/parametertree.hh>
#include <span>
#include <vector>

namespace ddm {
/** Returns the distance-based partition of unity on the local subdomains of the overlapping index set of ovlp (see
 *  partition_matrix_with_overlap()), i.e. the diagonal of D_j for each of our local subdomains j, with
 *  sum_j R_j^T D_j R_j = I over the local subdomains of all ranks. D_j decreases linearly from the core of subdomain j
 *  towards the outside and is positive exactly on its first k layers (layer < k), zero on all others. Collective.
 *
 *  A_ovlp is the matrix on the overlapping index set, it is only used to create the vectors, so that they have its
 *  backend.
 *
 *  Config (read directly from config, so the caller passes the subtree of the partition of unity):
 *  - layers: k, 1 <= k <= ovlp.layers (default ovlp.layers). With k < ovlp.layers, the partition of unity only lives
 *            on the inner layers, e.g. to use the outer ones as oversampling layers.
 */
template <class T>
std::vector<Vec<T>> partition_of_unity(const Dune::ParameterTree& config, const Overlap& ovlp, std::span<const LocalSubdomain> subdomains, const LocalMat<T>& A_ovlp)
{
  auto pou_sum = A_ovlp.create_domain_vector();
  DDM_CHECK(ovlp.layer.size() == static_cast<std::size_t>(pou_sum.size()), "partition_of_unity: size mismatch, the overlapping index set has {} indices, the matrix produced a vector of size {}",
            ovlp.layer.size(), pou_sum.size());

  // Get the "width" of the partition of unity. It can be smaller than the number of layers in
  // the overlapping subdomain which can be used to simulate the oversampling layers that are
  // needed in the MsGFEM coarse spaces.
  int pou_layers = config.get("layers", ovlp.layers);
  DDM_CHECK(pou_layers >= 1 and pou_layers <= ovlp.layers, "partition_of_unity: layers must be at least 1 and at most the number of layers of the overlap ({}), got {}", ovlp.layers, pou_layers);

  // Write max(k - layer[i], 0) into the POU of every local subdomain, with k = pou_layers: layer[i] is 0 in the core
  // and grows by one per layer, so the weight decreases linearly towards the outside and is zero from layer k on (in
  // particular on the outermost layer, since k <= L). Normalising this then produces exactly the "distance" partition
  // of unity.
  std::vector<Vec<T>> pous;
  pous.reserve(subdomains.size());
  for (const auto& sd : subdomains) {
    auto& pou = pous.emplace_back(pou_sum.multivec().create_multivector(sd.region.size(), 1));
    auto pou_host = pou.host_view(write);
    for (std::size_t k = 0; k < sd.layer.size(); ++k) pou_host[k] = std::max(pou_layers - sd.layer[k], 0);
  }

  // The sum of the weights over all subdomains holding an index, the local ones and those of the other ranks. It is
  // positive: the owner of the index holds it in its original index set, so it is in the core of one of the owner's
  // local subdomains, where the weight is k >= 1.
  pou_sum.zero();
  for (std::size_t j = 0; j < subdomains.size(); ++j) pou_sum.multivec().add_rows(pous[j].multivec(), subdomains[j].region.indices());
  ovlp.comm->reduce(pou_sum);
  {
    auto pou_sum_host = pou_sum.host_view(read);
    for (std::size_t j = 0; j < subdomains.size(); ++j) {
      auto pou_host = pous[j].host_view(read_write);
      const auto indices = subdomains[j].region.indices();
      for (Index k = 0; k < pous[j].size(); ++k) pou_host[k] /= pou_sum_host[indices[k]];
    }
  }

  return pous;
}
} // namespace ddm
