#pragma once

#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/overlap.hh"
#include "ddm/vec/host_view.hh"
#include "ddm/vec/vec.hh"

#include <algorithm>
#include <cstddef>
#include <dune/common/parametertree.hh>

namespace ddm {
/** Returns the distance-based partition of unity D on the overlapping index set of ovlp, i.e. the diagonal of D_i for
 *  our subdomain i, with sum_j R_j^T D_j R_j = I. D decreases linearly from the original indices towards the outside
 *  and is positive exactly on the first k layers (layer < k), zero on all others. Collective.
 *
 *  A_ovlp is only used to create the vector, so that it has the backend of the overlapping matrix.
 *
 *  Config (read directly from config, so the caller passes the subtree of the partition of unity):
 *  - layers: k, 1 <= k <= ovlp.layers (default ovlp.layers). With k < ovlp.layers, the partition of unity only lives
 *            on the inner layers, e.g. to use the outer ones as oversampling layers.
 */
template <class T>
Vec<T> partition_of_unity(const Dune::ParameterTree& config, const Overlap& ovlp, const LocalMat<T>& A_ovlp)
{
  auto pou = A_ovlp.create_domain_vector();
  DDM_CHECK(ovlp.layer.size() == static_cast<std::size_t>(pou.size()), "partition_of_unity: size mismatch, the overlapping index set has {} indices, the matrix produced a vector of size {}",
            ovlp.layer.size(), pou.size());

  // Get the "width" of the partition of unity. It can be smaller than the number of layers in
  // the overlapping subdomain which can be used to simulate the oversampling layers that are
  // needed in the MsGFEM coarse spaces.
  int pou_layers = config.get("layers", ovlp.layers);
  DDM_CHECK(pou_layers >= 1 and pou_layers <= ovlp.layers, "partition_of_unity: layers must be at least 1 and at most the number of layers of the overlap ({}), got {}", ovlp.layers, pou_layers);

  // Write max(k - layer[i], 0) into the POU, with k = pou_layers: layer[i] is 0 for the original indices and grows by
  // one per layer, so the weight decreases linearly towards the outside and is zero from layer k on (in particular on
  // the outermost layer, since k <= L). Normalising this then produces exactly the "distance" partition of unity.
  {
    auto pou_host = pou.host_view(write);
    for (std::size_t i = 0; i < ovlp.layer.size(); ++i) pou_host[i] = std::max(pou_layers - ovlp.layer[i], 0);
  }
  // The sum of the weights over all subdomains holding an index. It is positive: the owner of the index holds it in its
  // original index set (layer 0), so its weight is k >= 1.
  auto pou_sum = pou;
  ovlp.comm->reduce(pou_sum);
  {
    auto pou_host = pou.host_view(read_write);
    auto pou_sum_host = pou_sum.host_view(read);
    for (Index i = 0; i < pou.size(); ++i) pou_host[i] /= pou_sum_host[i];
  }

  return pou;
}
} // namespace ddm
