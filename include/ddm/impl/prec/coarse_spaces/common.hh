#pragma once

#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/overlap.hh"

#include <algorithm>
#include <vector>

namespace ddm::detail {

/** The Dirichlet dofs on the overlapping subdomain. This just broadcasts the indices in
 *  A.zeroed_rows() to everyone. Returns the indices sorted. Collective.
 */
template <class T>
std::vector<Index> overlapping_dirichlet_dofs(const Mat<T>& A, const LocalMat<T>& A_ovlp, const Overlap& ovlp)
{
  std::vector<Index> dirichlet;
  auto mask = A_ovlp.create_range_vector();
  {
    auto view = mask.host_view(read_write);
    for (auto i : A.zeroed_rows()) view[static_cast<std::size_t>(i)] = T{1};
  }
  ovlp.comm->broadcast(mask);

  const auto view = std::as_const(mask).host_view(read);
  for (std::size_t i = 0; i < view.size(); ++i)
    if (view[i] != T{0}) dirichlet.push_back(static_cast<Index>(i));

  std::ranges::sort(dirichlet);
  return dirichlet;
}
} // namespace ddm::detail
