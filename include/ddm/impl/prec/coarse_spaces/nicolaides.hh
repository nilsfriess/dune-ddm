#pragma once

#include "ddm/coarse_space.hh"
#include "ddm/impl/prec/coarse_spaces/common.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/overlap.hh"
#include "ddm/vec/vec.hh"

#include <memory>
#include <utility>

namespace ddm {
/** Nicolaides coarse space: one basis vector per subdomain, the partition of unity. It is set to zero at the
 *  Dirichlet dofs (the rows A's zero_rows() was called for), so that the coarse problem does not depend on the
 *  diagonal entries chosen for these rows. Collective.
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> nicolaides_coarse_space(std::shared_ptr<const Overlap> ovlp, const Vec<T>& pou, const Mat<T>& A, const LocalMat<T>& A_ovlp)
{
  auto v = pou;
  {
    auto view = v.host_view(read_write);
    for (auto i : detail::overlapping_dirichlet_dofs(A, A_ovlp, *ovlp)) view[i] = T{0};
  }

  auto basis = std::shared_ptr<MultiVec<T>>(A_ovlp.create_domain_multivector(1));
  basis->copy_into_column(0, v);
  return std::make_shared<CoarseSpace<T>>(std::move(ovlp), std::move(basis));
}
} // namespace ddm
