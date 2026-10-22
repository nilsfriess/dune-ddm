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
/** Nicolaides coarse space: If the matrix has near kernel vectors set (using Mat<T>::set_near_kernel()), these vectors multiplied
 *  by the partition of unity are used as basis vectors. Otherwise, only the partition of unity itself is used (i.e. the constant
 *  function multiplied by the partition of unity). The vectors are set to zero at the Dirichlet dofs (the rows A's zero_rows()
 *  was called for).
 *
 *  Collective.
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> nicolaides_coarse_space(std::shared_ptr<const Overlap> ovlp, const Vec<T>& pou, const Mat<T>& A, const LocalMat<T>& A_ovlp)
{
  std::shared_ptr<MultiVec<T>> basis;

  const auto zero_at_dirichlet = [&](auto& v) {
    auto view = v.host_view(read_write);
    for (auto i : detail::overlapping_dirichlet_dofs(A, A_ovlp, *ovlp)) view[i] = T{0};
  };

  if (not A.get_near_kernel().empty()) {
    const auto& kernel = A.get_near_kernel();
    basis = std::shared_ptr<MultiVec<T>>(A_ovlp.create_domain_multivector(kernel.size()));
    auto v = A_ovlp.create_domain_vector();
    for (unsigned int i = 0; i < kernel.size(); ++i) {
      v.zero();
      v.copy_n_from(kernel[i], kernel[i].size());
      ovlp->comm->broadcast(v);
      v.pointwise_mult(pou, v);
      zero_at_dirichlet(v);
      column(*basis, i) = v;
    }
  }
  else {
    auto v = pou;
    zero_at_dirichlet(v);
    basis = std::shared_ptr<MultiVec<T>>(A_ovlp.create_domain_multivector(1));
    column(*basis, 0) = v;
  }

  return std::make_shared<CoarseSpace<T>>(std::move(ovlp), std::move(basis));
}
} // namespace ddm
