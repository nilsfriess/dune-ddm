#pragma once

#include "ddm/coarse_space.hh"
#include "ddm/impl/prec/coarse_spaces/common.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/overlap.hh"
#include "ddm/region.hh"
#include "ddm/vec/vec.hh"

#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ddm {
/** Nicolaides coarse space: If the matrix has near kernel vectors set (using Mat<T>::set_near_kernel()), these vectors multiplied
 *  by the partition of unity are used as basis vectors. Otherwise, only the partition of unity itself is used (i.e. the constant
 *  function multiplied by the partition of unity). The vectors are set to zero at the Dirichlet dofs (the rows A's zero_rows()
 *  was called for).
 *
 *  Every local subdomain j (see partition_matrix_with_overlap()) contributes the vectors D_j R_j v, with its partition
 *  of unity D_j = pous[j]. They are block j of the coarse space (see CoarseSpace). A_ovlp is the matrix on the
 *  overlapping index set of the rank.
 *
 *  Collective.
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> nicolaides_coarse_space(std::shared_ptr<const Overlap> ovlp,
                                                        std::span<const LocalSubdomain> subdomains,
                                                        std::span<const std::shared_ptr<const Vec<T>>> pous,
                                                        const Mat<T>& A,
                                                        const LocalMat<T>& A_ovlp)
{
  DDM_CHECK(subdomains.size() == pous.size(), "nicolaides_coarse_space: {} local subdomains, but {} partitions of unity", subdomains.size(), pous.size());

  // The vectors v on the overlapping index set of the rank: the near kernel vectors, or the constant vector. They are
  // zero at the Dirichlet dofs, so that D_j R_j v is as well
  std::vector<Vec<T>> vectors;
  if (not A.get_near_kernel().empty()) {
    for (const auto& kernel : A.get_near_kernel()) {
      auto& v = vectors.emplace_back(A_ovlp.create_domain_vector());
      v.zero();
      v.copy_n_from(kernel, kernel.size());
      ovlp->comm->broadcast(v);
    }
  }
  else {
    auto& v = vectors.emplace_back(A_ovlp.create_domain_vector());
    v = T{1};
  }
  const auto dirichlet = detail::overlapping_dirichlet_dofs(A, A_ovlp, *ovlp);
  for (auto& v : vectors) {
    auto view = v.host_view(read_write);
    for (auto i : dirichlet) view[i] = T{0};
  }

  auto blocks = detail::local_bases<T>(subdomains, [&](std::size_t j) {
    const auto indices = subdomains[j].region.indices();
    auto basis_j = pous[j]->multivec().create_multivector(std::ssize(indices), std::ssize(vectors));
    auto w = *pous[j]; // a vector on local subdomain j
    for (std::size_t i = 0; i < vectors.size(); ++i) {
      vectors[i].multivec().pack_rows(indices, w.multivec());
      pous[j]->pointwise_mult(w, w);
      column(*basis_j, static_cast<Index>(i)) = w;
    }
    return basis_j;
  });

  return std::make_shared<CoarseSpace<T>>(std::move(ovlp), std::move(blocks));
}
} // namespace ddm
