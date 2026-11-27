#pragma once

#include "ddm/coarse_space.hh"
#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/impl/prec/coarse_spaces/common.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/neumann.hh"
#include "ddm/overlap.hh"
#include "ddm/region.hh"
#include "ddm/vec/vec.hh"

#include <memory>
#include <span>
#include <utility>

namespace ddm {
/** GenEO coarse space: the eigenvectors v of A_neu v = lambda D A_neu D v with the smallest eigenvalues, multiplied
 *  by the partition of unity D. A_neu is the Neumann matrix on the overlapping subdomain (A must have been assembled
 *  with record_values), A_ovlp is the restriction of A to the overlapping subdomain. The rows A's zero_rows() was
 *  called for are treated as Dirichlet dofs. If a symmetric matrix is attached to A (see Mat<T>::get_symmetric_part())
 *  then this matrix (instead of A) is used to build the Neumann matrix (this is the eigenvalue problem of the so-called
 *  Δ-GenEO method from arxiv:2103.16703).
 *
 *  With local subdomains (see partition_matrix_with_overlap()), every local subdomain j solves this eigenproblem on its
 *  own region, with its Neumann matrix and its partition of unity D_j = pous[j]. Its vectors are block j of the coarse
 *  space (see CoarseSpace). A_ovlp is then the matrix on the overlapping index set of the rank.
 *
 *  Collective.
 *
 *  Config:
 *  - eigensolver: config of the eigensolver (see create_eigensolver()), the same for every local subdomain
 *  - neumann_mat: config of the Neumann matrix
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> geneo_coarse_space(const Dune::ParameterTree& config,
                                                   std::shared_ptr<const Overlap> ovlp,
                                                   std::span<const LocalSubdomain> subdomains,
                                                   std::span<const std::shared_ptr<const Vec<T>>> pous,
                                                   std::shared_ptr<const Mat<T>> A,
                                                   const LocalMat<T>& A_ovlp)
{
  DDM_CHECK(subdomains.size() == pous.size(), "geneo_coarse_space: {} local subdomains, but {} partitions of unity", subdomains.size(), pous.size());

  // The Neumann matrices are built for A (or the symmmetric matrix attached to A, if there is one)
  NeumannAssembler<T> na(A->get_symmetric_part() != nullptr ? A->get_symmetric_part() : A, ovlp);
  const auto dirichlet = detail::overlapping_dirichlet_dofs(*A, A_ovlp, *ovlp);

  auto blocks = detail::local_bases<T>(subdomains, [&](std::size_t j) {
    const auto& region = subdomains[j].region;
    auto lhs = na.matrix(config.sub("neumann_mat"), region);
    auto rhs = lhs->clone();
    rhs->scale(*pous[j]);

    // Eliminate the Dirichlet dofs symmetrically but put zeros on the diagonal of B in rows
    // correpsonding to Dirichlet dofs. This ensures that the computed eigenvectors will
    // also be zero there, which is what we want.
    const auto region_dirichlet = region.region_indices(dirichlet);
    lhs->zero_rows_and_cols(region_dirichlet, T{1});
    rhs->zero_rows_and_cols(region_dirichlet, T{0});

    auto eigensolver = create_eigensolver<T>(config.sub("eigensolver"), lhs, rhs);
    eigensolver->solve();

    auto basis_j = eigensolver->vectors();
    basis_j->scale_rows(pous[j]->multivec());
    return basis_j;
  });
  return std::make_shared<CoarseSpace<T>>(std::move(ovlp), std::move(blocks));
}
} // namespace ddm
