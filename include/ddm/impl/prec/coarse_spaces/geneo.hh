#pragma once

#include "ddm/coarse_space.hh"
#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/impl/prec/coarse_spaces/common.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/neumann.hh"
#include "ddm/overlap.hh"
#include "ddm/vec/vec.hh"

#include <memory>
#include <utility>

namespace ddm {
/** GenEO coarse space: the eigenvectors v of A_neu v = lambda D A_neu D v with the smallest eigenvalues, multiplied
 *  by the partition of unity D. A_neu is the Neumann matrix on the overlapping subdomain (A must have been assembled
 *  with record_values), A_ovlp is the restriction of A to the overlapping subdomain. The rows A's zero_rows() was
 *  called for are treated as Dirichlet dofs. If a symmetric matrix is attached to A (see Mat<T>::get_symmetric_part())
 *  then this matrix (instead of A) is used to build the Neumann matrix (this is the eigenvalue problem of the so-called
 *  Δ-GenEO method from arxiv:2103.16703).
 *
 *  Collective.
 *
 *  Config:
 *  - eigensolver: config of the eigensolver (see create_eigensolver())
 *  - neumann_mat: config of the Neumann matrix
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> geneo_coarse_space(const Dune::ParameterTree& config,
                                                   std::shared_ptr<const Overlap> ovlp,
                                                   const Vec<T>& pou,
                                                   std::shared_ptr<const Mat<T>> A,
                                                   const LocalMat<T>& A_ovlp)
{
  // Build the Neumann matrix for A (or the symmmetric matrix attached to A, if there is one)
  NeumannAssembler<T> na(A->get_symmetric_part() != nullptr ? A->get_symmetric_part() : A, ovlp, Region::whole_subdomain(*ovlp));
  auto lhs = na.matrix(config.sub("neumann_mat"));
  auto rhs = lhs->clone();
  rhs->scale(pou);

  // Eliminate the Dirichlet dofs symmetrically but put zeros on the diagonal of B in rows
  // correpsonding to Dirichlet dofs. This ensures that the computed eigenvectors will
  // also be zero there, which is what we want.
  auto dirichlet = detail::overlapping_dirichlet_dofs(*A, A_ovlp, *ovlp);
  lhs->zero_rows_and_cols(dirichlet, T{1});
  rhs->zero_rows_and_cols(dirichlet, T{0});

  auto eigensolver = create_eigensolver<T>(config.sub("eigensolver"), lhs, rhs);
  eigensolver->solve();

  auto basis = eigensolver->vectors();
  basis->scale_rows(pou.multivec());
  return std::make_shared<CoarseSpace<T>>(ovlp, std::move(basis));
}
} // namespace ddm
