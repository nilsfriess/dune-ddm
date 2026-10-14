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
#include <vector>

namespace ddm {
/** GenEO coarse space: the eigenvectors v of A_neu v = lambda D A_ovlp D v with the smallest eigenvalues, multiplied
 *  by the partition of unity D. A_neu is the Neumann matrix on the overlapping subdomain (A must have been assembled
 *  with record_values), A_ovlp the restriction of A to it. The rows A's zero_rows() was called for are treated as
 *  Dirichlet dofs. Collective.
 *
 *  Config:
 *  - eigensolver: config of the eigensolver (see create_eigensolver())
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> geneo_coarse_space(const Dune::ParameterTree& config, std::shared_ptr<const Overlap> ovlp, const Vec<T>& pou, std::shared_ptr<const Mat<T>> A,
                                                   const LocalMat<T>& A_ovlp)
{
  NeumannAssembler<T> na(A->get_symmetric_part() != nullptr ? A->get_symmetric_part() : A, ovlp);

  auto region = Region::subdomain(*ovlp);
  auto lhs = na.matrix(Dune::ParameterTree{}, region);
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
  basis->scale_rows(pou);
  return std::make_shared<CoarseSpace<T>>(ovlp, std::move(basis));
}
} // namespace ddm
