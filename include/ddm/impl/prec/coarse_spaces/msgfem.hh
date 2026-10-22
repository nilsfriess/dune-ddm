#pragma once

#include "common.hh"
#include "ddm/coarse_space.hh"
#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/neumann.hh"
#include "ddm/overlap.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"
#include "harmonic_space.hh"

#include <memory>
#include <utility>

namespace ddm {
/** MsGFEM coarse space: the GenEO eigenproblem A_neu v = lambda D A_neu D v, restricted to the discrete A-harmonic
 *  functions on the overlapping subdomain (A_ovlp v = 0 at the interior dofs). The eigenvectors with the smallest
 *  eigenvalues are multiplied by the partition of unity D. A_neu is the Neumann matrix on the overlapping subdomain
 *  (A must have been assembled with record_values), A_ovlp the restriction of A to it. The rows A's zero_rows() was
 *  called for are treated as Dirichlet dofs. A_dir solves the Dirichlet problem on the overlapping subdomain, it is
 *  needed by all methods except `saddle_point`. Collective.
 *
 *  If a symmetric part is attached to A (see Mat::attach_symmetric_matrix()), A_neu is the Neumann matrix of the
 *  symmetric part (which then needs record_values instead of A), while harmonicity is still defined by A_ovlp. Since D
 *  vanishes on the outermost layer, D A_neu D equals the restriction of the symmetric part to the overlapping subdomain,
 *  scaled by D from both sides.
 *
 *  Config:
 *  - method:      how to enforce the constraint in the eigenvalue problem, see solve_harmonic_gevp()
 *  - eigensolver: config of the eigensolver (see create_eigensolver())
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> msgfem_coarse_space(const Dune::ParameterTree& config,
                                                    std::shared_ptr<const Overlap> ovlp,
                                                    const Vec<T>& pou,
                                                    std::shared_ptr<const Mat<T>> A,
                                                    std::shared_ptr<const Mat<T>> A_ovlp,
                                                    std::shared_ptr<Solver<T>> A_dir = nullptr)
{

  NeumannAssembler<T> na(A->get_symmetric_part() != nullptr ? A->get_symmetric_part() : A, ovlp, Region::whole_subdomain(*ovlp));
  auto lhs = na.matrix(Dune::ParameterTree{});
  auto rhs = lhs->clone();
  rhs->scale(pou);

  // Eliminate the Dirichlet dofs symmetrically; see geneo.hh
  auto dirichlet = detail::overlapping_dirichlet_dofs(*A, A_ovlp->local(), *ovlp);
  lhs->zero_rows_and_cols(dirichlet, T{1});
  rhs->zero_rows_and_cols(dirichlet, T{0});

  auto harmonic_space = detail::subdomain_harmonic_space<T>(*ovlp, A_ovlp, std::move(dirichlet), A_dir);

  auto basis = solve_harmonic_gevp<T>(config, lhs, rhs, harmonic_space);

  basis->scale_rows(pou.multivec());
  return std::make_shared<CoarseSpace<T>>(ovlp, std::move(basis));
}
} // namespace ddm
