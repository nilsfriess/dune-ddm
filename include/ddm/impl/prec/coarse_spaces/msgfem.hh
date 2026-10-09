#pragma once

#include "common.hh"
#include "ddm/coarse_space.hh"
#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/neumann.hh"
#include "ddm/overlap.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"
#include "harmonic_space.hh"

#include <algorithm>
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
 *  - method:      how to enforce the constraint in the eigenvalue problem, see solve_harmonic_gevp()
 *  - eigensolver: config of the eigensolver (see create_eigensolver())
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> msgfem_coarse_space(const Dune::ParameterTree& config, std::shared_ptr<const Overlap> ovlp, const Vec<T>& pou, std::shared_ptr<const Mat<T>> A,
                                                    std::shared_ptr<const Mat<T>> A_ovlp, std::shared_ptr<Solver<T>> A_dir = nullptr)
{

  NeumannAssembler<T> na(A, ovlp);

  auto region = Region::subdomain(*ovlp);
  auto lhs = na.matrix(Dune::ParameterTree{}, region);
  auto rhs = A_ovlp->local().clone();
  rhs->scale(pou);

  // Eliminate the Dirichlet dofs symmetrically; see geneo.hh
  auto dirichlet = detail::overlapping_dirichlet_dofs(*A, A_ovlp->local(), *ovlp);
  lhs->zero_rows_and_cols(dirichlet, T{1});
  rhs->zero_rows_and_cols(dirichlet, T{0});

  // Find the subdomain boundary indices (without the dirichlet dofs)
  std::vector<Index> boundary;
  boundary.reserve(ovlp->layer.size() - ovlp->n_original); // This is a (quite generous) upper bound
  for (std::size_t i = 0; i < ovlp->layer.size(); ++i)
    if (ovlp->layer[i] == ovlp->layers) boundary.push_back(static_cast<Index>(i));
  std::ranges::sort(boundary);

  // Remove dofs from boundary that are also Dirichlet dofs
  auto ib = std::begin(dirichlet);
  auto iter = std::remove_if(std::begin(boundary), std::end(boundary), [&](Index x) -> bool {
    while (ib != std::end(dirichlet) && *ib < x) ++ib;
    return (ib != std::end(dirichlet) && *ib == x);
  });
  boundary.erase(iter, boundary.end());

  HarmonicSpace<T> harmonic_space(A_ovlp, boundary, dirichlet, A_dir);
  auto basis = solve_harmonic_gevp<T>(config, lhs, rhs, harmonic_space);

  basis->scale_rows(pou);
  return std::make_shared<CoarseSpace<T>>(ovlp, std::move(basis));
}
} // namespace ddm
