#pragma once

#include "ddm/coarse_space.hh"
#include "ddm/eigensolver/eigensolver.hh"
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
  // The Dirichlet dofs on the overlapping subdomain. Every rank holding a row passes it to zero_rows(), so the owners
  // know all of them and can hand them on to the copies in the overlap.
  std::vector<Index> dirichlet;
  {
    auto mask = A_ovlp.create_range_vector();
    {
      auto view = mask.host_view(read_write);
      for (auto i : A->zeroed_rows()) view[static_cast<std::size_t>(i)] = T{1};
    }
    ovlp->comm->broadcast(mask);

    const auto view = std::as_const(mask).host_view(read);
    for (std::size_t i = 0; i < view.size(); ++i)
      if (view[i] != T{0}) dirichlet.push_back(static_cast<Index>(i));
  }

  NeumannAssembler<T> na(std::move(A), ovlp);

  auto region = Region::subdomain(*ovlp);
  auto lhs = na.matrix(Dune::ParameterTree{}, region);
  auto rhs = A_ovlp.clone();
  rhs->scale(pou);

  // Eliminate the Dirichlet dofs symmetrically: with a zero diagonal in rhs, their eigenvalues are infinite, so they are
  // never selected, and the computed eigenvectors vanish there
  lhs->zero_rows_and_cols(dirichlet, T{1});
  rhs->zero_rows_and_cols(dirichlet, T{0});

  auto eigensolver = create_eigensolver<T>(config.sub("eigensolver"), lhs, rhs);
  eigensolver->solve();

  auto basis = eigensolver->vectors();
  basis->scale_rows(pou);
  return std::make_shared<CoarseSpace<T>>(ovlp, std::move(basis));
}
} // namespace ddm
