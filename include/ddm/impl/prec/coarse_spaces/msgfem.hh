#pragma once

#include "common.hh"
#include "ddm/coarse_space.hh"
#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/index.hh"
#include "ddm/logger.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/neumann.hh"
#include "ddm/overlap.hh"
#include "ddm/region.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"
#include "harmonic_space.hh"

#include <dune/common/parametertree.hh>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
 *  With inner_product = ring, the harmonic functions are not measured in the A_neu-energy on the overlapping subdomain,
 *  but in the energy of the Neumann matrix on the ring of ring_coarse_space() (see detail::ring_layers()): the energy of
 *  the function on the ring with the same values on the outermost layer that is harmonic for the ring's Neumann matrix
 *  on the rest of the ring. The shift-invert solves of the eigensolver then only need the Neumann matrix on the ring
 *  instead of the one on the whole overlapping subdomain (see detail::solve_harmonic_gevp_two_sided_ring()).
 *
 *  Config:
 *  - method:        how to enforce the constraint in the eigenvalue problem, see solve_harmonic_gevp()
 *  - eigensolver:   config of the eigensolver (see create_eigensolver())
 *  - inner_product: subdomain | ring (default subdomain), see above. ring only works with method two_sided
 *  - ring_solver:   only with inner_product ring, see detail::solve_harmonic_gevp_two_sided_ring()
 *  - ring_prec:     only with inner_product ring, see detail::solve_harmonic_gevp_two_sided_ring()
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> msgfem_coarse_space(const Dune::ParameterTree& config,
                                                    std::shared_ptr<const Overlap> ovlp,
                                                    const Vec<T>& pou,
                                                    std::shared_ptr<const Mat<T>> A,
                                                    std::shared_ptr<const Mat<T>> A_ovlp,
                                                    std::shared_ptr<Solver<T>> A_dir = nullptr)
{
  const auto inner_product = config.get("inner_product", std::string("subdomain"));
  DDM_CHECK(inner_product == "subdomain" || inner_product == "ring", "msgfem_coarse_space: unknown inner_product `{}`, known: subdomain, ring", inner_product);
  DDM_CHECK(inner_product == "subdomain" || config.get("method", std::string("saddle_point")) == "two_sided", "msgfem_coarse_space: inner_product ring only works with method two_sided");

  // TODO: With inner_product ring, the Neumann matrix on the whole subdomain is only needed for D A_neu D (and the size
  //       of the problem), which could also be built without it
  const auto A_sym = A->get_symmetric_part() != nullptr ? A->get_symmetric_part() : A;
  NeumannAssembler<T> na(A_sym, ovlp, Region::whole_subdomain(*ovlp));
  auto lhs = na.matrix(Dune::ParameterTree{});
  auto rhs = lhs->clone();
  rhs->scale(pou);

  // Eliminate the Dirichlet dofs symmetrically; see geneo.hh
  auto dirichlet = detail::overlapping_dirichlet_dofs(*A, A_ovlp->local(), *ovlp);
  lhs->zero_rows_and_cols(dirichlet, T{1});
  rhs->zero_rows_and_cols(dirichlet, T{0});

  auto harmonic_space = detail::subdomain_harmonic_space<T>(*ovlp, A_ovlp, std::move(dirichlet), A_dir);

  std::shared_ptr<MultiVec<T>> basis;
  if (inner_product == "subdomain") basis = solve_harmonic_gevp<T>(config, lhs, rhs, harmonic_space);
  else {
    // The ring of ring_coarse_space(), it contains the outermost layer
    const auto ring_layer = detail::ring_layers(pou, A_ovlp->local().pattern());
    std::vector<Index> ring_indices;
    for (Index i = 0; i < std::ssize(ring_layer); ++i)
      if (ring_layer[i] == 0) ring_indices.push_back(i);
    const auto ring = Region::from_indices(std::ssize(ring_layer), std::move(ring_indices));
    Logger::get().set_stat("Coarse space/ring dofs", static_cast<double>(ring.size()));

    NeumannAssembler<T> na_ring(A_sym, ovlp, ring);
    auto A_ring = na_ring.matrix(Dune::ParameterTree{});

    // B_ring = D A_ring D with D restricted to the ring, only used for the shift
    auto ring_pou = A_ring->create_domain_vector();
    {
      const auto pou_view = pou.host_view(read);
      auto ring_pou_view = ring_pou.host_view(write);
      for (Index k = 0; k < ring.size(); ++k) ring_pou_view[k] = pou_view[ring.subdomain_index(k)];
    }
    auto B_ring = A_ring->clone();
    B_ring->scale(ring_pou);

    // The Dirichlet dofs in the numbering of the ring, eliminated like above
    std::vector<Index> ring_dirichlet;
    for (auto i : harmonic_space.dirichlet())
      if (const auto k = ring.region_index(i); k >= 0) ring_dirichlet.push_back(k);
    A_ring->zero_rows_and_cols(ring_dirichlet, T{1});
    B_ring->zero_rows_and_cols(ring_dirichlet, T{0});

    basis = detail::solve_harmonic_gevp_two_sided_ring<T>(config, lhs, rhs, harmonic_space, ring, A_ring, B_ring);
  }

  basis->scale_rows(pou.multivec());
  return std::make_shared<CoarseSpace<T>>(ovlp, std::move(basis));
}
} // namespace ddm
