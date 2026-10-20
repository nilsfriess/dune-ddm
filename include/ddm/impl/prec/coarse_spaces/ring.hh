#pragma once

#include "ddm/coarse_space.hh"
#include "ddm/debug/vec_sink.hh"
#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/impl/prec/coarse_spaces/common.hh"
#include "ddm/impl/prec/coarse_spaces/harmonic_space.hh"
#include "ddm/index.hh"
#include "ddm/logger.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/neumann.hh"
#include "ddm/overlap.hh"
#include "ddm/region.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ddm {
namespace detail {
/* The layers of the ring of ring_coarse_space(), for every index of the overlapping subdomain:
 *   0: the overlap, where the partition of unity D is not 1 (this includes the outermost layer, where D is 0)
 *   1: the indices where D is 1 with a neighbour in layer 0
 *   2: the indices where D is 1 with a neighbour in layer 1, but none in layer 0 (the inner boundary of the ring)
 *  -1: all other indices, further inside of the subdomain
 * Neighbours are taken in graph, a graph on the overlapping index set (e.g. the pattern of the overlapping matrix). For
 * continuous elements with the dofs on the vertices, layers 1 and 2 are the vertices of the first layer of elements
 * where D is 1.
 */
template <class T>
std::vector<int> ring_layers(const Vec<T>& pou, const Pattern& graph)
{
  const Index n = pou.size();
  DDM_CHECK(graph.rows() == n && graph.cols() == n, "ring_layers: graph is {}x{}, but the partition of unity has size {}", graph.rows(), graph.cols(), n);

  // Breadth-first search for two steps, starting from the indices where D is not 1
  std::vector<int> layer(n, -1);
  std::vector<Index> frontier;
  {
    const auto pou_view = pou.host_view(read);
    for (Index i = 0; i < n; ++i)
      if (pou_view[i] != T{1}) {
        layer[i] = 0;
        frontier.push_back(i);
      }
  }
  for (int l = 1; l <= 2; ++l) {
    std::vector<Index> next;
    for (auto i : frontier)
      for (auto j : graph.row(i))
        if (layer[j] < 0) {
          layer[j] = l;
          next.push_back(static_cast<Index>(j));
        }
    frontier = std::move(next);
  }
  return layer;
}

// Projects the columns of V (on the overlapping subdomain) onto H, a space of harmonic functions on the region r and
// numbered by it: on r, every column is replaced by the function in H with its values on the boundary of H. The rows
// outside of r are not changed.
template <class T>
void project_rows(const HarmonicSpace<T>& H, const Region& r, MultiVec<T>& V)
{
  auto X = H.constraint_matrix().local().create_domain_multivector(V.cols());
  V.pack_rows(r.indices(), *X);
  auto x = H.constraint_matrix().create_domain_vector();
  auto y = H.constraint_matrix().create_domain_vector();
  for (Index j = 0; j < X->cols(); ++j) {
    X->extract_column_to_vec(j, x);
    H.project(x, y);
    X->copy_into_column(j, y);
  }
  V.unpack_rows(*X, r.indices(), 0);
}

/* The from_ring extension of ring_coarse_space(), with layer the layers of the ring (see ring_layers()): keeps the
 * values of the columns of V (on the overlapping subdomain) on layers 0 and 1 and replaces them everywhere else, i.e.
 * on layer 2 and further inside, by the A_ovlp-harmonic extension of their values on layer 1. dirichlet are the
 * Dirichlet dofs, sorted.
 */
template <class T>
void extend_from_ring(const Dune::ParameterTree& config, std::span<const int> layer, const LocalMat<T>& A_ovlp, std::span<const Index> dirichlet, MultiVec<T>& V)
{
  // The region of the extension: all indices where the partition of unity is 1, i.e. all but layer 0. Its rows from
  // layer 2 on are complete: their neighbours are not in layer 0.
  std::vector<Index> indices;
  for (Index i = 0; i < std::ssize(layer); ++i)
    if (layer[i] != 0) indices.push_back(i);
  const auto region = Region::from_indices(std::ssize(layer), std::move(indices));

  // Its boundary is layer 1, without the Dirichlet dofs
  std::vector<Index> region_dirichlet;
  std::vector<bool> is_dirichlet(region.size(), false);
  for (auto i : dirichlet)
    if (const auto k = region.region_index(i); k >= 0) {
      region_dirichlet.push_back(k);
      is_dirichlet[k] = true;
    }
  std::vector<Index> boundary;
  for (Index k = 0; k < region.size(); ++k)
    if (layer[region.subdomain_index(k)] == 1 && !is_dirichlet[k]) boundary.push_back(k);

  // Nothing to extend to if the region has no interior dofs, e.g. if the ring covers the whole subdomain
  if (std::ssize(boundary) + std::ssize(region_dirichlet) == region.size()) return;

  auto A_ext = submatrix(A_ovlp, region);
  A_ext->zero_rows(boundary);
  auto A_ext_mat = std::make_shared<const Mat<T>>(Dune::ParameterTree{}, std::move(A_ext), nullptr);
  auto solver = create_subproblem_solver<T>(config.sub("extension_solver"), config.sub("extension_prec"), A_ext_mat);
  const HarmonicSpace<T> space(std::move(A_ext_mat), std::move(boundary), std::move(region_dirichlet), std::move(solver));
  project_rows(space, region, V);
}
} // namespace detail

/** Ring coarse spaces (arXiv:2507.12226): the eigenproblem of geneo_coarse_space() (harmonic = false) or
 *  msgfem_coarse_space() (harmonic = true), posed only on a ring along the boundary of the overlapping subdomain. The
 *  eigenvectors are extended to the overlapping subdomain (see `extension`) and multiplied by the partition of unity D.
 *  Collective.
 *
 *  The ring is determined by D (see detail::ring_layers()): it consists of the overlap, where D is not 1 (layer 0),
 *  and the next two layers of indices in the graph of A_ovlp, where D is 1 (layers 1 and 2). Layer 2 is the inner
 *  boundary of the ring.
 *
 *  On the ring, the eigenproblem is A_neu v = lambda D_R A_neu D_R v. A_neu is the Neumann matrix on the ring, so it
 *  also has Neumann conditions on the inner boundary of the ring. D_R is D on the ring, except on its inner boundary,
 *  where it is zero: D_R follows D up to layer 1, where both are 1, and drops to 0 between layers 1 and 2. This is the
 *  cut-off of arXiv:2507.12226: layer 1 is the boundary of the region where D is 1, and it lies inside of the ring.
 *
 *  With harmonic, v is restricted to the functions on the ring that are A_ovlp-harmonic on all indices of the ring
 *  except the outermost layer and the inner boundary, where they are free, and the Dirichlet dofs, where they are
 *  zero.
 *
 *  A_neu, A_ovlp, the Dirichlet dofs and A_dir are as in msgfem_coarse_space(), A_dir is only needed for the extension
 *  from_boundary.
 *
 *  Config:
 *  - extension:        from_boundary | from_ring (default from_boundary). from_boundary: the A_ovlp-harmonic function
 *                      on the overlapping subdomain with the values of v on the outermost layer, using A_dir.
 *                      from_ring: v on layers 0 and 1, and further inside the A_ovlp-harmonic extension of its values
 *                      on layer 1 (as in arXiv:2507.12226)
 *  - method:           only with harmonic, see solve_harmonic_gevp()
 *  - eigensolver:      config of the eigensolver (see create_eigensolver())
 *  - ring_solver:      config of the solver for the ring matrix with identity rows on the boundary of the ring, only
 *                      with harmonic and a method other than saddle_point (see create_subproblem_solver(), default:
 *                      exact solve)
 *  - ring_prec:        config of the preconditioner of that solver (see create_subproblem_solver())
 *  - extension_solver: config of the solver for the extension from_ring (see create_subproblem_solver(), default:
 *                      exact solve)
 *  - extension_prec:   config of the preconditioner of that solver (see create_subproblem_solver())
 */
template <class T>
std::shared_ptr<CoarseSpace<T>> ring_coarse_space(const Dune::ParameterTree& config, std::shared_ptr<const Overlap> ovlp, const Vec<T>& pou, std::shared_ptr<const Mat<T>> A,
                                                  std::shared_ptr<const Mat<T>> A_ovlp, std::shared_ptr<Solver<T>> A_dir, bool harmonic)
{
  const auto extension = config.get("extension", std::string("from_boundary"));
  DDM_CHECK(extension == "from_boundary" || extension == "from_ring", "ring_coarse_space: unknown extension `{}`, known: from_boundary, from_ring", extension);
  DDM_CHECK(extension != "from_boundary" || A_dir != nullptr, "ring_coarse_space: the extension from_boundary needs the subdomain solver, which requires eliminate_boundary");

  auto subdomain_dirichlet = detail::overlapping_dirichlet_dofs(*A, A_ovlp->local(), *ovlp);

  // The layers of the ring on the overlapping subdomain, see detail::ring_layers()
  const auto ring_layer = detail::ring_layers(pou, A_ovlp->local().pattern());
  std::vector<Index> ring_indices;
  for (Index i = 0; i < std::ssize(ring_layer); ++i)
    if (ring_layer[i] >= 0) ring_indices.push_back(i);
  const auto ring = Region::from_indices(std::ssize(ring_layer), std::move(ring_indices));

  // Visualise the ring of the debug rank (1 on the ring, 0 elsewhere) if the user has installed a vector sink
  if (debug::has_vec_sink<T>()) {
    auto v = A_ovlp->create_domain_vector();
    if (debug::vec_sink_debug_rank<T>() == ovlp->comm->rank()) {
      auto view = v.host_view(read_write);
      for (auto i : ring.indices()) view[i] = T{1};
    }
    ovlp->comm->reduce(v);
    auto v_novlp = A->create_domain_vector();
    v_novlp.copy_n_from(v, v_novlp.size());
    debug::add_vec_to_sink("ring", v_novlp);
  }

  NeumannAssembler<T> na(A->get_symmetric_part() != nullptr ? A->get_symmetric_part() : A, ovlp, ring);
  auto lhs = na.matrix(Dune::ParameterTree{});

  // D_R: the partition of unity on the ring, cut off on its inner boundary
  auto ring_pou = lhs->create_domain_vector();
  {
    const auto pou_view = pou.host_view(read);
    auto ring_pou_view = ring_pou.host_view(write);
    for (Index k = 0; k < ring.size(); ++k) {
      const auto i = ring.subdomain_index(k);
      ring_pou_view[k] = ring_layer[i] == 2 ? T{0} : pou_view[i];
    }
  }
  auto rhs = lhs->clone();
  rhs->scale(ring_pou);

  // The Dirichlet dofs in the numbering of the ring, still sorted. Eliminate them symmetrically; see geneo.hh
  std::vector<Index> dirichlet;
  std::vector<bool> is_dirichlet(ring.size(), false);
  for (auto i : subdomain_dirichlet)
    if (const auto k = ring.region_index(i); k >= 0) {
      dirichlet.push_back(k);
      is_dirichlet[k] = true;
    }
  lhs->zero_rows_and_cols(dirichlet, T{1});
  rhs->zero_rows_and_cols(dirichlet, T{0});

  std::shared_ptr<MultiVec<T>> ring_vectors;
  if (!harmonic) {
    auto eigensolver = create_eigensolver<T>(config.sub("eigensolver"), lhs, rhs);
    eigensolver->solve();
    ring_vectors = eigensolver->vectors();
  }
  else {
    // The boundary of the ring: the outermost layer and the inner boundary, without the Dirichlet dofs. The rows of
    // the inner boundary miss their neighbours outside of the ring, but as boundary rows they are not used. All other
    // rows of the ring are complete.
    std::vector<Index> boundary;
    for (Index k = 0; k < ring.size(); ++k) {
      const auto i = ring.subdomain_index(k);
      if ((ovlp->layer[i] == ovlp->layers || ring_layer[i] == 2) && !is_dirichlet[k]) boundary.push_back(k);
    }

    auto A_ring = submatrix(A_ovlp->local(), ring);
    A_ring->zero_rows(boundary);
    auto A_ring_mat = std::make_shared<const Mat<T>>(Dune::ParameterTree{}, std::move(A_ring), nullptr);

    std::shared_ptr<Solver<T>> ring_solver;
    if (config.get("method", std::string("saddle_point")) != "saddle_point") ring_solver = create_subproblem_solver<T>(config.sub("ring_solver"), config.sub("ring_prec"), A_ring_mat);

    HarmonicSpace<T> ring_space(std::move(A_ring_mat), std::move(boundary), std::move(dirichlet), std::move(ring_solver));
    ring_vectors = solve_harmonic_gevp<T>(config, lhs, rhs, ring_space);
  }

  // Copy the eigenvectors to the overlapping subdomain and extend them
  auto basis = std::shared_ptr<MultiVec<T>>(A_ovlp->local().create_domain_multivector(ring_vectors->cols()));
  basis->unpack_rows(*ring_vectors, ring.indices(), 0);
  if (extension == "from_boundary") {
    // project() only reads the values on the outermost layer, which is part of the ring
    const auto subdomain_space = detail::subdomain_harmonic_space<T>(*ovlp, A_ovlp, std::move(subdomain_dirichlet), std::move(A_dir));
    detail::project_rows(subdomain_space, Region::whole_subdomain(*ovlp), *basis);
  }
  else detail::extend_from_ring(config, ring_layer, A_ovlp->local(), subdomain_dirichlet, *basis);

  basis->scale_rows(pou);
  return std::make_shared<CoarseSpace<T>>(ovlp, std::move(basis));
}
} // namespace ddm
