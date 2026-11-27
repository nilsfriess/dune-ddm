#pragma once

#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/overlap.hh"
#include "ddm/region.hh"
#include "ddm/vec/vec.hh"

#include <algorithm>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ddm::detail {

/** The Dirichlet dofs on the overlapping subdomain. This just broadcasts the indices in
 *  A.zeroed_rows() to everyone. Returns the indices sorted. Collective.
 */
template <class T>
std::vector<Index> overlapping_dirichlet_dofs(const Mat<T>& A, const LocalMat<T>& A_ovlp, const Overlap& ovlp)
{
  std::vector<Index> dirichlet;
  auto mask = A_ovlp.create_range_vector();
  {
    auto view = mask.host_view(read_write);
    for (auto i : A.zeroed_rows()) view[static_cast<std::size_t>(i)] = T{1};
  }
  ovlp.comm->broadcast(mask);

  const auto view = std::as_const(mask).host_view(read);
  for (std::size_t i = 0; i < view.size(); ++i)
    if (view[i] != T{0}) dirichlet.push_back(static_cast<Index>(i));

  std::ranges::sort(dirichlet);
  return dirichlet;
}

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
/** Builds the coarse basis of a rank from its local subdomains (see partition_matrix_with_overlap()).
 *
 *  local_basis(j) returns the basis vectors of local subdomain j, numbered by its region. They are extended by zero to
 *  the overlapping index set of the rank and put side by side: first the vectors of subdomain 0, then those of
 *  subdomain 1, and so on. The subdomains can have different numbers of vectors. A single local subdomain covers the
 *  whole overlapping subdomain, so its basis is returned as it is.
 *
 *  local_basis is called for one subdomain after the other, so only one subdomain needs its memory (e.g. for an
 *  eigensolver) at a time. A_ovlp is the matrix on the overlapping index set of the rank, it is only used to create
 *  the result.
 */
template <class T, class F>
std::shared_ptr<MultiVec<T>> local_bases(std::span<const LocalSubdomain> subdomains, const LocalMat<T>& A_ovlp, F&& local_basis)
{
  std::vector<std::shared_ptr<MultiVec<T>>> bases;
  bases.reserve(subdomains.size());
  Index cols = 0;
  for (std::size_t j = 0; j < subdomains.size(); ++j) {
    const auto& basis = bases.emplace_back(local_basis(j));
    DDM_CHECK(basis->rows() == subdomains[j].region.size(), "local_bases: the basis of local subdomain {} has {} rows, but the subdomain has {} indices", j, basis->rows(),
              subdomains[j].region.size());
    cols += basis->cols();
  }
  if (subdomains.size() == 1 && subdomains[0].region.is_subdomain()) return bases[0];

  // The result is zero-initialized, so every basis vector is zero outside of its subdomain
  std::shared_ptr<MultiVec<T>> result(A_ovlp.create_domain_multivector(cols));
  Index first = 0;
  for (std::size_t j = 0; j < subdomains.size(); ++j) {
    result->unpack_rows(*bases[j], subdomains[j].region.indices(), first);
    first += bases[j]->cols();
  }
  return result;
}
} // namespace ddm::detail
