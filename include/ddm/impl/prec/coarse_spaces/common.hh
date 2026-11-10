#pragma once

#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/overlap.hh"
#include "ddm/vec/vec.hh"

#include <algorithm>
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
} // namespace ddm::detail
