#pragma once

#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/overlap.hh"
#include "metis.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace ddm {
/** A part of the overlapping subdomain of an Overlap, e.g. a ring along its boundary. Matrices and vectors on a region
 *  are numbered by the region: region index r is the subdomain index subdomain_index(r). The region indices are in the
 *  same order as their subdomain indices.
 */
class Region {
public:
  // The whole overlapping subdomain
  static Region whole_subdomain(const Overlap& ovlp)
  {
    Region r;
    r.indices_.resize(ovlp.layer.size());
    std::iota(r.indices_.begin(), r.indices_.end(), Index{0});
    r.region_index_ = r.indices_;
    r.is_subdomain_ = true;
    return r;
  }

  // The given indices of an overlapping subdomain with subdomain_size indices, they must be ascending
  static Region from_indices(Index subdomain_size, std::vector<Index> indices)
  {
    DDM_CHECK(std::ranges::is_sorted(indices) && std::ranges::adjacent_find(indices) == indices.end(), "Region::from_indices: the indices must be ascending");
    DDM_CHECK(indices.empty() || (indices.front() >= 0 && indices.back() < subdomain_size), "Region::from_indices: the indices must be in [0, {})", subdomain_size);
    Region r;
    r.region_index_.assign(subdomain_size, -1);
    for (Index k = 0; k < std::ssize(indices); ++k) r.region_index_[indices[k]] = k;
    r.indices_ = std::move(indices);
    return r;
  }

  // True if the region was created with whole_subdomain()
  bool is_subdomain() const
  {
    return is_subdomain_;
  }

  // The number of indices in the region
  Index size() const
  {
    return std::ssize(indices_);
  }

  // The number of indices in the overlapping subdomain the region was created for
  Index subdomain_size() const
  {
    return std::ssize(region_index_);
  }

  // The subdomain index of every region index, ascending
  std::span<const Index> indices() const
  {
    return indices_;
  }

  // The subdomain index of a region index
  Index subdomain_index(Index region_index) const
  {
    DDM_ASSERT(0 <= region_index && region_index < size(), "Region: region index {} out of range for {} indices", region_index, size());
    return indices_[region_index];
  }

  // The region index of a subdomain index, -1 if it is not in the region
  Index region_index(Index subdomain_index) const
  {
    DDM_ASSERT(0 <= subdomain_index && subdomain_index < std::ssize(region_index_), "Region: subdomain index {} out of range for {} indices", subdomain_index, region_index_.size());
    return region_index_[subdomain_index];
  }

private:
  std::vector<Index> indices_;      ///< region index -> subdomain index
  std::vector<Index> region_index_; ///< subdomain index -> region index, -1 outside the region
  bool is_subdomain_ = false;
};

/** A local subdomain, i.e. one of several subdomains on one rank: a region of the rank's overlapping index set and the
 *  layer of each of its indices.
 */
struct LocalSubdomain {
  Region region;
  std::vector<int> layer; ///< for every region index: 0 in the core, k for the indices added in round k (at most ovlp.layers)
};

/** Splits the overlapping subdomain of ovlp into `parts` local subdomains. A is the matrix on the overlapping index set
 *  (e.g. from overlapping_matrix()), only its pattern is used. This doesn't return the matrices, it only returns the
 *  regions. Use `submatrix` to extract the submatrix of a region.
 *
 *  1. METIS partitions the original indices of ovlp (the first ovlp.n_original) into `parts` disjoint sets.
 *  2. Minimal overlap: neighbouring parts share the indices on their interface. For every edge (i, j) of the graph with
 *     part(i) < part(j), i is also added to part j. These are the cores of the local subdomains. They overlap like the
 *     original index sets of the ranks from a partition of the elements, which share the dofs on their interface.
 *  3. Every core is extended by ovlp.layers layers along the rows of A (as in extend_overlap()). The cores only contain
 *     original indices, so the extension does not leave the overlapping index set.
 *
 *  With parts == 1, the result is the whole overlapping subdomain with the layers of ovlp.
 */
template <class T>
std::vector<LocalSubdomain> partition_matrix_with_overlap(const LocalMat<T>& A, const Overlap& ovlp, int parts)
{
  const auto n_ext = static_cast<Index>(ovlp.layer.size());
  DDM_CHECK(parts >= 1, "partition_matrix_with_overlap: need at least one part, got {}", parts);
  DDM_CHECK(A.rows() == n_ext && A.cols() == n_ext, "partition_matrix_with_overlap: matrix is {}x{}, but the overlapping index set has {} indices", A.rows(), A.cols(), n_ext);
  if (parts == 1) return {LocalSubdomain{Region::whole_subdomain(ovlp), ovlp.layer}};

  const auto& pattern = A.pattern();
  const Index n = ovlp.n_original;

  // Extract the graph of the original indices in a format METIS understands
  // TODO(20261127-123537): This assumes that the graph of A is symmetric
  std::vector<idx_t> xadj(n + 1);
  std::vector<idx_t> adjncy;
  xadj[0] = 0;
  for (Index i = 0; i < n; ++i) {
    for (auto j : pattern.row(i))
      if (static_cast<Index>(j) != i && static_cast<Index>(j) < n) adjncy.push_back(static_cast<idx_t>(j));
    xadj[i + 1] = static_cast<idx_t>(adjncy.size());
  }

  idx_t nvtxs = n;
  idx_t ncon = 1;
  idx_t nparts = parts;
  idx_t edgecut;
  std::vector<idx_t> part(n);

  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_CONTIG] = 1;

  auto ok = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr, &nparts, nullptr, nullptr, options, &edgecut, part.data());
  DDM_CHECK(ok == METIS_OK, "partition_matrix_with_overlap: METIS_PartGraphKway failed");

  // The cores: the parts with minimal overlap. An index can be added to a core several times, the duplicates are
  // removed below
  std::vector<std::vector<Index>> cores(parts);
  for (Index i = 0; i < n; ++i) {
    cores[part[i]].push_back(i);
    for (auto j : pattern.row(i))
      if (static_cast<Index>(j) < n && part[j] > part[i]) cores[part[j]].push_back(i);
  }

  // Extend every core by breadth-first search. dist is -1 for the indices not reached yet, it is reset after each core
  std::vector<LocalSubdomain> result;
  result.reserve(parts);
  std::vector<int> dist(n_ext, -1);
  for (auto& core : cores) {
    std::ranges::sort(core);
    core.erase(std::unique(core.begin(), core.end()), core.end());
    DDM_CHECK(!core.empty(), "partition_matrix_with_overlap: METIS returned an empty part ({} parts of {} indices)", parts, n);

    std::vector<Index> indices = core;
    for (auto i : core) dist[i] = 0;
    std::vector<Index> frontier = std::move(core);
    for (int round = 1; round <= ovlp.layers; ++round) {
      std::vector<Index> added;
      for (auto i : frontier)
        for (auto j : pattern.row(i))
          if (dist[j] < 0) {
            dist[j] = round;
            added.push_back(static_cast<Index>(j));
          }
      indices.insert(indices.end(), added.begin(), added.end());
      frontier = std::move(added);
    }

    std::ranges::sort(indices);
    std::vector<int> layer(indices.size());
    for (std::size_t k = 0; k < indices.size(); ++k) layer[k] = dist[indices[k]];
    for (auto i : indices) dist[i] = -1;
    // TODO(20261127-144929): The region stores a map of the size of the whole overlapping index set, per local subdomain
    result.push_back({Region::from_indices(n_ext, std::move(indices)), std::move(layer)});
  }
  return result;
}

/** Returns the restriction of A to the region r: the rows and columns of the indices in r, numbered by r. A is a
 *  matrix on the overlapping subdomain r was created for, the result has the same type. The backend of A must support
 *  host_csr().
 */
template <class T>
std::shared_ptr<LocalMat<T>> submatrix(const LocalMat<T>& A, const Region& r)
{
  DDM_CHECK(A.rows() == r.subdomain_size() && A.cols() == r.subdomain_size(), "submatrix: matrix is {}x{}, but the region was created for {} indices", A.rows(), A.cols(), r.subdomain_size());
  const auto csr = A.host_csr();

  // Calls f(row, col, value) for all entries in rows and columns of the region, in the numbering of the region
  const auto for_each_entry = [&](auto&& f) {
    for (Index k = 0; k < r.size(); ++k) {
      const auto i = r.subdomain_index(k);
      for (auto p = csr.row_ptr[i]; p < csr.row_ptr[i + 1]; ++p)
        if (const auto c = r.region_index(csr.cols[p]); c >= 0) f(k, c, csr.values[p]);
    }
  };

  Pattern pattern(r.size(), r.size());
  for_each_entry([&](Index k, Index c, T) { pattern.add(k, c); });
  pattern.finalize();

  auto S = A.create_like(pattern);
  for_each_entry([&](Index k, Index c, T v) { S->add_values(std::span(&k, 1), std::span(&c, 1), std::span(&v, 1)); });
  S->assemble();
  return S;
}
} // namespace ddm
