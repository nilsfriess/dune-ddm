#pragma once

#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/overlap.hh"

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
