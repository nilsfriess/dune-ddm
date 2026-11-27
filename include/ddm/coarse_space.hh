#pragma once

#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/overlap.hh"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace ddm {
/** The coarse basis vectors of one rank, defined on its overlapping index set.
 *
 *  The vectors are stored in blocks, one per local subdomain (see partition_matrix_with_overlap()). A block only holds
 *  the entries of its vectors on its local subdomain, they are zero everywhere else. The basis of the rank is all
 *  blocks side by side: first the vectors of block 0, then those of block 1, and so on. With one subdomain per rank,
 *  there is a single block that covers the whole overlapping index set.
 */
template <class T>
struct CoarseSpace {
  struct Block {
    std::vector<Index> indices;           ///< the indices of the overlapping index set the rows of `vectors` belong to, ascending; empty if the block covers all of them
    std::shared_ptr<MultiVec<T>> vectors; ///< one column per basis vector

    // True if the block covers the whole overlapping index set
    bool covers_all() const
    {
      return indices.empty();
    }
  };

  // A single block that covers the whole overlapping index set
  CoarseSpace(std::shared_ptr<const Overlap> ovlp, std::shared_ptr<MultiVec<T>> basis)
      : overlap(std::move(ovlp))
  {
    blocks.push_back({{}, std::move(basis)});
    check();
  }

  CoarseSpace(std::shared_ptr<const Overlap> ovlp, std::vector<Block> blocks_)
      : overlap(std::move(ovlp))
      , blocks(std::move(blocks_))
  {
    check();
  }

  // The number of basis vectors of the rank
  Index size() const
  {
    Index n = 0;
    for (const auto& block : blocks) n += block.vectors->cols();
    return n;
  }

  std::shared_ptr<const Overlap> overlap; ///< the overlapping index set the basis is defined on
  std::vector<Block> blocks;

private:
  void check() const
  {
    DDM_CHECK(!blocks.empty(), "CoarseSpace: needs at least one block");
    const auto n = static_cast<Index>(overlap->layer.size());
    for (std::size_t b = 0; b < blocks.size(); ++b) {
      const auto& block = blocks[b];
      DDM_CHECK(block.vectors != nullptr, "CoarseSpace: block {} has no vectors", b);
      DDM_CHECK(!block.covers_all() || blocks.size() == 1, "CoarseSpace: a block that covers the whole overlapping index set must be the only one");
      const auto rows = block.covers_all() ? n : std::ssize(block.indices);
      DDM_CHECK(block.vectors->rows() == rows, "CoarseSpace: block {} has {} rows, but {} indices", b, block.vectors->rows(), rows);
    }
  }
};
} // namespace ddm
