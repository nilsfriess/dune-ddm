#pragma once

#include "ddm/check.hh"
#include "ddm/index.hh"

#include <dune/istl/sparseindexranges.hh>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace ddm {
// Sparsity pattern in CSR form, built by inserting (row, col) pairs in any order (duplicates are ignored), followed
// by finalize(). Thin wrapper around ISTL's UnsequencedSparseIndexRangeBuilder / SparseIndexRanges, so that the rest
// of dune-ddm does not depend on that API directly.
//
// A pattern is also used as a graph on the indices, e.g. for the overlap extension. A finalized pattern can be copied,
// the copies share the CSR data.
class Pattern {
public:
  // ISTL only supports unsigned index types
  using ranges_type = Dune::SparseIndexRanges<std::make_unsigned_t<Index>>;

  // avg_per_row is a performance hint for the initial reservation
  Pattern(Index rows, Index cols, Index avg_per_row = 0)
      : rows_(rows)
      , cols_(cols)
  {
    DDM_CHECK(rows >= 0 && cols >= 0 && avg_per_row >= 0, "pattern: negative sizes {}x{}, avg_per_row {}", rows, cols, avg_per_row);
    builder_ = std::make_unique<builder_type>(rows, cols, avg_per_row);
  }

  void add(Index i, Index j)
  {
    DDM_CHECK(builder_ != nullptr, "pattern: add() called after finalize()");
    DDM_CHECK(0 <= i && i < rows_ && 0 <= j && j < cols_, "pattern: add() index ({}, {}) out of range for {}x{} pattern", i, j, rows_, cols_);
    builder_->addIndex(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
  }

  Pattern(const Pattern& other)
      : rows_(other.rows_)
      , cols_(other.cols_)
      , ranges_(other.ranges())
  {
  }

  Pattern& operator=(const Pattern& other)
  {
    rows_ = other.rows_;
    cols_ = other.cols_;
    builder_.reset();
    ranges_ = other.ranges();
    return *this;
  }

  Pattern(Pattern&&) = default;
  Pattern& operator=(Pattern&&) = default;

  void finalize()
  {
    DDM_CHECK(builder_ != nullptr, "pattern: finalize() called twice");
    ranges_ = std::make_shared<const ranges_type>(std::move(*builder_));
    builder_.reset();
  }

  bool is_finalized() const { return ranges_ != nullptr; }

  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  Index nnz() const { return static_cast<Index>(ranges()->count()); }

  // The column indices of row i, in ascending order. Iterating it yields the indices as unsigned integers
  auto row(Index i) const
  {
    DDM_ASSERT(0 <= i && i < rows_, "pattern: row {} out of range for {} rows", i, rows_);
    return (*ranges())[static_cast<std::size_t>(i)];
  }

  // ISTL representation of the finalized pattern, e.g. for IstlMat to share it without copying
  const std::shared_ptr<const ranges_type>& ranges() const
  {
    DDM_CHECK(ranges_ != nullptr, "pattern: used before finalize()");
    return ranges_;
  }

private:
  using builder_type = Dune::UnsequencedSparseIndexRangeBuilder<>;

  Index rows_;
  Index cols_;
  std::unique_ptr<builder_type> builder_; // the builder is neither copyable nor movable
  std::shared_ptr<const ranges_type> ranges_;
};
} // namespace ddm
