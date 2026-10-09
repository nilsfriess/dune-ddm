#pragma once

#include "ddm/index.hh"

#include <cstddef>
#include <span>
#include <vector>

namespace ddm {
template <class T>
class ElementBlocks {
public:
  void reserve(Index n_elements, Index n_dofs_per_element)
  {
    dof_offsets_.reserve(n_elements + 1);
    value_offsets_.reserve(n_elements + 1);
    dofs_.reserve(static_cast<std::size_t>(n_elements) * n_dofs_per_element);
    values_.reserve(static_cast<std::size_t>(n_elements) * n_dofs_per_element * n_dofs_per_element);
  }

  void add(std::span<const Index> dofs, std::span<const T> values)
  {
    dofs_.insert(dofs_.end(), dofs.begin(), dofs.end());
    values_.insert(values_.end(), values.begin(), values.end());
    dof_offsets_.push_back(dofs_.size());
    value_offsets_.push_back(values_.size());
  }

  std::size_t size() const { return dof_offsets_.size() - 1; }
  std::span<const Index> dofs(std::size_t b) const { return {dofs_.data() + dof_offsets_[b], dofs_.data() + dof_offsets_[b + 1]}; }
  std::span<const T> values(std::size_t b) const { return {values_.data() + value_offsets_[b], values_.data() + value_offsets_[b + 1]}; }

private:
  std::vector<Index> dof_offsets_{0};   // dofs belonging to block i are at dofs_[dof_offsets_[i], dof_offsets[i+1])
  std::vector<Index> value_offsets_{0}; // same for values_
  std::vector<Index> dofs_;
  std::vector<T> values_; // the values for the blocks; stored as a coniguous array (individual blocks are row-major)
};
} // namespace ddm
