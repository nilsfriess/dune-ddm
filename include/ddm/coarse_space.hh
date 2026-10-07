#pragma once

#include "ddm/multivec/multivec.hh"
#include "overlap.hh"

#include <memory>

namespace ddm {
template <class T>
struct CoarseSpace {
  // The subdomain that this coarse space basis is defined on
  std::shared_ptr<Overlap> overlap;

  // The actual basis vectors
  std::shared_ptr<MultiVec<T>> basis;
};
} // namespace ddm
