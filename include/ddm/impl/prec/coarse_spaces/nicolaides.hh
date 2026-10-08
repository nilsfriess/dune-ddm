#pragma once

#include "ddm/coarse_space.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/overlap.hh"
#include "ddm/vec/vec.hh"

#include <memory>
#include <utility>

namespace ddm {
template <class T>
std::shared_ptr<CoarseSpace<T>> nicolaides_coarse_space(std::shared_ptr<const Overlap> ovlp, const Vec<T>& pou, const LocalMat<T>& A_ovlp)
{
  auto basis = std::shared_ptr<MultiVec<T>>(A_ovlp.create_domain_multivector(1));
  basis->copy_into_column(0, pou);
  return std::make_shared<CoarseSpace<T>>(std::move(ovlp), std::move(basis));
}
} // namespace ddm
