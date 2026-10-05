#pragma once

#include "ddm/check.hh"
#include "ddm/impl/mat/istlmat.hh"
#include "ddm/impl/vec/istlvec.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/preconditioners.hh>
#include <memory>
#include <utility>

namespace ddm {
template <class T>
class IstlILUPrec final : public Prec<T> {
  using NativeMat = typename ddm::IstlMat<T>::Native;
  using NativeVec = typename ddm::IstlVec<T>::Native;

public:
  IstlILUPrec(const Dune::ParameterTree& /*config*/, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
      , ilu(as_istl(this->mat()->local()).native(), 0, 1., false)
  {
  }

private:
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    auto& z_istl = as_istl(z).native();
    const auto& r_istl = as_istl(r).native();
    ilu.apply(z_istl, r_istl);
  }

  void do_update() override { TODO("IstlILUPrec::do_update"); }

  Dune::SeqILU<NativeMat, NativeVec, NativeVec> ilu;
};
} // namespace ddm
