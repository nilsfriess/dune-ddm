#pragma once

#include "ddm/check.hh"
#include "ddm/impl/mat/istlmat.hh"
#include "ddm/impl/multivec/istlmultivec.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"
#include "ddm/logger.hh"

#include <algorithm>
#include <dune/common/parametertree.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/preconditioners.hh>
#include <memory>
#include <utility>

namespace ddm {
template <class T>
class IstlILUPrec final : public Prec<T> {
  using NativeMat = typename ddm::IstlMat<T>::Native;
  using NativeVec = Dune::BlockVector<T>;

public:
  IstlILUPrec(const Dune::ParameterTree& /*config*/, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
      , ilu(as_istl(this->mat()->local()).native(), 0, 1., false)
      , apply_event_(Logger::get().registerOrGetEvent("ILU", "apply"))
  {
  }

private:
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    Logger::ScopedLog sl{apply_event_};
    // ISTL's ILU works on ISTL vectors, so the vectors are copied through z_tmp_ and r_tmp_
    const auto r_data = as_istl(r).data();
    auto z_data = as_istl(z).data();
    r_tmp_.resize(r_data.size());
    z_tmp_.resize(z_data.size());
    std::copy(r_data.begin(), r_data.end(), r_tmp_.begin());
    ilu.apply(z_tmp_, r_tmp_);
    std::copy(z_tmp_.begin(), z_tmp_.end(), z_data.begin());
  }

  void do_update() override { TODO("IstlILUPrec::do_update"); }

  void do_info() const override { logger::info("ILU preconditioner (using DUNE ISTL's SeqILU)"); }

  Dune::SeqILU<NativeMat, NativeVec, NativeVec> ilu;
  NativeVec z_tmp_;
  NativeVec r_tmp_;
  Logger::Event* apply_event_;
};
} // namespace ddm
