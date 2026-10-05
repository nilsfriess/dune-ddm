#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"

#include <dune/common/parametertree.hh>
#include <memory>
#include <utility>

namespace ddm {
template <class T>
class NonePrec final : public Prec<T> {
public:
  NonePrec(const Dune::ParameterTree& /*config*/, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
  {
  }

private:
  void do_apply(Vec<T>& z, const Vec<T>& r) override { z = r; }

  void do_update() override
  {
    // Update is a no-op here
  }
};
} // namespace ddm
