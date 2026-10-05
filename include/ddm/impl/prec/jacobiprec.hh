#pragma once

#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <memory>
#include <utility>

namespace ddm {
template <class T>
class JacobiPrec final : public Prec<T> {
public:
  JacobiPrec(const Dune::ParameterTree& /*config*/, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
      , diag(this->mat()->create_range_vector())
  {
    do_update();
  }

private:
  void do_apply(Vec<T>& z, const Vec<T>& r) override { r.pointwise_mult(diag, z); }

  void do_update() override
  {
    this->mat()->get_diag(diag);
    auto diag_view = diag.host_view(read_write);
    for (auto& d : diag_view) d = (d == 0) ? T(1) : T(1) / d;
  }

  Vec<T> diag;
};
} // namespace ddm
