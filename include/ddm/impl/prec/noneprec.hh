#pragma once

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

  Dune::SolverCategory::Category category() const override
  {
    // This works in sequential and in parallel, so category is whatever the matrix reports
    return this->mat()->category();
  }

private:
  void do_apply(Vec<T>& z, const Vec<T>& r) override { z = r; }

  void do_update() override
  {
    // Update is a no-op here
  }

  void do_info() const override { logger::info("NonePrec preconditioner (only copies the input into the output)"); }
};
} // namespace ddm
