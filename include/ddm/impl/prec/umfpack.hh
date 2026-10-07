#pragma once

#include "ddm/check.hh"
#include "ddm/impl/mat/istlmat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/ibcrsmatrix.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>
#include <memory>
#include <utility>

namespace ddm {
template <class T>
class UMFPACKPrec final : public Prec<T> {
public:
  UMFPACKPrec(const Dune::ParameterTree& /*config*/, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
  {
    // TODO(20261006-192425): It should be possible to mark a preconditioner or solver as "sequential" on a higher level
    DDM_CHECK(this->mat()->sequential(), "UMFPACK is a sequential solver");
    do_update();
  }

private:
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    Dune::InverseOperatorResult res;

    auto &x = as_istl(z).native();
    auto b = as_istl(r).native(); // UMFPack solver takes the rhs as a non-const reference so we need to copy here

    solver_.apply(x, b, res);
  }

  void do_update() override
  {
    solver_.setOption(UMFPACK_ORDERING, UMFPACK_ORDERING_METIS);
    solver_.setOption(UMFPACK_IRSTEP, 0);
    solver_.setMatrix(as_istl(this->mat()->local()).native());
  }

  void do_info() const override { logger::info("UMFPACK solver"); }

  Dune::UMFPack<Dune::IBCRSMatrix<double>> solver_;
};

// UMFPACK can't handle float, so we create a template specialisation that just throws when constructed
template <>
class UMFPACKPrec<float> final : public Prec<float> {
public:
  UMFPACKPrec(const Dune::ParameterTree& /*config*/, std::shared_ptr<const Mat<float>> A)
      : Prec<float>(std::move(A))
  {
    DDM_CHECK(false, "UMFPACK does not support float");
  }

private:
  void do_apply(Vec<float>&, const Vec<float>&) override {}

  void do_update() override {}

  void do_info() const override {}
};
} // namespace ddm
