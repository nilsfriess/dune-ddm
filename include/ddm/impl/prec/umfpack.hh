#pragma once

#include "ddm/check.hh"
#include "ddm/impl/mat/istlmat.hh"
#include "ddm/logger.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/experimental/bcrsmatrix.hh>
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
      , factorize_event_(Logger::get().registerOrGetEvent("UMFPACK", "factorize"))
      , solve_event_(Logger::get().registerOrGetEvent("UMFPACK", "solve"))
  {
    // TODO(20261006-192425): It should be possible to mark a preconditioner or solver as "sequential" on a higher level
    DDM_CHECK(this->mat()->sequential(), "UMFPACK is a sequential solver");
    do_update();
  }

private:
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    Logger::ScopedLog sl{solve_event_};
    solve(UMFPACK_A, z, r);
  }

  void do_apply_transposed(Vec<T>& z, const Vec<T>& r) override
  {
    Logger::ScopedLog sl{solve_event_};
    solve(UMFPACK_At, z, r);
  }

  // Solves the system sys (UMFPACK_A or UMFPACK_At) with the factorization of Dune::UMFPack. Calls UMFPACK directly on
  // the data of the vectors, because Dune::UMFPack only solves with A and works on ISTL vectors only
  void solve(int sys, Vec<T>& z, const Vec<T>& r)
  {
    if (r.size() == 0) return;

    auto x = as_istl(z).data();
    const auto b = as_istl(r).data();

    double control[UMFPACK_CONTROL];
    double info[UMFPACK_INFO];
    umfpack_dl_defaults(control);
    control[UMFPACK_IRSTEP] = 0; // as in do_update()

    auto& M = solver_.getInternalMatrix();

    // These casts are necessary in the case of complex arithmetic; there we can pass the complex
    // data as a single double* array.
    const double* Ax = reinterpret_cast<double*>(M.getValues());
    double* Xx = reinterpret_cast<double*>(x.data());
    const double* Bx = reinterpret_cast<const double*>(b.data());

    if constexpr (std::is_same_v<T, double>) umfpack_dl_solve(sys, M.getColStart(), M.getRowIndex(), Ax, Xx, Bx, solver_.getFactorization(), control, info);
    else if constexpr (std::is_same_v<T, std::complex<double>>)
      umfpack_zl_solve(sys, M.getColStart(), M.getRowIndex(), Ax, nullptr, Xx, nullptr, Bx, nullptr, solver_.getFactorization(), control, info);
    else static_assert(false, "UMFPack supports only double and complex<double>");

    DDM_CHECK(info[UMFPACK_STATUS] >= UMFPACK_OK, "UMFPACK: solve failed with status {}", info[UMFPACK_STATUS]);
  }

  void do_update() override
  {
    Logger::ScopedLog sl{factorize_event_};
    solver_.setOption(UMFPACK_ORDERING, UMFPACK_ORDERING_METIS);
    solver_.setOption(UMFPACK_IRSTEP, 0);
    solver_.setMatrix(as_istl(this->mat()->local()).native());
  }

  void do_info() const override
  {
    logger::info("UMFPACK solver");
  }

  Logger::Event* factorize_event_;
  Logger::Event* solve_event_;
  Dune::UMFPack<Dune::Experimental::BCRSMatrix<T>> solver_;
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
