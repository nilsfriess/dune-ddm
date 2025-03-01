#pragma once

#include "logger.hh"

#include <dune/common/exceptions.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioner.hh>

#include <cstdint>
#include <memory>
#include <mpi.h>
#include <vector>

enum class ApplyMode : std::uint8_t { Additive, Multiplicative };

template <class X, class Y = X>
class CombinedPreconditioner : public Dune::Preconditioner<X, Y> {
public:
  explicit CombinedPreconditioner(ApplyMode mode = ApplyMode::Additive) : mode(mode)
  {
    precs.reserve(2); // In many cases, we expect this to be used to combine two preconditioners (e.g. two level Schwarz)

    initLogEvents();
  }

  explicit CombinedPreconditioner(std::shared_ptr<Dune::LinearOperator<X, Y>> A) : mode(ApplyMode::Multiplicative), A(A) { initLogEvents(); }

  Dune::SolverCategory::Category category() const override
  {
    if (precs.size() == 0) {
      DUNE_THROW(Dune::Exception, "ERROR: No preconditioners added yet, add them using the `add` method");
    }

    return precs[0]->category();
  }

  void add(std::shared_ptr<Dune::Preconditioner<X, Y>> prec)
  {
    if (precs.size() > 0) {
      if (prec->category() != precs[0]->category()) {
        DUNE_THROW(Dune::Exception, "ERROR: Categories of the new preconditioner does not match");
      }
    }
    precs.push_back(prec);
  }

  void setMat(std::shared_ptr<Dune::LinearOperator<X, Y>> A) { this->A = A; }

  void pre(X &x, Y &y) override
  {
    for (auto &prec : precs) {
      prec->pre(x, y);
    }
  }

  void post(X &x) override
  {
    for (auto &prec : precs) {
      prec->post(x);
    }
  }

  void apply(X &x, const Y &d) override
  {
    Logger::ScopedLog se(apply_event);

    assert(precs.size() > 0 && "ERROR: No preconditioners to apply, add them using the `add` method"); // Error should be caught in the "category" method already

    precs[0]->apply(x, d);
    if (mode == ApplyMode::Additive) {
      X xnext(x.N());
      for (std::size_t i = 1; i < precs.size(); ++i) {
        xnext = 0;
        precs[i]->apply(xnext, d);

        x += xnext;
      }
    }
    else if (mode == ApplyMode::Multiplicative) {
      if (!A) {
        DUNE_THROW(Dune::Exception, "ERROR: ApplyMode is multiplicative but operator A is not provided. Set with `setMat`");
      }

      Y dnext(d);
      X xnext(x);
      for (std::size_t i = 1; i < precs.size(); ++i) {
        A->applyscaleadd(-1, xnext, dnext);

        xnext = 0;
        precs[i]->apply(xnext, dnext);

        x += xnext;
      }
    }
    else {
      __builtin_unreachable();
    }
  }

private:
  void initLogEvents() { apply_event = Logger::get().registerEvent("CombinedPreconditioner", "apply"); }

  std::vector<std::shared_ptr<Dune::Preconditioner<X, Y>>> precs;
  ApplyMode mode;

  std::shared_ptr<Dune::LinearOperator<X, Y>> A;

  Logger::Event *apply_event{};
};
