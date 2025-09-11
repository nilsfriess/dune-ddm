#pragma once

#include "logger.hh"
#include "spdlog/spdlog.h"

#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioner.hh>

#include <cstdint>
#include <memory>
#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Enumeration for different modes of combining preconditioners
 */
enum class ApplyMode : std::uint8_t {
  Additive,      /**< Sum all preconditioner results: P = P₁ + P₂ + ... + Pₙ */
  Multiplicative /**< Apply preconditioners sequentially with residual updates */
};

/**
 * @brief A class to combine multiple preconditioners to create a new one.
 *
 * This class allows to combine multiple preconditioners of type Dune::Preconditioner<X, Y>
 * into a new one (either additively or multiplicatively). Preconditioners are applied
 * in the order they are added using the add() method.
 *
 * In additive mode, the result is the sum of all preconditioner applications:
 * P = P₁ + P₂ + ... + Pₙ
 *
 * In multiplicative mode, preconditioners are applied sequentially with residual updates:
 * x₀ = P₁(d), x₁ = x₀ + P₂(d - A*x₀), x₂ = x₁ + P₃(d - A*x₁), ...
 * For multiplicative mode, an operator A must be provided using set_op().
 */
template <class X, class Y = X>
class CombinedPreconditioner : public Dune::Preconditioner<X, Y> {
public:
  /**
   * @brief Constructor that sets up the combined preconditioner from configuration.
   *
   * @param ptree Parameter tree containing configuration options
   * @param subtree_name Name of the subtree in ptree to read configuration from (default: "combined_preconditioner")
   *
   * Configuration options:
   * - mode: Either "additive" or "multiplicative" (default: "additive")
   *
   * @throws Dune::NotImplemented if an unknown mode is specified
   */
  explicit CombinedPreconditioner(const Dune::ParameterTree &ptree, const std::string &subtree_name = "combined_preconditioner")
  {
    const auto &subtree = subtree_name.size() == 0 ? ptree : ptree.sub(subtree_name);
    const auto mode_string = subtree.get("mode", "additive");

    if (mode_string == "additive") {
      mode = ApplyMode::Additive;
      spdlog::info("Setting up CombinedPreconditioner in 'additive' mode (currently has {} preconditioners)", precs.size());
    }
    else if (mode_string == "multiplicative") {
      mode = ApplyMode::Multiplicative;
      spdlog::info("Setting up CombinedPreconditioner in 'multiplicative' mode (currently has {} preconditioners)", precs.size());
    }
    else {
      DUNE_THROW(Dune::NotImplemented, "Unknown apply mode in CombinedPreconditioner, use either additive or multiplicative");
    }

    initLogEvents();
  }

  Dune::SolverCategory::Category category() const override
  {
    if (precs.size() == 0) {
      DUNE_THROW(Dune::Exception, "ERROR: No preconditioners added yet, add them using the `add` method");
    }

    return precs[0]->category();
  }

  /**
   * Add another preconditioner. The order in which the preconditioners are applied is
   * the same order in which they are added using this method.
   *
   * @param prec The new preconditioner to add
   * @throws Dune::Exception if the category of the new preconditioner doesn't match existing ones
   */
  void add(std::shared_ptr<Dune::Preconditioner<X, Y>> prec)
  {
    if (precs.size() > 0) {
      if (prec->category() != precs[0]->category()) {
        DUNE_THROW(Dune::Exception, "ERROR: Categories of the new preconditioner does not match");
      }
    }
    precs.push_back(prec);

    spdlog::info("Adding new preconditioner to CombinedPreconditioner, now has {}", precs.size());
  }

  /**
   * @brief Set the linear operator for multiplicative mode.
   *
   * This operator is required when using multiplicative mode to compute residuals
   * between preconditioner applications. The operator A should represent the same
   * linear system that the preconditioners are designed to solve.
   *
   * @param A The linear operator
   */
  void set_op(std::shared_ptr<Dune::LinearOperator<X, Y>> A) { this->A = A; }

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

    x = 0;
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
        DUNE_THROW(Dune::Exception, "ERROR: ApplyMode is multiplicative but operator A is not provided. Set with `set_op`");
      }

      Y dnext(d);
      for (std::size_t i = 1; i < precs.size(); ++i) {
        A->applyscaleadd(-1.0, x, dnext);

        X xnext(x.N());
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
  /** @brief Initialize logging events for performance monitoring */
  void initLogEvents() { apply_event = Logger::get().registerEvent("CombinedPreconditioner", "apply"); }

  /** @brief Vector of preconditioners to be combined */
  std::vector<std::shared_ptr<Dune::Preconditioner<X, Y>>> precs;

  /** @brief Mode for combining preconditioners (additive or multiplicative) */
  ApplyMode mode;

  /** @brief Linear operator used for residual computation in multiplicative mode */
  std::shared_ptr<Dune::LinearOperator<X, Y>> A;

  /** @brief Logging event for timing the apply method */
  Logger::Event *apply_event{};
};
