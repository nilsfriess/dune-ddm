#pragma once

/** @file schwarz.hh
    @brief Implementation of Schwarz domain decomposition preconditioners.

    This file provides the SchwarzPreconditioner class that implements both standard
    and restricted additive Schwarz methods for domain decomposition preconditioning.
*/

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/cholmod.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>

#include <cstdint>
#include <memory>
#include <mpi.h>

#include "helpers.hh"
#include "logger.hh"
#include "pou.hh"
#include "spdlog/spdlog.h"
#include "strumpack.hh"

#include <taskflow/taskflow.hpp>

/**
 * @brief Type of Schwarz domain decomposition method.
 */
enum class SchwarzType : std::uint8_t {
  Standard,  ///< Standard additive Schwarz method
  Restricted ///< Restricted additive Schwarz method (with partition of unity)
};

/**
 * @brief Schwarz domain decomposition preconditioner.
 *
 * This class implements both standard and restricted additive Schwarz methods
 * for preconditioning linear systems in domain decomposition contexts.
 *
 * The preconditioner operates on overlapping subdomains and uses local solvers
 * to compute corrections. The restricted variant uses a partition of unity
 * to ensure proper scaling at subdomain boundaries.
 *
 * @tparam Vec Vector type for the linear system
 * @tparam Mat Matrix type for the linear system
 * @tparam ExtendedRemoteIndices Index set type for communication
 */
template <class Vec, class Mat, class ExtendedRemoteIndices>
class SchwarzPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  using Op = Dune::MatrixAdapter<Mat, Vec, Vec>;
  using Solver = Dune::InverseOperator<Vec, Vec>;

  AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  AttributeSet ownerAttribute{Attribute::owner};
  AttributeSet copyAttribute{Attribute::copy};

  struct AddGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] += v; }
  };

  struct CopyGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] = v; }
  };

public:
  // /**
  //  * @brief Construct Schwarz preconditioner with explicit parameters.
  //  *
  //  * @param Aovlp Shared pointer to the overlapping subdomain matrix
  //  * @param ext_indices Extended remote indices for communication setup
  //  * @param pou Shared pointer to partition of unity (can be nullptr for standard Schwarz)
  //  * @param type Type of Schwarz method (Standard or Restricted)
  //  * @param factorise_at_first_iteration If true, delay matrix factorization until first apply
  //  */
  // SchwarzPreconditioner(std::shared_ptr<Mat> Aovlp, const ExtendedRemoteIndices &ext_indices, std::shared_ptr<PartitionOfUnity> pou, SchwarzType type = SchwarzType::Restricted,
  //                       bool factorise_at_first_iteration = false)
  //     : Aovlp(std::move(Aovlp)), ext_indices(ext_indices), type(type), pou(std::move(pou)), factorise_at_first_iteration(factorise_at_first_iteration)
  // {
  //   init();
  // }

  /**
   * @brief Construct Schwarz preconditioner from parameter tree.
   *
   * Reads configuration from a parameter tree, supporting the following options:
   * - factorise_at_first_iteration: boolean, default false
   * - type: "standard" or "restricted", default "restricted"
   *
   * @param Aovlp Shared pointer to the overlapping subdomain matrix
   * @param ext_indices Extended remote indices for communication setup
   * @param pou Shared pointer to partition of unity
   * @param taskflow Taskflow instance to execute the Schwarz setup asynchronously
   * @param ptree Parameter tree containing configuration
   * @param subtree_name Name of the subtree containing Schwarz parameters
   */
  SchwarzPreconditioner(std::shared_ptr<Mat> Aovlp, const ExtendedRemoteIndices &ext_indices, std::shared_ptr<PartitionOfUnity> pou, const Dune::ParameterTree &ptree, tf::Taskflow &taskflow,
                        const std::string &subtree_name = "schwarz")
      : Aovlp(std::move(Aovlp)), pou(std::move(pou)), ext_indices(ext_indices)
  {
    const auto &subtree = ptree.sub(subtree_name);
    factorise_at_first_iteration = subtree.get("factorise_at_first_iteration", false);
    auto type_string = subtree.get("type", "restricted");
    if (type_string == "restricted") {
      type = SchwarzType::Restricted;
    }
    else if (type_string == "standard") {
      type = SchwarzType::Standard;
    }
    else {
      DUNE_THROW(Dune::NotImplemented, "Unknown Schwarz type '" + type_string + "'");
    }

    setup_task = taskflow.emplace([&]() { init(); }).name("Init overlapping Schwarz preconditioner");
  }

  /**
   * @brief Construct Schwarz preconditioner from parameter tree.
   *
   * Reads configuration from a parameter tree, supporting the following options:
   * - factorise_at_first_iteration: boolean, default false
   * - type: "standard" or "restricted", default "restricted"
   *
   * @param Aovlp Shared pointer to the overlapping subdomain matrix
   * @param ext_indices Extended remote indices for communication setup
   * @param pou Shared pointer to partition of unity
   * @param ptree Parameter tree containing configuration
   * @param subtree_name Name of the subtree containing Schwarz parameters
   */
  SchwarzPreconditioner(std::shared_ptr<Mat> Aovlp, const ExtendedRemoteIndices &ext_indices, std::shared_ptr<PartitionOfUnity> pou, const Dune::ParameterTree &ptree,
                        const std::string &subtree_name = "schwarz", const std::string &solver_subtree_name = "subdomain_solver")
      : Aovlp(std::move(Aovlp)), pou(std::move(pou)), ext_indices(ext_indices), ptree(ptree), solver_subtree_name(solver_subtree_name)
  {
    const auto &subtree = ptree.sub(subtree_name);
    factorise_at_first_iteration = subtree.get("factorise_at_first_iteration", false);
    auto type_string = subtree.get("type", "restricted");
    if (type_string == "restricted") {
      type = SchwarzType::Restricted;
    }
    else if (type_string == "standard") {
      type = SchwarzType::Standard;
    }
    else {
      DUNE_THROW(Dune::NotImplemented, "Unknown Schwarz type '" + type_string + "'");
    }

    Dune::initSolverFactories<Op>();
    auto op = std::make_shared<Op>(this->Aovlp);
    solver = Dune::getSolverFromFactory(op, subtree.sub(solver_subtree_name));
    init();
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void pre(Vec &, Vec &) override {}
  void post(Vec &) override {}

  /**
   * @brief Apply the Schwarz preconditioner.
   *
   * This method implements the core Schwarz algorithm:
   * 1. Extend the defect vector to the overlapping subdomain
   * 2. Solve the local subdomain problem
   * 3. Apply communication pattern based on Schwarz type:
   *    - Standard: Simple addition across subdomains
   *    - Restricted: Multiply by partition of unity before addition
   * 4. Restrict solution back to non-overlapping subdomain
   *
   * @param x Output: preconditioned solution vector
   * @param d Input: defect/residual vector to be preconditioned
   */
  void apply(Vec &x, const Vec &d) override
  {
    Logger::ScopedLog sl(apply_event);

    if (!solver) {
      Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Schwarz", "init")};
      auto op = std::make_shared<Op>(Aovlp);
      solver = Dune::getSolverFromFactory(op, ptree.sub(solver_subtree_name));
    }

    // 1. Copy local values from non-overlapping to overlapping defect
    Logger::get().startEvent(get_defect_event);
    *d_ovlp = 0;
    for (std::size_t i = 0; i < d.size(); ++i) {
      (*d_ovlp)[i] = d[i];
    }

    // 2. Get remaining values from other ranks
    owner_copy_comm.forward<CopyGatherScatter>(*d_ovlp);
    Logger::get().endEvent(get_defect_event);

    // 3. Solve using the overlapping subdomain matrix
    Logger::get().startEvent(subdomain_solve_event);
    Dune::InverseOperatorResult res;
    *x_ovlp = 0.0;
    solver->apply(*x_ovlp, *d_ovlp, res);
    Logger::get().endEvent(subdomain_solve_event);

    // 4. Make the solution consistent according to the type of the Schwarz method
    Logger::get().startEvent(add_solution_event);
    if (type == SchwarzType::Standard) {
      all_all_comm.forward<AddGatherScatter>(*x_ovlp);
    }
    else if (type == SchwarzType::Restricted) {
      if (pou) {
        for (std::size_t i = 0; i < pou->size(); ++i) {
          (*x_ovlp)[i] *= (*pou)[i];
        }
      }
      all_all_comm.forward<AddGatherScatter>(*x_ovlp);
    }

    // 4. Restrict the solution to the non-overlapping subdomain
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = (*x_ovlp)[i];
    }
    Logger::get().endEvent(add_solution_event);
  }

  /**
   * @brief Get reference to the local subdomain solver.
   * @return Reference to the solver instance
   */
  Solver &getSolver() { return *solver; }

  /**
   * @brief Return the setup task.
   * @return Reference to the setup task
   */
  tf::Task &get_setup_task() { return setup_task; }

private:
  /**
   * @brief Initialize the preconditioner.
   *
   * Sets up communication interfaces, creates solver instance (if not delayed),
   * and allocates working vectors.
   */
  void init()
  {
    spdlog::info("Setting up Schwarz preconditioner in {} mode", type == SchwarzType::Standard ? "standard" : "restricted");

    apply_event = Logger::get().registerEvent("Schwarz", "apply");
    subdomain_solve_event = Logger::get().registerEvent("Schwarz", "local solve");
    get_defect_event = Logger::get().registerEvent("Schwarz", "get defect");
    add_solution_event = Logger::get().registerEvent("Schwarz", "add solution");
    auto *init_event = Logger::get().registerEvent("Schwarz", "init");

    Logger::ScopedLog sl(init_event);

    all_all_interface.build(ext_indices.get_remote_indices(), allAttributes, allAttributes);
    all_all_comm.build<Vec>(all_all_interface);

    owner_copy_interface.build(ext_indices.get_remote_indices(), ownerAttribute, copyAttribute);
    owner_copy_comm.build<Vec>(owner_copy_interface);

    d_ovlp = std::make_unique<Vec>(Aovlp->N());
    x_ovlp = std::make_unique<Vec>(Aovlp->N());
  }

  std::shared_ptr<Mat> Aovlp; ///< Overlapping subdomain matrix

  std::shared_ptr<Solver> solver; ///< Local subdomain solver
  std::unique_ptr<Vec> d_ovlp;    ///< Defect on overlapping index set
  std::unique_ptr<Vec> x_ovlp;    ///< Solution on overlapping index set

  std::shared_ptr<PartitionOfUnity> pou{nullptr}; ///< Partition of unity (might be null)

  Dune::Interface all_all_interface;       ///< Interface for all-to-all communication
  Dune::BufferedCommunicator all_all_comm; ///< Communicator for all-to-all exchange

  Dune::Interface owner_copy_interface;       ///< Interface for owner-to-copy communication
  Dune::BufferedCommunicator owner_copy_comm; ///< Communicator for owner-to-copy exchange

  const ExtendedRemoteIndices &ext_indices; ///< Extended remote indices for communication setup

  SchwarzType type; ///< Type of Schwarz method (standard or restricted)

  const Dune::ParameterTree &ptree;
  const std::string &solver_subtree_name;

  bool factorise_at_first_iteration; ///< Whether to delay factorization until first apply

  // Task-related
  tf::Task setup_task;

  // Performance monitoring events
  Logger::Event *apply_event{nullptr};           ///< Event for timing the apply method
  Logger::Event *subdomain_solve_event{nullptr}; ///< Event for timing local solves
  Logger::Event *get_defect_event{nullptr};      ///< Event for timing defect communication
  Logger::Event *add_solution_event{nullptr};    ///< Event for timing solution communication
};
