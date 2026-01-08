#pragma once

/** @file schwarz.hh
    @brief Implementation of Schwarz domain decomposition preconditioners.

    This file provides the SchwarzPreconditioner class that implements both standard
    and restricted additive Schwarz methods for domain decomposition preconditioning.
*/

#include "helpers.hh"
#include "logger.hh"
#include "pou.hh"
#include "strumpack.hh"

#include <cstdint>
#include <dune/common/exceptions.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/cholmod.hh>
#include <dune/istl/io.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>
#include <memory>
#include <mpi.h>

#if DUNE_DDM_HAVE_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif

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
 * @tparam Communication A communication object, e.g. ISTL's OwnerOverlapCopyCommunication
 */
template <class Mat, class Vec, class Communication>
class SchwarzPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  using Op = Dune::MatrixAdapter<Mat, Vec, Vec>;
  using Solver = Dune::InverseOperator<Vec, Vec>;

public:
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
  SchwarzPreconditioner(std::shared_ptr<Mat> Aovlp, std::shared_ptr<Communication> comm, std::shared_ptr<PartitionOfUnity> pou, const Dune::ParameterTree& ptree,
                        const std::string& subtree_name = "schwarz", const std::string& solver_subtree_name = "subdomain_solver")
      : Aovlp(std::move(Aovlp))
      , comm(std::move(comm))
      , pou(std::move(pou))
  {
    const auto& subtree = ptree.sub(subtree_name);
    auto type_string = subtree.get("type", "restricted");
    if (type_string == "restricted") type = SchwarzType::Restricted;
    else if (type_string == "standard") type = SchwarzType::Standard;
    else DUNE_THROW(Dune::NotImplemented, "Unknown Schwarz type '" + type_string + "'");

    Dune::initSolverFactories<Op>();
    auto op = std::make_shared<Op>(this->Aovlp);
    // Since the error message that Dune gives us when there is no 'type' key in the solver_subtree
    // is useless, we check ourselves first and tell the user what they need to do.
    const auto& solver_subtree = subtree.sub(solver_subtree_name);
    if (not solver_subtree.hasKey("type"))
      DUNE_THROW(Dune::Exception, "You must specify the solver in the subtree " << get_parameter_tree_prefix(ptree) << subtree_name << "." << solver_subtree_name << " using the key 'type'");
    solver = Dune::getSolverFromFactory(op, solver_subtree);
    init();
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void pre(Vec&, Vec&) override {}
  void post(Vec&) override {}

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
  void apply(Vec& x, const Vec& d) override
  {
    Logger::ScopedLog sl(apply_event);

    // 1. Copy local values from non-overlapping to overlapping defect
    Logger::get().startEvent(get_defect_event);
    *d_ovlp = 0;
    for (std::size_t i = 0; i < d.size(); ++i) (*d_ovlp)[i] = d[i];

    // 2. Get remaining values from other ranks
    comm->copyOwnerToAll(*d_ovlp, *d_ovlp);

    Logger::get().endEvent(get_defect_event);

    // 3. Solve using the overlapping subdomain matrix
    Logger::get().startEvent(subdomain_solve_event);
    Dune::InverseOperatorResult res;
    *x_ovlp = 0.0;
    solver->apply(*x_ovlp, *d_ovlp, res);
    Logger::get().endEvent(subdomain_solve_event);

    // 4. Make the solution consistent according to the type of the Schwarz method
    Logger::get().startEvent(add_solution_event);
    if (type == SchwarzType::Standard) { comm->addOwnerCopyToOwnerCopy(*x_ovlp, *x_ovlp); }
    else if (type == SchwarzType::Restricted) {
      if (pou)
        for (std::size_t i = 0; i < pou->size(); ++i) (*x_ovlp)[i] *= (*pou)[i];
      comm->addOwnerCopyToOwnerCopy(*x_ovlp, *x_ovlp);
    }

    // 4. Restrict the solution to the non-overlapping subdomain
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = (*x_ovlp)[i];

    Logger::get().endEvent(add_solution_event);
  }

  /**
   * @brief Get reference to the local subdomain solver.
   * @return Reference to the solver instance
   */
  Solver& getSolver() { return *solver; }

#if DUNE_DDM_HAVE_TASKFLOW
  /**
   * @brief Return the setup task.
   * @return Reference to the setup task
   */
  tf::Task& get_setup_task() { return setup_task; }
#endif

  std::shared_ptr<Communication> novlp_comm;

private:
  /**
   * @brief Initialize the preconditioner.
   *
   * Sets up communication interfaces, creates solver instance (if not delayed),
   * and allocates working vectors.
   */
  void init()
  {
    logger::debug("Setting up Schwarz preconditioner in {} mode", type == SchwarzType::Standard ? "standard" : "restricted");

    apply_event = Logger::get().registerOrGetEvent("Schwarz", "apply");
    subdomain_solve_event = Logger::get().registerOrGetEvent("Schwarz", "local solve");
    get_defect_event = Logger::get().registerOrGetEvent("Schwarz", "get defect");
    add_solution_event = Logger::get().registerOrGetEvent("Schwarz", "add solution");
    auto* init_event = Logger::get().registerOrGetEvent("Schwarz", "init");

    Logger::ScopedLog sl(init_event);

    // Validate that remote indices match the overlapping matrix size
    const auto remote_indices_size = comm->indexSet().size();
    const auto matrix_size = Aovlp->N();
    if (remote_indices_size != matrix_size)
      DUNE_THROW(Dune::InvalidStateException, "Remote indices size (" << remote_indices_size << ") does not match overlapping matrix size (" << matrix_size << ").");

    // Validate that partition of unity has the correct size
    if (pou && pou->size() != matrix_size) DUNE_THROW(Dune::InvalidStateException, "Partition of unity size (" << pou->size() << ") does not match overlapping matrix size (" << matrix_size << ").");

    d_ovlp = std::make_unique<Vec>(Aovlp->N());
    x_ovlp = std::make_unique<Vec>(Aovlp->N());
  }

  std::shared_ptr<Mat> Aovlp; ///< Overlapping subdomain matrix
  std::shared_ptr<Communication> comm;

  std::shared_ptr<Solver> solver; ///< Local subdomain solver
  std::unique_ptr<Vec> d_ovlp;    ///< Defect on overlapping index set
  std::unique_ptr<Vec> x_ovlp;    ///< Solution on overlapping index set

  std::shared_ptr<PartitionOfUnity> pou{nullptr}; ///< Partition of unity (might be null)

  SchwarzType type; ///< Type of Schwarz method (standard or restricted)

#if DUNE_DDM_HAVE_TASKFLOW
  // Task-related
  tf::Task setup_task;
#endif

  // Performance monitoring events
  Logger::Event* apply_event{nullptr};           ///< Event for timing the apply method
  Logger::Event* subdomain_solve_event{nullptr}; ///< Event for timing local solves
  Logger::Event* get_defect_event{nullptr};      ///< Event for timing defect communication
  Logger::Event* add_solution_event{nullptr};    ///< Event for timing solution communication
};
