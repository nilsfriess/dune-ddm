#pragma once

/** @file schwarz.hh
    @brief Implementation of Schwarz domain decomposition preconditioners.

    This file provides the SchwarzPreconditioner class that implements both standard
    and restricted additive Schwarz methods for domain decomposition preconditioning.
*/

#include "backend/host/backend.hh"
#include "communication.hh"
#include "helpers.hh"
#include "logger.hh"
#include "pou.hh"

#include <cstddef>
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
template <class Mat, class Vec>
class SchwarzPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  using Op = Dune::MatrixAdapter<Mat, Vec, Vec>;
  using Solver = Dune::InverseOperator<Vec, Vec>;
  using Backend = ddm::backend::backend_of_t<Vec>;

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
  template <class OwnerOverlapCopyCommunication>
  SchwarzPreconditioner(std::shared_ptr<Mat> Aovlp, const OwnerOverlapCopyCommunication& oocc, std::shared_ptr<PartitionOfUnity> pou, const Dune::ParameterTree& ptree,
                        const std::string& subtree_name = "schwarz", const std::string& solver_subtree_name = "subdomain_solver")
      : Aovlp(std::move(Aovlp))
      , comm(ddm::make_communication_from_dune(oocc))
      , pou(std::move(pou))
  {
    auto* init_event = Logger::get().registerOrGetEvent("Schwarz", "init");
    Logger::ScopedLog sl(init_event);

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

    // We also handle one special case ourselves, namely a solver named umfpack_metis
    // which we define as UMFPack with METIS reordering
    if (solver_subtree["type"] == "umfpack_metis") {
      solver = std::make_shared<Dune::UMFPack<Mat>>();
      auto umfpack_solver = std::dynamic_pointer_cast<Dune::UMFPack<Mat>>(solver);
      umfpack_solver->setOption(UMFPACK_ORDERING, UMFPACK_ORDERING_METIS);
      umfpack_solver->setOption(UMFPACK_IRSTEP, 0); // Disable iterative refinement for performance
      umfpack_solver->setMatrix(*this->Aovlp);
    }
    else solver = Dune::getSolverFromFactory(op, solver_subtree);
    init();
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::overlapping; }

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

    // 1. Copy local values from the incoming defect to the overlapping one
    Logger::get().startEvent(get_defect_event);
    Backend::copy_n(d, d.size(), *d_ovlp);

    // 2. Fetch the entries in the overlap region from the owner rank (by the general assumption of this module
    //    incoming defects are consistent, so it's sufficient to ask the owner for the value)
    if (d.size() < d_ovlp->size()) comm.broadcast(*d_ovlp); // comm->copyOwnerToAll(*d_ovlp, *d_ovlp);

    Logger::get().endEvent(get_defect_event);

    // 3. Solve using the overlapping subdomain matrix
    Logger::get().startEvent(subdomain_solve_event);
    Dune::InverseOperatorResult res;
    *x_ovlp = 0.0;
    solver->apply(*x_ovlp, *d_ovlp, res);
    Logger::get().endEvent(subdomain_solve_event);

    // 4. Make the solution consistent according to the type of the Schwarz method
    Logger::get().startEvent(add_solution_event);
    if (type == SchwarzType::Standard) { comm.reduce(*x_ovlp); }
    else if (type == SchwarzType::Restricted) {
      if (pou)
        for (std::size_t i = 0; i < pou->size(); ++i) (*x_ovlp)[i] *= (*pou)[i];
      comm.reduce(*x_ovlp);
    }

    // 4. Restrict the solution to the non-overlapping subdomain
    Backend::copy_n(*x_ovlp, x.size(), x);

    Logger::get().endEvent(add_solution_event);
  }

  // /**
  //  * @brief Get reference to the local subdomain solver.
  //  * @return Reference to the solver instance
  //  */
  // std::shared_ptr<Solver> get_solver() { return solver; }

private:
  /** @brief Initialize the preconditioner.
   *
   */
  void init()
  {
    logger::debug("Setting up Schwarz preconditioner in {} mode", type == SchwarzType::Standard ? "standard" : "restricted");

    apply_event = Logger::get().registerOrGetEvent("Schwarz", "apply");
    subdomain_solve_event = Logger::get().registerOrGetEvent("Schwarz", "local solve");
    get_defect_event = Logger::get().registerOrGetEvent("Schwarz", "get defect");
    add_solution_event = Logger::get().registerOrGetEvent("Schwarz", "add solution");

    d_ovlp = std::make_unique<Vec>(Aovlp->N());
    x_ovlp = std::make_unique<Vec>(Aovlp->N());
  }

  std::shared_ptr<Mat> Aovlp; ///< Overlapping subdomain matrix
  ddm::Communication comm;

  std::shared_ptr<Solver> solver; ///< Local subdomain solver
  std::unique_ptr<Vec> d_ovlp;    ///< Defect on overlapping index set
  std::unique_ptr<Vec> x_ovlp;    ///< Solution on overlapping index set

  std::shared_ptr<PartitionOfUnity> pou{nullptr}; ///< Partition of unity (might be null)

  SchwarzType type; ///< Type of Schwarz method (standard or restricted)

  // Performance monitoring events
  Logger::Event* apply_event{nullptr};           ///< Event for timing the apply method
  Logger::Event* subdomain_solve_event{nullptr}; ///< Event for timing local solves
  Logger::Event* get_defect_event{nullptr};      ///< Event for timing defect communication
  Logger::Event* add_solution_event{nullptr};    ///< Event for timing solution communication
};
