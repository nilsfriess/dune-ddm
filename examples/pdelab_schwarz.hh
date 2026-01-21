#pragma once

#include "generic_ddm_problem.hh"
#include "pdelab_helper.hh"

#include <algorithm>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/ddm/coarsespaces/coarse_spaces.hh>
#include <dune/ddm/combined_preconditioner.hh>
#include <dune/ddm/galerkin_preconditioner.hh>
#include <dune/ddm/logger.hh>
#include <dune/ddm/nonoverlapping_operator.hh>
#include <dune/ddm/overlap_extension.hh>
#include <dune/ddm/pdelab_helper.hh>
#include <dune/ddm/pou.hh>
#include <dune/ddm/schwarz.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/pdelab/backend/interface.hh>
#include <random>

#if DUNE_DDM_HAVE_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif

template <class X, class Y = X>
class TwoLevelSchwarzPreconditioner : public Dune::Preconditioner<X, Y> {
public:
  using NativeVec = X;
  using NativeMat = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  using Communication = Dune::OwnerOverlapCopyCommunication<std::size_t, int>;
  using FineLevel = SchwarzPreconditioner<NativeMat, NativeVec, Communication>;
  using CoarseLevel = GalerkinPreconditioner<NativeVec, Communication>;

  template <class GridView, class Traits>
  TwoLevelSchwarzPreconditioner(GenericDDMProblem<GridView, Traits>& problem, const Dune::ParameterTree& ptree, const Dune::MPIHelper& helper)
  {
    using Dune::PDELab::Backend::native;

    auto* setup_event = Logger::get().registerEvent("TwoLevelSchwarz", "Setup");
    Logger::ScopedLog sl(setup_event);

    // Get overlap from configuration
    const int overlap = ptree.get("overlap", 1);

    // Create non-overlapping communication
    logger::debug("Creating non-overlapping communication");
    novlp_comm_ = make_communication(*problem.getGFS());

    // Create overlapping communication
    logger::debug("Creating overlapping communication with overlap={}", overlap);
    auto [ovlp_comm, boundary_mask] = make_overlapping_communication(*novlp_comm_, native(problem.getA()), overlap);
    this->ovlp_comm_ = ovlp_comm;

    // Determine coarse space type and assemble appropriate matrices
    const auto& coarsespace_subtree = ptree.sub("coarsespace");
    auto coarsespace = coarsespace_subtree.get("type", "geneo");

    logger::debug("Setting up coarse space of type '{}'", coarsespace);

    if (coarsespace == "geneo" || coarsespace == "constraint_geneo") problem.assemble_overlapping_matrices(*ovlp_comm_, NeumannRegion::All, NeumannRegion::Overlap, overlap, true, novlp_comm_.get());
    else if (coarsespace == "msgfem") problem.assemble_overlapping_matrices(*ovlp_comm_, NeumannRegion::All, NeumannRegion::All, overlap, true, novlp_comm_.get());
    else if (coarsespace == "pou" || coarsespace == "harmonic_extension" || coarsespace == "algebraic_geneo" || coarsespace == "algebraic_msgfem" || coarsespace == "svd" ||
             coarsespace == "msgfem_euclid" || coarsespace == "none")
      problem.assemble_dirichlet_matrix_only(*ovlp_comm_, novlp_comm_.get());
    else if (coarsespace == "geneo_ring") problem.assemble_overlapping_matrices(*ovlp_comm_, NeumannRegion::ExtendedOverlap, NeumannRegion::ExtendedOverlap, overlap, false, novlp_comm_.get());
    else if (coarsespace == "msgfem_ring") problem.assemble_overlapping_matrices(*ovlp_comm_, NeumannRegion::Overlap, NeumannRegion::Overlap, overlap, false, novlp_comm_.get());
    else DUNE_THROW(Dune::NotImplemented, "Unknown coarse space type: " + coarsespace);

    // Get assembled matrices
    auto A_dir = problem.get_dirichlet_matrix();
    auto A_neu = problem.get_first_neumann_matrix();
    auto B_neu = problem.get_second_neumann_matrix();

    // Create partition of unity
    logger::debug("Creating partition of unity");
    pou_ = std::make_shared<PartitionOfUnity>(*A_dir, *ovlp_comm_, ptree, overlap);

    // Set up taskflow for coarse space construction
    tf::Taskflow taskflow("Coarse space setup");

    // Create coarse space
    std::unique_ptr<CoarseSpaceBuilder<>> coarse_space = nullptr;
    std::shared_ptr<CoarseLevel> coarse = nullptr;

    const auto zero_at_dirichlet = [&](auto&& x) {
      for (std::size_t i = 0; i < x.size(); ++i)
        if (problem.get_overlapping_dirichlet_mask()[i] > 0) x[i] = 0;
    };

    std::string coarse_space_ptree_prefix = "coarse_space";

    if (coarsespace == "geneo") { coarse_space = std::make_unique<GenEOCoarseSpace<NativeMat>>(A_neu, B_neu, pou_, ptree, taskflow, coarse_space_ptree_prefix); }
    else if (coarsespace == "constraint_geneo") {
      coarse_space = std::make_unique<ConstraintGenEOCoarseSpace<NativeMat, std::remove_reference_t<decltype(boundary_mask)>>>(A_dir, A_neu, B_neu, pou_, boundary_mask, ptree, taskflow,
                                                                                                                               coarse_space_ptree_prefix);
    }
    else if (coarsespace == "geneo_ring") {
      coarse_space = std::make_unique<GenEORingCoarseSpace<NativeMat>>(A_dir, A_neu, pou_, problem.get_neumann_region_to_subdomain(), ptree, taskflow, coarse_space_ptree_prefix);
    }
    else if (coarsespace == "msgfem") {
      coarse_space = std::make_unique<MsGFEMCoarseSpace<NativeMat, std::remove_reference_t<decltype(problem.get_overlapping_dirichlet_mask())>, std::remove_reference_t<decltype(boundary_mask)>>>(
          A_neu, A_dir, pou_, problem.get_overlapping_dirichlet_mask(), boundary_mask, ptree, taskflow, coarse_space_ptree_prefix);
    }
    else if (coarsespace == "msgfem_ring") {
      coarse_space = std::make_unique<MsGFEMRingCoarseSpace<NativeMat, std::remove_reference_t<decltype(problem.get_overlapping_dirichlet_mask())>, std::remove_reference_t<decltype(boundary_mask)>>>(
          A_dir, A_neu, overlap, pou_, problem.get_overlapping_dirichlet_mask(), boundary_mask, problem.get_neumann_region_to_subdomain(), ptree, taskflow, coarse_space_ptree_prefix);
    }
    else if (coarsespace == "pou") {
      coarse_space = std::make_unique<POUCoarseSpace<>>(pou_, taskflow);
    }
    else if (coarsespace == "harmonic_extension") {
      const auto& subtree = ptree.sub(coarse_space_ptree_prefix);
      int n_basis_vectors = subtree.get("n_basis_vectors", 8);

      std::size_t n_boundary = std::count_if(boundary_mask.begin(), boundary_mask.end(), [](const auto& x) { return x != 0; });

      std::mt19937 rng(ptree.get("seed", 1));
      std::normal_distribution<double> dist;
      auto basis_vectors = std::make_shared<std::vector<NativeVec>>(n_boundary, NativeVec(n_boundary));
      std::for_each(basis_vectors->begin(), basis_vectors->end(), [&](auto& x) { std::generate(x.begin(), x.end(), [&]() { return dist(rng); }); });

      coarse_space = std::make_unique<HarmonicExtensionCoarseSpace<>>(A_dir, pou_, basis_vectors, boundary_mask, taskflow);
    }
    else if (coarsespace == "svd") {
      coarse_space = std::make_unique<SVDCoarseSpace<>>(A_dir, pou_, boundary_mask, problem.get_overlapping_dirichlet_mask(), ptree, taskflow, coarse_space_ptree_prefix);
    }
    else if (coarsespace == "msgfem_euclid") {
      auto I = std::make_shared<NativeMat>(A_dir->N(), A_dir->N(), 1, 0.1, NativeMat::BuildMode::implicit);
      for (std::size_t i = 0; i < I->N(); ++i) I->entry(i, i) = 1.;
      I->compress();

      coarse_space = std::make_unique<MsGFEMCoarseSpace<NativeMat, std::remove_reference_t<decltype(problem.get_overlapping_dirichlet_mask())>, std::remove_reference_t<decltype(boundary_mask)>>>(
          I, A_dir, pou_, problem.get_overlapping_dirichlet_mask(), boundary_mask, ptree, taskflow, coarse_space_ptree_prefix);
    }
    else if (coarsespace == "none") {
      logger::debug("No coarse space selected");
    }
    else {
      DUNE_THROW(Dune::NotImplemented, "Unknown coarse space type: " + coarsespace);
    }

    // Execute taskflow to build coarse space
    if (coarse_space) {
      logger::debug("Building coarse space basis");
      tf::Task prec_setup_task = taskflow.emplace([&]() {
        basis_ = coarse_space->get_basis();
        std::ranges::for_each(basis_, zero_at_dirichlet);
        coarse = std::make_shared<CoarseLevel>(*A_dir, basis_, ovlp_comm_, ptree, "coarse_solver");
      });

      prec_setup_task.name("Build coarse preconditioner");
      prec_setup_task.succeed(coarse_space->get_setup_task());

      tf::Executor executor(ptree.get("taskflow_executor_threads", 1));
      executor.run(taskflow).get();
    }

    helper.getCommunication().barrier();

    // Create fine level Schwarz preconditioner
    logger::debug("Setting up fine level Schwarz preconditioner");
    // Modify the A_dir matrix to eliminate subdomain boundary dofs
    if (ptree.get("modify_subdomain_matrix", false)) eliminate_dirichlet(*A_dir, boundary_mask);
    auto schwarz = std::make_shared<FineLevel>(A_dir, ovlp_comm_, pou_, ptree);

    // Create non-overlapping operator (needed for multiplicative coarse space correction)
    logger::debug("Creating non-overlapping operator");
    using Op = NonOverlappingOperator<NativeMat, NativeVec, NativeVec, Communication>;
    novlp_op_ = std::make_shared<Op>(problem.getA().storage(), novlp_comm_);

    // Combine fine and coarse level preconditioners
    logger::debug("Setting up combined preconditioner");
    combined_prec_ = std::make_shared<CombinedPreconditioner<NativeVec>>(ptree);
    combined_prec_->set_op(novlp_op_);
    combined_prec_->add(schwarz);
    if (coarse) combined_prec_->add(coarse);

    logger::debug("TwoLevelSchwarzPreconditioner setup complete");
  }

  Dune::SolverCategory::Category category() const override { return combined_prec_->category(); }

  void pre(X& x, Y& b) override { combined_prec_->pre(x, b); }

  void apply(X& v, const Y& d) override { combined_prec_->apply(v, d); }

  void post(X& x) override { combined_prec_->post(x); }

  std::shared_ptr<Communication> getNonOverlappingCommunication() const { return novlp_comm_; }
  std::shared_ptr<Communication> getOverlappingCommunication() const { return ovlp_comm_; }

  using NonOverlappingOp = NonOverlappingOperator<NativeMat, NativeVec, NativeVec, Communication>;
  std::shared_ptr<NonOverlappingOp> getNonOverlappingOperator() const { return novlp_op_; }

  const std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>>& get_basis() const { return basis_; }

private:
  std::shared_ptr<Communication> novlp_comm_;
  std::shared_ptr<Communication> ovlp_comm_;
  std::shared_ptr<PartitionOfUnity> pou_;
  std::shared_ptr<CombinedPreconditioner<NativeVec>> combined_prec_;
  std::shared_ptr<NonOverlappingOp> novlp_op_;
  std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> basis_;
};
