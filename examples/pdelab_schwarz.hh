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
    auto domain = coarsespace_subtree.get("domain", "full");
    auto constraint = coarsespace_subtree.get("constraint", "none");

    logger::debug("Setting up coarse space with domain='{}', constraint='{}'", domain, constraint);

    // Determine which Neumann regions to assemble based on domain x constraint
    if (domain == "full" and constraint == "none") problem.assemble_overlapping_matrices(*ovlp_comm_, NeumannRegion::All, NeumannRegion::Overlap, overlap, true, novlp_comm_.get());
    else if (domain == "full" and constraint == "harmonic") problem.assemble_overlapping_matrices(*ovlp_comm_, NeumannRegion::All, NeumannRegion::All, overlap, true, novlp_comm_.get());
    else if (domain == "ring" and constraint == "none")
      problem.assemble_overlapping_matrices(*ovlp_comm_, NeumannRegion::ExtendedOverlap, NeumannRegion::ExtendedOverlap, overlap, false, novlp_comm_.get());
    else if (domain == "ring" and constraint == "harmonic") problem.assemble_overlapping_matrices(*ovlp_comm_, NeumannRegion::Overlap, NeumannRegion::Overlap, overlap, false, novlp_comm_.get());
    else DUNE_THROW(Dune::NotImplemented, "Unknown coarse space configuration: domain=" + domain + ", constraint=" + constraint);

    // Get assembled matrices
    auto A_sub = problem.get_subdomain_matrix();
    auto A_neu = problem.get_first_neumann_matrix();
    auto B_neu = problem.get_second_neumann_matrix();

    // Ensure boundary_mask is not set at Dirichlet boundary nodes
    const auto& dirichlet_mask = problem.get_overlapping_dirichlet_mask();
    for (std::size_t i = 0; i < boundary_mask.size(); ++i)
      if (dirichlet_mask[i]) boundary_mask[i] = 0;

    // Create partition of unity
    logger::debug("Creating partition of unity");
    pou_ = std::make_shared<PartitionOfUnity>(*A_sub, *ovlp_comm_, ptree, overlap);

    // Create coarse space
    std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> coarse_basis;
    std::shared_ptr<CoarseLevel> coarse = nullptr;

    const auto zero_at_dirichlet = [&](auto&& x) {
      for (std::size_t i = 0; i < x.size(); ++i)
        if (problem.get_overlapping_dirichlet_mask()[i] > 0) x[i] = 0;
    };

    // Create an A_dir matrix that corresponds to A_sub with subdomain boundary dofs eliminated
    auto A_dir = std::make_shared<NativeMat>(*A_sub);
    eliminate_dirichlet(*A_dir, boundary_mask, false); // false => don't eliminate symetrically

    // Create fine level Schwarz preconditioner
    logger::debug("Setting up fine level Schwarz preconditioner");
    auto schwarz = std::make_shared<FineLevel>(A_dir, ovlp_comm_, pou_, ptree);

    std::string coarse_space_ptree_prefix = "coarsespace";

    if (domain == "full" and constraint == "none") { coarse_basis = build_geneo_coarse_space(*A_neu, *B_neu, *pou_, ptree, coarse_space_ptree_prefix); }
    else if (domain == "ring" and constraint == "none") {
      coarse_basis =
          build_geneo_ring_coarse_space(*A_neu, *B_neu, problem.get_neumann_region_to_subdomain(), *pou_, *A_sub, ptree, coarse_space_ptree_prefix, schwarz->get_solver().get(), boundary_mask);
    }
    else if (domain == "full" and constraint == "harmonic") {
      coarse_basis = build_msgfem_coarse_space(*A_neu, *pou_, boundary_mask, ptree, coarse_space_ptree_prefix, A_sub.get(), dirichlet_mask, schwarz->get_solver().get());
    }
    else if (domain == "ring" and constraint == "harmonic") {
      coarse_basis = build_msgfem_ring_coarse_space(*A_neu, problem.get_neumann_region_to_subdomain(), *pou_, *A_sub, dirichlet_mask, boundary_mask, ptree, coarse_space_ptree_prefix,
                                                    schwarz->get_solver().get());
    }
    else DUNE_THROW(Dune::NotImplemented, "Unknown coarse space configuration: domain=" + domain + ", constraint=" + constraint);

    if (!coarse_basis.empty()) {
      basis_ = std::move(coarse_basis);
      std::ranges::for_each(basis_, zero_at_dirichlet);
      coarse = std::make_shared<CoarseLevel>(*A_sub, basis_, ovlp_comm_, ptree, "coarse_solver");
    }

    helper.getCommunication().barrier();

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

  std::shared_ptr<PartitionOfUnity> get_pou() { return pou_; }

private:
  std::shared_ptr<Communication> novlp_comm_;
  std::shared_ptr<Communication> ovlp_comm_;
  std::shared_ptr<PartitionOfUnity> pou_;
  std::shared_ptr<CombinedPreconditioner<NativeVec>> combined_prec_;
  std::shared_ptr<NonOverlappingOp> novlp_op_;
  std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> basis_;
};
