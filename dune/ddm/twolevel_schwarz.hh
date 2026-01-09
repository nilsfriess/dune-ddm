#pragma once

#include <cassert>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvercategory.hh>

#if HAVE_DUNE_PDELAB
#include "coarsespaces/coarse_spaces.hh"
#include "combined_preconditioner.hh"
#include "galerkin_preconditioner.hh"
#include "nonoverlapping_operator.hh"
#include "overlap_extension.hh"
#include "pdelab_helper.hh"
#include "schwarz.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/pdelab/backend/istl/novlpistlsolverbackend.hh>
#include <dune/pdelab/backend/istl/parallelhelper.hh>
#include <dune/pdelab/backend/solver.hh>
#include <dune/pdelab/gridfunctionspace/genericdatahandle.hh>
#include <dune/pdelab/gridfunctionspace/interpolate.hh>
#include <dune/pdelab/solver/newton.hh>
#include <memory>

// Wrapper for usage in PDELab's Newton solver
template <class Mat, class Vec>
class TwoLevelSchwarzSolver : public Dune::PDELab::LinearResultStorage {
  using NativeMat = Dune::PDELab::Backend::Native<Mat>;
  using NativeVec = Dune::PDELab::Backend::Native<Vec>;

  using Communication = Dune::OwnerOverlapCopyCommunication<std::size_t, int>;

  using FineLevel = SchwarzPreconditioner<NativeMat, NativeVec, Communication>;
  using CoarseLevel = GalerkinPreconditioner<NativeVec, Communication>;

public:
  template <class GFS, class CC>
  explicit TwoLevelSchwarzSolver(const GFS& gfs, const CC& cc, const Dune::ParameterTree& ptree, const std::string& subtree_name = "twolevelschwarz", bool matrix_is_additive = true)
      : novlp_comm(make_communication(gfs))
      , subtree(ptree.sub(subtree_name))
      , matrix_is_additive(matrix_is_additive)
  {
    using Dune::PDELab::Backend::Native;
    using Dune::PDELab::Backend::native;

    std::vector<Vec> template_vecs(3, gfs);
    Dune::PDELab::interpolate([](auto&&) { return 1; }, gfs, template_vecs[0]);
    Dune::PDELab::interpolate([](auto&& x) { return x[0]; }, gfs, template_vecs[1]);
    Dune::PDELab::interpolate([](auto&& x) { return x[1]; }, gfs, template_vecs[2]);
    std::for_each(template_vecs.begin(), template_vecs.end(), [&](auto&& v) { Dune::PDELab::set_constrained_dofs(cc, 0., v); });

    native_template_vecs.resize(template_vecs.size(), NativeVec(template_vecs[0].N()));
    std::transform(template_vecs.begin(), template_vecs.end(), native_template_vecs.begin(), [](auto&& v) { return native(v); });
  }

  void apply(Mat& A, Vec& z, Vec& r, typename Dune::template FieldTraits<typename Vec::ElementType>::real_type reduction)
  {
    using Dune::PDELab::Backend::Native;
    using Dune::PDELab::Backend::native;
    using Op = NonOverlappingOperator<NativeMat, NativeVec, NativeVec, Communication>;

    if (!matrix_is_additive) ::make_additive(A, *novlp_comm);

    // If it's the first time this function is called, create the overlapping matrix.
    // In the subsequent calls, we only update the matrix.
    if (!ovlp_comm) {
      Dune::initSolverFactories<Op>();

      int overlap = subtree.get("overlap", 1);
      ovlp_comm = make_overlapping_communication(*novlp_comm, native(A), overlap).first;

      // Create a communicator on the overlapping index set
      typename Communication::AllSet allset;
      interface_ext.build(ovlp_comm->remoteIndices(), allset, allset);
      varcomm = std::make_unique<Dune::VariableSizeCommunicator<>>(interface_ext);

      CreateMatrixDataHandle cmdh(native(A), ovlp_comm->indexSet());
      varcomm->forward(cmdh);
      A_ovlp = std::make_shared<NativeMat>(cmdh.getOverlappingMatrix());

      AddMatrixDataHandle amdh(native(A), *A_ovlp, ovlp_comm->indexSet());
      varcomm->forward(amdh);

      pou = std::make_shared<PartitionOfUnity>(*A_ovlp, *ovlp_comm, subtree.sub("pou"), overlap);

      // Extend the template vectors that were created in the constructor to the overlapping
      // subdomains.

      std::vector<NativeVec> extended_template_vecs(native_template_vecs.size(), NativeVec(A_ovlp->N()));
      for (std::size_t i = 0; i < native_template_vecs.size(); ++i) {
        extended_template_vecs[i] = 0;
        for (std::size_t j = 0; j < native_template_vecs[i].N(); ++j) extended_template_vecs[i][j] = native_template_vecs[i][j];
        ovlp_comm->copyOwnerToAll(extended_template_vecs[i], extended_template_vecs[i]);
      }
      native_template_vecs = std::move(extended_template_vecs);
    }
    else {
      // Update the overlapping matrix for subsequent calls
      *A_ovlp = 0;
      AddMatrixDataHandle amdh(native(A), *A_ovlp, ovlp_comm->indexSet());
      varcomm->forward(amdh);
    }

    // Set up the preconditioner
    POUCoarseSpace coarse_space(native_template_vecs, *pou);
    auto fine = std::make_shared<FineLevel>(A_ovlp, ovlp_comm, pou, subtree, "fine");
    fine->novlp_comm = novlp_comm;

    auto coarse = std::make_shared<CoarseLevel>(*A_ovlp, coarse_space.get_basis(), ovlp_comm, subtree, "coarse");

    auto prec = std::make_shared<CombinedPreconditioner<NativeVec>>(subtree, "");
    auto op = std::make_shared<Op>(A.storage(), novlp_comm);
    prec->set_op(op);
    prec->add(fine);
    prec->add(coarse);

    // Set up the solver
    int rank = ovlp_comm->communicator().rank();
    Dune::ParameterTree solver_subtree;
    if (subtree.hasSub("solver")) { solver_subtree = subtree.sub("solver"); }
    else {
      solver_subtree["type"] = "restartedgmressolver";
      solver_subtree["restart"] = "30";
      solver_subtree["maxit"] = "1000";
      solver_subtree["verbose"] = "0";
    }
    solver_subtree["verbose"] = rank == 0 ? solver_subtree["verbose"] : "0"; // verbosity is only set on MPI root rank
    solver_subtree["reduction"] = std::to_string(reduction);

    auto solver = Dune::getSolverFromFactory(op, solver_subtree, prec);

    // Make the rhs consistent (this is how the preconditioner, nonoverlapping operator and scalar product expect it)
    auto& d = native(r);
    novlp_comm->addOwnerCopyToAll(d, d);

    // Solve the linear system
    Dune::InverseOperatorResult stat;
    solver->apply(native(z), d, reduction, stat);
    res.converged = stat.converged;
    res.iterations = stat.iterations;
    res.elapsed = stat.elapsed;
    res.reduction = stat.reduction;
    res.conv_rate = stat.conv_rate;
  }

  typename Vec::ElementType norm(const Vec& v) const
  {
    using Dune::PDELab::Backend::Native;
    using Dune::PDELab::Backend::native;

    // Make vector consistent before computing norm
    auto x = native(v);
    novlp_comm->addOwnerCopyToOwnerCopy(x, x);
    return novlp_comm->norm(x);
  }

private:
  std::shared_ptr<NativeMat> A_ovlp;
  std::shared_ptr<PartitionOfUnity> pou;

  std::shared_ptr<Communication> novlp_comm;
  std::shared_ptr<Communication> ovlp_comm;

  Dune::Interface interface_ext;
  std::unique_ptr<Dune::VariableSizeCommunicator<>> varcomm;

  std::vector<NativeVec> native_template_vecs;

  Dune::ParameterTree subtree;

  bool matrix_is_additive;
};
#endif // HAVE_DUNE_PDELAB
