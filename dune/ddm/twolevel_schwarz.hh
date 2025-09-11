#pragma once

#include "coarsespaces/coarse_spaces.hh"
#include "combined_preconditioner.hh"
#include "dune/common/exceptions.hh"
#include "dune/ddm/datahandles.hh"
#include "galerkin_preconditioner.hh"
#include "helpers.hh"
#include "nonoverlapping_operator.hh"
#include "overlap_extension.hh"
#include "schwarz.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/preconditioner.hh>

#include <memory>

#if HAVE_DUNE_PDELAB
#include <dune/pdelab/backend/solver.hh>
#include <dune/pdelab/gridfunctionspace/genericdatahandle.hh>
#include <dune/pdelab/gridfunctionspace/interpolate.hh>
#include <dune/pdelab/solver/newton.hh>
#endif

template <class Mat, class X = Dune::BlockVector<Dune::FieldVector<double, 1>>, class Y = X>
class TwoLevelSchwarz : public Dune::Preconditioner<X, Y> {
  template <class Vec>
  struct CopyGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] = v; }
  };

public:
  using FineLevel = SchwarzPreconditioner<X, Mat>;
  using CoarseLevel = GalerkinPreconditioner<X>;

  template <class Vec, class RemoteIndices>
  TwoLevelSchwarz(std::shared_ptr<Mat> A_novlp, const RemoteIndices &remote_ids, const std::vector<Vec> &template_vecs, const Dune::ParameterTree &ptree,
                  const std::string &subtree_name = "twolevelschwarz")
      : prec(ptree, subtree_name)
  {
    const auto &subtree = ptree.sub(subtree_name);

    int overlap = subtree.get("overlap", 1);

    ExtendedRemoteIndices ext_indices(remote_ids, *A_novlp, overlap);
    A = std::make_shared<Mat>(ext_indices.create_overlapping_matrix(*A_novlp));

    auto pou = std::make_shared<PartitionOfUnity>(*A, ext_indices, ptree);
    const AttributeSet ownerAttribute{Attribute::owner};
    const AttributeSet copyAttribute{Attribute::copy};

    Dune::Interface owner_copy_interface;
    owner_copy_interface.build(ext_indices.get_remote_indices(), ownerAttribute, copyAttribute);
    Dune::BufferedCommunicator owner_copy_comm;
    owner_copy_comm.build<Vec>(owner_copy_interface);

    std::vector<Vec> extended_template_vecs(template_vecs.size(), Vec(A->N()));
    for (std::size_t i = 0; i < template_vecs.size(); ++i) {
      extended_template_vecs[i] = 0;
      for (std::size_t j = 0; j < template_vecs[i].N(); ++j) {
        extended_template_vecs[i][j] = template_vecs[i][j];
      }
      owner_copy_comm.forward<CopyGatherScatter<Vec>>(extended_template_vecs[i]);
    }

    POUCoarseSpace coarse_space(extended_template_vecs, *pou);
    auto fine = std::make_shared<FineLevel>(A, ext_indices.get_remote_indices(), pou, subtree, "fine");
    auto coarse = std::make_shared<CoarseLevel>(*A, coarse_space.get_basis(), ext_indices.get_remote_indices(), subtree, "coarse");

    auto op = std::make_shared<NonOverlappingOperator<Mat, Vec>>(A_novlp, remote_ids);
    prec.set_op(op);
    prec.add(fine);
    prec.add(coarse);
  }

  template <class RemoteIndices>
  TwoLevelSchwarz(std::shared_ptr<Mat> A_novlp, const RemoteIndices &remote_ids, const Dune::ParameterTree &ptree, const std::string &subtree_name = "twolevelschwarz") : prec(ptree, subtree_name)
  {
    const auto &subtree = ptree.sub(subtree_name);

    int overlap = subtree.get("overlap", 1);

    ExtendedRemoteIndices ext_indices(remote_ids, *A_novlp, overlap);
    A = std::make_shared<Mat>(ext_indices.create_overlapping_matrix(*A_novlp));

    auto pou = std::make_shared<PartitionOfUnity>(*A, ext_indices, ptree);
    POUCoarseSpace coarse_space(*pou);
    auto fine = std::make_shared<FineLevel>(A, ext_indices.get_remote_indices(), pou, subtree, "fine");
    auto coarse = std::make_shared<CoarseLevel>(*A, coarse_space.get_basis(), ext_indices.get_remote_indices(), subtree, "coarse");

    auto op = std::make_shared<NonOverlappingOperator<Mat, X>>(A_novlp, remote_ids);
    prec.set_op(op);
    prec.add(fine);
    prec.add(coarse);
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void pre(X &, Y &) override {}
  void post(X &) override {}

  void apply(X &v, const Y &d) override { prec.apply(v, d); }

private:
  std::shared_ptr<Mat> A; /// The overlapping matrix

  std::shared_ptr<FineLevel> fine;
  std::shared_ptr<CoarseLevel> coarse;
  CombinedPreconditioner<X, Y> prec;
};

#if HAVE_DUNE_PDELAB

template <class GFS>
auto make_remote_indices(const GFS &gfs, const Dune::MPIHelper &helper)
{
  using Dune::PDELab::Backend::native;

  // Using the grid function space, we can generate a globally unique numbering
  // of the dofs. This is done by taking the local index, shifting it to the
  // upper 32 bits of a 64 bit number and taking our MPI rank as the lower 32
  // bits.
  using GlobalIndexVec = Dune::PDELab::Backend::Vector<GFS, std::uint64_t>;
  GlobalIndexVec giv(gfs);
  for (std::size_t i = 0; i < giv.N(); ++i) {
    native(giv)[i] = (static_cast<std::uint64_t>(i + 1) << 32ULL) + helper.rank();
  }

  // Now we have a unique global indexing scheme in the interior of each process
  // subdomain; at the process boundary we take the smallest among all
  // processes.
  GlobalIndexVec giv_before(gfs);
  giv_before = giv; // Copy the vector so that we can find out if we are the
                    // owner of a border index after communication
  Dune::PDELab::MinDataHandle mindh(gfs, giv);
  gfs.gridView().communicate(mindh, Dune::All_All_Interface, Dune::ForwardCommunication);

  using BooleanVec = Dune::PDELab::Backend::Vector<GFS, bool>;
  BooleanVec isPublic(gfs);
  isPublic = false;
  Dune::PDELab::SharedDOFDataHandle shareddh(gfs, isPublic);
  gfs.gridView().communicate(shareddh, Dune::All_All_Interface, Dune::ForwardCommunication);

  using AttributeLocalIndex = Dune::ParallelLocalIndex<Attribute>;
  using GlobalIndex = std::uint64_t;
  using ParallelIndexSet = Dune::ParallelIndexSet<GlobalIndex, AttributeLocalIndex>;
  using RemoteIndices = Dune::RemoteIndices<ParallelIndexSet>;

  auto paridxs = std::make_shared<ParallelIndexSet>();
  paridxs->beginResize();
  for (std::size_t i = 0; i < giv.N(); ++i) {
    paridxs->add(native(giv)[i],
                 {i,                                                                            // Local index is just i
                  native(giv)[i] == native(giv_before)[i] ? Attribute::owner : Attribute::copy, // If the index didn't change above, we own it
                  native(isPublic)[i]}                                                          // SharedDOFDataHandle determines if an index is public
    );
  }
  paridxs->endResize();

  std::set<int> neighboursset;
  Dune::PDELab::GFSNeighborDataHandle nbdh(gfs, helper.rank(), neighboursset);
  gfs.gridView().communicate(nbdh, Dune::All_All_Interface, Dune::ForwardCommunication);
  std::vector<int> neighbours(neighboursset.begin(), neighboursset.end());

  auto *remoteindices = new RemoteIndices(*paridxs, *paridxs, helper.getCommunicator(), neighbours);
  remoteindices->rebuild<false>();

  // RemoteIndices store a reference to the paridxs that are passed to the constructor.
  // In order to avoid dangling references, we capture the paridxs shared_ptr in the lambda
  // to increase the reference count which ensures that it will be deleted as soon as the
  // remoteindices are deleted.
  return std::shared_ptr<RemoteIndices>(remoteindices, [paridxs](auto *ptr) mutable { delete ptr; });
}

// Wrapper for usage in PDELab's Newton solver
template <class Mat, class Vec>
class TwoLevelSchwarzSolver : public Dune::PDELab::LinearResultStorage {
  using NativeMat = Dune::PDELab::Backend::Native<Mat>;
  using NativeVec = Dune::PDELab::Backend::Native<Vec>;

  using AttributeLocalIndex = Dune::ParallelLocalIndex<Attribute>;
  using GlobalIndex = std::uint64_t;
  using ParallelIndexSet = Dune::ParallelIndexSet<GlobalIndex, AttributeLocalIndex>;
  using RemoteIndices = Dune::RemoteIndices<ParallelIndexSet>;

  using ExtendedIndices = ExtendedRemoteIndices<RemoteIndices, NativeMat>;
  using FineLevel = SchwarzPreconditioner<NativeVec, NativeMat>;
  using CoarseLevel = GalerkinPreconditioner<NativeVec>;

  struct CopyGatherScatter {
    using DataType = double;

    static DataType gather(const NativeVec &x, std::size_t i) { return x[i]; }
    static void scatter(NativeVec &x, DataType v, std::size_t i) { x[i] = v; }
  };

public:
  template <class GFS, class CC>
  explicit TwoLevelSchwarzSolver(const GFS &gfs, const CC &cc, const Dune::MPIHelper &helper, const Dune::ParameterTree &ptree, const std::string &subtree_name = "twolevelschwarz")
      : remoteids(make_remote_indices(gfs, helper)), comm(*this->remoteids), subtree(ptree.sub(subtree_name))
  {
    using Dune::PDELab::Backend::Native;
    using Dune::PDELab::Backend::native;

    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    interface_ext.build(*this->remoteids, allAttributes, allAttributes);
    varcomm = std::make_unique<Dune::VariableSizeCommunicator<>>(interface_ext);

    std::vector<Vec> template_vecs(1, gfs);
    int cnt = 0;
    Dune::PDELab::interpolate([](auto &&) { return 1; }, gfs, template_vecs[cnt++]);
    // Dune::PDELab::interpolate([](auto &&x) { return x[0]; }, gfs, template_vecs[cnt++]);
    // Dune::PDELab::interpolate([](auto &&x) { return x[1]; }, gfs, template_vecs[cnt++]);
    // Dune::PDELab::interpolate([](auto &&x) { return x[0] * x[0]; }, gfs, template_vecs[cnt++]);
    // Dune::PDELab::interpolate([](auto &&x) { return x[1] * x[1]; }, gfs, template_vecs[cnt++]);
    // Dune::PDELab::interpolate([](auto &&x) { return x[0] * x[1]; }, gfs, template_vecs[cnt++]);
    std::for_each(template_vecs.begin(), template_vecs.end(), [&](auto &&v) { Dune::PDELab::set_constrained_dofs(cc, 0., v); }); // ensure they're zero on the Dirichlet boundary

    native_template_vecs.resize(template_vecs.size(), NativeVec(template_vecs[0].N()));
    std::transform(template_vecs.begin(), template_vecs.end(), native_template_vecs.begin(), [](auto &&v) { return native(v); });
  }

  template <class M, class V, class W>
  void apply(M &A, V &z, W &r, typename Dune::template FieldTraits<typename W::ElementType>::real_type reduction)
  {
    using Dune::PDELab::Backend::Native;
    using Dune::PDELab::Backend::native;
    using Prec = TwoLevelSchwarz<Native<M>, Native<V>>;
    using Op = NonOverlappingOperator<Native<M>, Native<V>>;

    // If it's the first time this function is called, create the overlapping matrix.
    // In the subsequent calls, we only update the matrix.
    // TODO: Currently, updating the matrix is actually the expensive part (due to inefficiencies in
    //       Dune's VariableSizeCommunicator. We should try to improve this.
    if (!ext_ids) {
      Dune::initSolverFactories<Op>();

      int overlap = subtree.get("overlap", 1);
      ext_ids = std::make_unique<ExtendedIndices>(*remoteids, native(A), overlap);
      A_ovlp = std::make_shared<NativeMat>(ext_ids->create_overlapping_matrix(native(A)));

      pou = std::make_shared<PartitionOfUnity>(*A_ovlp, *ext_ids, subtree.sub("pou"));

      // Extend the template vectors that were created in the constructor to the overlapping
      // subdomains.
      const AttributeSet ownerAttribute{Attribute::owner};
      const AttributeSet copyAttribute{Attribute::copy};
      Dune::Interface owner_copy_interface;
      owner_copy_interface.build(ext_ids->get_remote_indices(), ownerAttribute, copyAttribute);
      Dune::BufferedCommunicator owner_copy_comm;
      owner_copy_comm.build<Vec>(owner_copy_interface);

      std::vector<NativeVec> extended_template_vecs(native_template_vecs.size(), NativeVec(A_ovlp->N()));
      for (std::size_t i = 0; i < native_template_vecs.size(); ++i) {
        extended_template_vecs[i] = 0;
        for (std::size_t j = 0; j < native_template_vecs[i].N(); ++j) {
          extended_template_vecs[i][j] = native_template_vecs[i][j];
        }
        owner_copy_comm.forward<CopyGatherScatter>(extended_template_vecs[i]);
      }
      native_template_vecs = std::move(extended_template_vecs);
    }
    else {
      ext_ids->update_overlapping_matrix(native(A), *A_ovlp);
    }

    // Set up the preconditioner
    POUCoarseSpace coarse_space(native_template_vecs, *pou);
    auto fine = std::make_shared<FineLevel>(A_ovlp, ext_ids->get_remote_indices(), pou, subtree, "fine");
    auto coarse = std::make_shared<CoarseLevel>(*A_ovlp, coarse_space.get_basis(), ext_ids->get_remote_indices(), subtree, "coarse");
    auto op = std::make_shared<Op>(A.storage(), *remoteids);
    auto prec = std::make_shared<CombinedPreconditioner<NativeVec>>(subtree, "");
    prec->set_op(op);
    prec->add(fine);
    prec->add(coarse);

    // Set up the solver
    int rank{};
    MPI_Comm_rank(remoteids->communicator(), &rank);
    Dune::ParameterTree solver_subtree;
    if (subtree.hasSub("solver")) {
      solver_subtree = subtree.sub("solver");
    }
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
    AddVectorDataHandle<Native<V>> advdh;
    auto d = native(r);
    advdh.setVec(d);
    varcomm->forward(advdh);

    // Solve the linear system
    Dune::InverseOperatorResult stat;
    solver->apply(native(z), d, reduction, stat);
    res.converged = stat.converged;
    res.iterations = stat.iterations;
    res.elapsed = stat.elapsed;
    res.reduction = stat.reduction;
    res.conv_rate = stat.conv_rate;
  }

  template <class V>
  typename V::ElementType norm(const V &v) const
  {
    using Dune::PDELab::Backend::Native;
    using Dune::PDELab::Backend::native;

    // Make vector consistent before computing norm
    AddVectorDataHandle<Native<V>> advdh;
    auto x = native(v);
    advdh.setVec(x);
    varcomm->forward(advdh);

    return comm.norm(native(x));
  }

private:
  std::unique_ptr<ExtendedIndices> ext_ids{};
  std::shared_ptr<NativeMat> A_ovlp{};
  std::shared_ptr<PartitionOfUnity> pou;

  std::shared_ptr<RemoteIndices> remoteids;
  NonOverlappingCommunicator comm;

  Dune::Interface interface_ext;
  std::unique_ptr<Dune::VariableSizeCommunicator<>> varcomm;

  std::vector<NativeVec> native_template_vecs;

  Dune::ParameterTree subtree;
};
#endif
