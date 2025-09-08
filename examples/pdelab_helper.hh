#pragma once

#include <dune/common/parallel/mpihelper.hh>
#include <dune/pdelab.hh>

#include "dune/ddm/helpers.hh"

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
  gfs.gridView().communicate(mindh, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);

  using BooleanVec = Dune::PDELab::Backend::Vector<GFS, bool>;
  BooleanVec isPublic(gfs);
  isPublic = false;
  Dune::PDELab::SharedDOFDataHandle shareddh(gfs, isPublic);
  gfs.gridView().communicate(shareddh, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);

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
  gfs.gridView().communicate(nbdh, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);
  std::vector<int> neighbours(neighboursset.begin(), neighboursset.end());

  auto *remoteindices = new RemoteIndices(*paridxs, *paridxs, helper.getCommunicator(), neighbours);
  remoteindices->rebuild<false>();

  // RemoteIndices store a reference to the paridxs that are passed to the constructor.
  // In order to avoid dangling references, we capture the paridxs shared_ptr in the lambda
  // to increase the reference count which ensures that it will be deleted as soon as the
  // remoteindices are deleted.
  return std::shared_ptr<RemoteIndices>(remoteindices, [paridxs](auto *ptr) mutable { delete ptr; });
}
