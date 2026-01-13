#pragma once

#include "dune/ddm/logger.hh"

#include <dune/grid/common/gridenums.hh>
#include <dune/istl/solvercategory.hh>

#if HAVE_DUNE_PDELAB

#include <dune/istl/owneroverlapcopy.hh>
#include <dune/pdelab/backend/interface.hh>
#include <dune/pdelab/gridfunctionspace/genericdatahandle.hh>
#include <memory>

template <class GFS>
auto make_communication(const GFS& gfs)
{
  using Dune::PDELab::Backend::native;

  using Communication = Dune::OwnerOverlapCopyCommunication<std::size_t, int>;
  auto communicator = gfs.gridView().comm();
  auto communication = std::make_shared<Communication>(communicator, Dune::SolverCategory::nonoverlapping);
  auto rank = gfs.gridView().comm().rank();

  using EntitySet = typename GFS::Traits::EntitySet;
  constexpr auto has_ghosts = EntitySet::partitions().contains(Dune::GhostEntity);

  auto interiorborder_all_interface = has_ghosts ? Dune::InteriorBorder_All_Interface : Dune::InteriorBorder_InteriorBorder_Interface;

  auto all_all_interface = has_ghosts ? Dune::All_All_Interface : Dune::InteriorBorder_InteriorBorder_Interface;

  // Determine an owner for each dof
  using RankVector = Dune::PDELab::Backend::Vector<GFS, int>;
  RankVector rank_partition(gfs, communicator.rank());
  Dune::PDELab::DisjointPartitioningDataHandle<GFS, RankVector> pdh(gfs, rank_partition);
  gfs.gridView().communicate(pdh, interiorborder_all_interface, Dune::ForwardCommunication);

  // Find out which dofs are shared with other processes
  using BooleanVec = Dune::PDELab::Backend::Vector<GFS, bool>;
  BooleanVec isPublic(gfs, false);
  Dune::PDELab::SharedDOFDataHandle shareddh(gfs, isPublic);
  gfs.gridView().communicate(shareddh, all_all_interface, Dune::ForwardCommunication);

  // Count dofs that we own. This will be used to create the global numbering
  using GlobalIndex = typename Communication::ParallelIndexSet::GlobalIndex;
  auto count = std::count_if(native(rank_partition).begin(), native(rank_partition).end(), [&](const auto& x) { return x == rank; });

  // Tell other processes how many indices we own so they can find out where their global numbering should start
  std::vector<std::size_t> counts(gfs.gridView().comm().size());
  gfs.gridView().comm().allgather(&count, 1, counts.data());

  logger::debug("Nonoverlapping dof count per rank: {}", counts);

  // Find out where we need to start counting our own dofs
  auto start = std::accumulate(counts.begin(), counts.begin() + rank, 0UL);
  using GlobalIndexVec = Dune::PDELab::Backend::Vector<GFS, std::uint64_t>;
  GlobalIndexVec giv(gfs);

  // Initialize with max value
  for (auto& val : native(giv)) val = std::numeric_limits<std::uint64_t>::max();

  // Assign global indices to owned DOFs
  auto it_giv = native(giv).begin();
  auto it_rank = native(rank_partition).begin();
  for (; it_giv != native(giv).end(); ++it_giv, ++it_rank)
    if (*it_rank == rank) *it_giv = start++;

  // Now tell other processes about the correct indices at shared dofs
  Dune::PDELab::MinDataHandle mindh(gfs, giv);
  gfs.gridView().communicate(mindh, all_all_interface, Dune::ForwardCommunication);

  auto& paridxs = communication->indexSet();
  paridxs.beginResize();
  unsigned int public_count = 0;
  for (std::size_t i = 0; i < giv.N(); ++i) {
    bool owned = native(rank_partition)[i] == rank;
    paridxs.add(native(giv)[i], {i, owned ? Dune::OwnerOverlapCopyAttributeSet::owner : Dune::OwnerOverlapCopyAttributeSet::copy, native(isPublic)[i]}

    );
    if (native(isPublic)[i]) public_count++;
  }
  paridxs.endResize();

  logger::debug_all("Created parallel index set with {} public indices", public_count);

  std::set<int> neighboursset;
  Dune::PDELab::GFSNeighborDataHandle nbdh(gfs, communicator.rank(), neighboursset);
  gfs.gridView().communicate(nbdh, all_all_interface, Dune::ForwardCommunication);

  communication->remoteIndices().setNeighbours(neighboursset);
  communication->remoteIndices().template rebuild<false>();

  return communication;
}

/**
 * @brief Make the matrix additive by zeroing appropriate entries.
 *
 * For a consistent matrix (where duplicate entries on different ranks are identical),
 * this method zeros out entries to create an additive matrix (where duplicate entries
 * sum to the correct value). The logic ensures each matrix entry is contributed by
 * exactly one process.
 *
 * Zeroing rules:
 * - Owner rows: zero entries to overlap columns (mask[j] == 2)
 * - Copy rows: zero all entries (only owners contribute)
 */
template <class Mat, class Communication>
void make_additive(Mat& A, const Communication& comm)
{
  using Dune::PDELab::Backend::Native;
  using Dune::PDELab::Backend::native;

  auto& A_native = native(A);
  const auto& pis = comm.indexSet();

  // Build mask: 0 = copy, 1 = owner, 2 = overlap
  std::vector<int> mask(A_native.N(), 1); // default: owner
  for (auto it = pis.begin(); it != pis.end(); ++it) {
    if (it->local().attribute() == Dune::OwnerOverlapCopyAttributeSet::copy) { mask[it->local().local()] = 0; }
    else if (it->local().attribute() == Dune::OwnerOverlapCopyAttributeSet::overlap) {
      mask[it->local().local()] = 2;
      // In a nonoverlapping decomposition, overlap DOFs should not exist
      // assert(false && "Overlap DOFs found in nonoverlapping matrix");
    }
  }

  // Zero out entries to make matrix additive
  using NativeMat = std::decay_t<decltype(A_native)>;
  using RowIterator = typename NativeMat::RowIterator;
  using ColIterator = typename NativeMat::ColIterator;

  for (RowIterator row = A_native.begin(); row != A_native.end(); ++row) {
    const auto i = row.index();

    if (mask[i] == 0) {
      // Copy row: zero all entries (only owner processes contribute)
      for (ColIterator col = (*row).begin(); col != (*row).end(); ++col) *col = 0;
    }
    else if (mask[i] == 1) {
      // Owner row: zero entries to overlap columns
      for (ColIterator col = (*row).begin(); col != (*row).end(); ++col) {
        const auto j = col.index();
        if (mask[j] == 2) *col = 0;
      }
    }
    // mask[i] == 2 (overlap rows) should not occur in nonoverlapping matrix
  }
}
#endif
