#pragma once

#include <ddm/communication.hh>
#include <ddm/communication_nodes.hh>

#include <cstdint>
#include <dune/grid/common/gridenums.hh>
#include <dune/pdelab/backend/interface.hh>
#include <dune/pdelab/gridfunctionspace/genericdatahandle.hh>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

/** Creates the ddm::Communication for the dofs of a PDELab GridFunctionSpace. The local numbering is the container
 *  index of the GFS (flat, i.e. a GFS without blocking), the global numbering is computed here: every dof gets one
 *  owner rank, the owners number their dofs consecutively (rank by rank) and send the numbers to the other ranks
 *  holding the dof. Collective.
 *
 *  This is the global numbering part of PDELab's ISTL::ParallelHelper, without the ISTL index sets.
 */
template <class GFS>
std::shared_ptr<ddm::Communication> make_communication(const GFS& gfs)
{
  using Dune::PDELab::Backend::native;

  const auto& gv = gfs.gridView();
  const int rank = gv.comm().rank();

  using EntitySet = typename GFS::Traits::EntitySet;
  constexpr bool has_ghosts = EntitySet::partitions().contains(Dune::GhostEntity);
  const auto interiorborder_all_interface = has_ghosts ? Dune::InteriorBorder_All_Interface : Dune::InteriorBorder_InteriorBorder_Interface;
  const auto all_all_interface = has_ghosts ? Dune::All_All_Interface : Dune::InteriorBorder_InteriorBorder_Interface;

  // One owner rank for every dof
  using RankVector = Dune::PDELab::Backend::Vector<GFS, int>;
  RankVector owner(gfs, rank);
  Dune::PDELab::DisjointPartitioningDataHandle<GFS, RankVector> pdh(gfs, owner);
  gv.communicate(pdh, interiorborder_all_interface, Dune::ForwardCommunication);

  // The owners number their dofs consecutively, starting after the dofs of the ranks before them
  const auto n = native(owner).N();
  std::uint64_t count = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (native(owner)[i][0] == rank) ++count;

  std::vector<std::uint64_t> counts(gv.comm().size());
  gv.comm().allgather(&count, 1, counts.data());
  auto next = std::accumulate(counts.begin(), counts.begin() + rank, std::uint64_t{0});

  // Every rank holding a dof that it does not own gets the number of the owner, the minimum of the owner's number and
  // the placeholder of all other ranks
  using GlobalIndexVector = Dune::PDELab::Backend::Vector<GFS, std::uint64_t>;
  GlobalIndexVector gid(gfs, std::numeric_limits<std::uint64_t>::max());
  for (std::size_t i = 0; i < n; ++i)
    if (native(owner)[i][0] == rank) native(gid)[i][0] = next++;
  Dune::PDELab::MinDataHandle<GFS, GlobalIndexVector> mdh(gfs, gid);
  gv.communicate(mdh, all_all_interface, Dune::ForwardCommunication);

  std::vector<ddm::CommunicationNodes> nodes(n);
  for (std::size_t i = 0; i < n; ++i) nodes[i] = {.rank = native(owner)[i][0], .gid = static_cast<std::int64_t>(native(gid)[i][0])};
  return std::make_shared<ddm::Communication>(gv.comm(), nodes);
}
