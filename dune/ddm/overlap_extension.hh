#pragma once

#include "datahandles.hh"
#include "helpers.hh"
#include "logger.hh"

#include <dune/common/enumset.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/mpitraits.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <limits>
#include <mpi.h>
#include <unordered_set>

/** @brief Creates an overlapping communication object from a non-overlapping one.
 *
 *  This function extends the communication pattern by adding overlap layers to the
 *  domain decomposition. It takes a non-overlapping communication object and creates
 *  a new communication object with extended index sets and remote indices.
 *
 *  @tparam Communication The type of the communication object (e.g., OwnerOverlapCopyCommunication)
 *  @tparam Mat The matrix type
 *  @param novlp_comm The non-overlapping communication object
 *  @param A The matrix defining the graph structure
 *  @param overlap The number of overlap layers to add
 *  @return A shared pointer to the new overlapping communication object
 */
template <class Communication, class Mat>
auto make_overlapping_communication(const Communication& novlp_comm, const Mat& A, int overlap)
{
  using RemoteIndices = typename Communication::RemoteIndices;
  using ParallelIndexSet = typename RemoteIndices::ParallelIndexSet;
  using GlobalIndex = typename RemoteIndices::GlobalIndex;

  using AttributeSet = Dune::OwnerOverlapCopyAttributeSet::AttributeSet;
  using AllSet = Dune::AllSet<AttributeSet>;

  auto* extend_event = Logger::get().registerOrGetEvent("OverlapExtension", "extend overlap");
  Logger::ScopedLog sl{extend_event};

  const auto& novlp_remoteids = novlp_comm.remoteIndices();
  MPI_Comm comm = novlp_remoteids.communicator();
  int rank{};
  MPI_Comm_rank(comm, &rank);

  if (novlp_remoteids.sourceIndexSet().size() != A.N()) {
    logger::error_all("make_overlapping_communication: Index set and matrix don't match, index set has size {}, matrix has size {}", novlp_remoteids.sourceIndexSet().size(), A.N());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Create the overlapping communication object early and work with its internal structures
  auto ovlp_comm = std::make_shared<Communication>(comm);
  auto& ext_remoteids = ovlp_comm->remoteIndices();
  auto& ext_indexset = ovlp_comm->indexSet();

  AllSet all_att;
  std::vector<std::size_t> index_set_sizes;
  index_set_sizes.reserve(overlap + 1);
  index_set_sizes.push_back(A.N());

  // Create communicator data structures
  Dune::Interface comm_if;
  comm_if.build(novlp_remoteids, all_att, all_att);
  auto varcomm = std::make_unique<Dune::VariableSizeCommunicator<>>(comm_if);

  // Initialize the local-to-global and global-to-local maps
  std::vector<GlobalIndex> ltg;
  std::unordered_set<GlobalIndex> gis;
  ltg.resize(novlp_remoteids.sourceIndexSet().size());
  for (const auto& it : novlp_remoteids.sourceIndexSet()) {
    ltg[it.local().local()] = it.global();
    gis.insert(it.global());
  }

  // Initialize the "boundary distance map"
  IdentifyBoundaryDataHandle ibdh(A, novlp_remoteids.sourceIndexSet());
  varcomm->forward(ibdh);
  const auto& boundary_mask = ibdh.get_boundary_mask();

  std::vector<int> boundary_distance;
  boundary_distance.resize(novlp_remoteids.sourceIndexSet().size(), std::numeric_limits<int>::max() - 1);
  for (std::size_t i = 0; i < boundary_distance.size(); ++i)
    if (boundary_mask[i]) boundary_distance[i] = 0;

  for (int round = 0; round <= 8 * overlap; ++round) {
    for (std::size_t i = 0; i < boundary_distance.size(); ++i) {
      for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
        auto nb_dist_plus_one = boundary_distance[cit.index()] + 1;
        if (nb_dist_plus_one < boundary_distance[i]) boundary_distance[i] = nb_dist_plus_one;
      }
    }
  }

  // Lambda to modify parallel index set public state
  auto modify_parindexset_public_state = [](const ParallelIndexSet& indexSet, auto&& isPublic) {
    ParallelIndexSet newIndexSet;
    newIndexSet.beginResize();
    for (const auto& idx : indexSet) newIndexSet.add(idx.global(), {idx.local().local(), idx.local().attribute(), idx.local().isPublic() or isPublic(idx.local().local())});
    newIndexSet.endResize();
    return newIndexSet;
  };

  // For each index, we find out which other ranks also know that index
  Dune::BufferedCommunicator buffcomm;
  buffcomm.build<RankTuple>(comm_if);

  RankTuple rt;
  rt.rank = rank;
  rt.rankmap.resize(index_set_sizes[0]);
  buffcomm.forward<RankDataHandle>(rt);

  // Store all neighbours in one set
  std::set<int> nbs_set;
  for (const auto& ranks : rt.rankmap) nbs_set.insert(ranks.begin(), ranks.end());

  // Set up the extended parallel index set by modifying public state.
  // TODO: Fix this.
  ext_indexset = modify_parindexset_public_state(novlp_remoteids.sourceIndexSet(), [&](int li) { return true or boundary_distance[li] <= overlap + 2; });

  // Set up the extended remote indices
  std::vector<int> nbs(nbs_set.begin(), nbs_set.end());
  ext_remoteids.setIndexSets(ext_indexset, ext_indexset, comm, nbs);
  ext_remoteids.template rebuild<false>();

  // Rebuild the communication data structures
  comm_if.free();
  comm_if.build(ext_remoteids, all_att, all_att);
  varcomm = std::make_unique<Dune::VariableSizeCommunicator<>>(comm_if, 10 * 1024 * 1024); // This will reserve 10*1024*1024*64 bits ≈ 80 megabytes per rank

  IndexsetExtensionMatrixGraphDataHandle extdh(rank, A, gis);
  UpdateRankInfoDataHandle uprdh(rank);

  std::vector<MPI_Request> reqs;
  std::vector<std::size_t> sendcount;
  std::vector<std::size_t> recvcount;

  std::vector<std::vector<int>> new_nbs_data;
  std::vector<std::vector<int>> new_nbs_data_recv;

  // Perform overlap extension rounds
  for (int round = 0; round < overlap; ++round) {
    extdh.set_index_set(ltg);
    extdh.rankmap = rt.rankmap;
    extdh.updated_rankmap = rt.rankmap;
    varcomm->forward(extdh);

    index_set_sizes.push_back(ltg.size());

    uprdh.set_rankmap(extdh.updated_rankmap);
    varcomm->forward(uprdh);

    std::map<int, std::set<int>> new_nbs;
    for (const auto& ranks : extdh.updated_rankmap) {
      for (const auto& p : ranks) {
        for (const auto& q : ranks)
          if (p != q and nbs_set.count(p)) new_nbs[p].insert(q);
      }
    }

    for (const auto& p : nbs_set)
      if (not new_nbs.contains(p)) new_nbs[p].insert(p);

    reqs.resize(2 * new_nbs.size());
    sendcount.resize(new_nbs.size());
    recvcount.resize(new_nbs.size());
    std::size_t i = 0; // Counter for reqs
    std::size_t j = 0; // Counter for send/recvcount
    for (const auto& [p, pnbs] : new_nbs) {
      sendcount[j] = pnbs.size();
      MPI_Irecv(&(recvcount[j]), 1, Dune::MPITraits<std::size_t>::getType(), p, 1, comm, &(reqs[i++]));
      MPI_Isend(&(sendcount[j++]), 1, Dune::MPITraits<std::size_t>::getType(), p, 1, comm, &(reqs[i++]));
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    new_nbs_data.resize(new_nbs.size());
    new_nbs_data_recv.resize(new_nbs.size());
    i = 0;
    j = 0;
    for (const auto& [p, pnbs] : new_nbs) {
      new_nbs_data[j].assign(pnbs.begin(), pnbs.end());
      new_nbs_data_recv[j].resize(recvcount[j]);

      MPI_Irecv(new_nbs_data_recv[j].data(), static_cast<int>(recvcount[j]), MPI_INT, p, 2, comm, &(reqs[i++]));
      MPI_Isend(new_nbs_data[j].data(), static_cast<int>(sendcount[j]), MPI_INT, p, 2, comm, &(reqs[i++]));
      j++;
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

    j = 0;
    for (const auto& [p, pnbs] : new_nbs) {
      (void)p; // Silence warning
      for (const auto& q : new_nbs_data_recv[j++])
        if (q != rank) nbs_set.insert(q);
    }

    ext_indexset.beginResize();
    for (std::size_t i = index_set_sizes[round]; i < index_set_sizes[round + 1]; ++i) ext_indexset.add(ltg[i], {i, Dune::OwnerOverlapCopyAttributeSet::copy, true});
    ext_indexset.endResize();

    ext_remoteids.setNeighbours(nbs_set);
    ext_remoteids.template rebuild<false>();

    comm_if.free();
    comm_if.build(ext_remoteids, all_att, all_att);

    rt.rankmap.clear();
    rt.rankmap.resize(index_set_sizes[round + 1]);

    buffcomm.free();
    buffcomm.build<RankTuple>(comm_if);
    buffcomm.forward<RankDataHandle>(rt);
  }

  // Final rebuild to ensure everything is consistent
  ext_remoteids.template rebuild<false>();

  std::vector<bool> ext_boundary_mask(ext_indexset.size(), false);
  for (std::size_t i = index_set_sizes[overlap - 1]; i < index_set_sizes[overlap]; ++i) ext_boundary_mask[i] = true;

  return std::make_pair(ovlp_comm, ext_boundary_mask);
}
