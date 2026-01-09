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
 *  The algorithm works by:
 *  1. Identifying boundary DOFs using matrix graph structure
 *  2. Computing boundary distance via BFS from boundary DOFs
 *  3. Iteratively extending the index set by communicating matrix graph neighbors
 *  4. Tracking which ranks know which indices for communication setup
 *  5. Rebuilding remote indices after each extension round
 *
 *  @tparam Communication The type of the communication object (e.g., OwnerOverlapCopyCommunication)
 *  @tparam Mat The matrix type (must support row iteration via A[i].begin()/end())
 *
 *  @param novlp_comm The non-overlapping communication object. Must have:
 *                    - remoteIndices().sourceIndexSet().size() == A.N()
 *                    - Non-overlapping partitioning (no shared DOFs between ranks)
 *  @param A The matrix defining the graph structure for overlap extension
 *  @param overlap The number of overlap layers to add. Must be > 0.
 *
 *  @return A pair containing:
 *          - First: shared_ptr to the new overlapping Communication object
 *          - Second: vector<bool> boundary mask where true indicates DOFs on the
 *                    outermost layer of the extended overlap region (useful for
 *                    applying boundary conditions in Schwarz methods)
 *
 *  @pre overlap > 0
 *  @pre novlp_comm.remoteIndices().sourceIndexSet().size() == A.N()
 *
 *  @note Memory usage: Allocates approximately buffer_size_mb MB per rank for
 *        variable-size communication buffers (configurable, default 10 MB).
 *  @note Complexity: O(overlap * (|V| + |E|)) where |V| is number of DOFs and
 *        |E| is number of matrix non-zeros.
 */
template <class Communication, class Mat>
auto make_overlapping_communication(const Communication& novlp_comm, const Mat& A, int overlap, std::size_t buffer_size_mb = 10)
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
  MPI_CHECK(MPI_Comm_rank(comm, &rank));

  // Validate overlap parameter
  if (overlap <= 0) {
    logger::error_all("make_overlapping_communication: overlap must be positive, got {}", overlap);
    MPI_Abort(comm, 1);
  }

  if (novlp_remoteids.sourceIndexSet().size() != A.N()) {
    logger::error_all("make_overlapping_communication: Index set and matrix don't match, index set has size {}, matrix has size {}", novlp_remoteids.sourceIndexSet().size(), A.N());
    MPI_Abort(comm, 1);
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

  // Initialize the "boundary distance map" using BFS from boundary DOFs
  // This computes shortest path distance to boundary in the matrix graph
  IdentifyBoundaryDataHandle ibdh(A, novlp_remoteids.sourceIndexSet());
  varcomm->forward(ibdh);
  const auto& boundary_mask = ibdh.get_boundary_mask();

  std::vector<int> boundary_distance(novlp_remoteids.sourceIndexSet().size(), std::numeric_limits<int>::max() - 1);

  // BFS initialization: enqueue all boundary DOFs with distance 0
  std::vector<std::size_t> bfs_queue;
  bfs_queue.reserve(boundary_distance.size());
  for (std::size_t i = 0; i < boundary_distance.size(); ++i) {
    if (boundary_mask[i]) {
      boundary_distance[i] = 0;
      bfs_queue.push_back(i);
    }
  }

  // BFS traversal to compute boundary distances
  // We only need distances up to (overlap + 2) for the public state modification
  const int max_distance = overlap + 2;
  std::size_t queue_start = 0;
  while (queue_start < bfs_queue.size()) {
    std::size_t current = bfs_queue[queue_start++];
    int current_dist = boundary_distance[current];
    if (current_dist >= max_distance) continue; // No need to explore further

    for (auto cit = A[current].begin(); cit != A[current].end(); ++cit) {
      std::size_t neighbor = cit.index();
      if (boundary_distance[neighbor] > current_dist + 1) {
        boundary_distance[neighbor] = current_dist + 1;
        bfs_queue.push_back(neighbor);
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
  // DOFs within (overlap + 2) layers of the boundary are marked as public.
  //
  // Theoretical lower bound: DOFs at distance <= overlap-1 need to send graph info
  // during the overlap extension rounds (round r needs distance r DOFs to send).
  //
  // However, in practice <= overlap fails because:
  // 1. After adding DOFs at layer k, RemoteIndices.rebuild<false>() needs the
  //    corresponding original DOFs (at distance k) to be public to establish
  //    the owner-copy mapping correctly.
  // 2. The UpdateRankInfoDataHandle propagates rank knowledge, which may require
  //    an additional layer of public DOFs for correct bookkeeping.
  //
  // The value (overlap + 2) was determined empirically to be sufficient.
  // Using exactly (overlap) causes RemoteIndices rebuild failures.
  // TODO: Derive the exact mathematical bound or simplify the algorithm.
  ext_indexset = modify_parindexset_public_state(novlp_remoteids.sourceIndexSet(), [&](int li) { return boundary_distance[li] <= overlap + 2; });

  // Set up the extended remote indices
  std::vector<int> nbs(nbs_set.begin(), nbs_set.end());
  ext_remoteids.setIndexSets(ext_indexset, ext_indexset, comm, nbs);
  ext_remoteids.template rebuild<false>();

  // Rebuild the communication data structures
  comm_if.free();
  comm_if.build(ext_remoteids, all_att, all_att);
  // Buffer size for variable-size communication (in bytes, converted from MB parameter)
  // Each rank allocates buffer_size_mb * 1024 * 1024 bytes for message buffers
  varcomm = std::make_unique<Dune::VariableSizeCommunicator<>>(comm_if, buffer_size_mb * 1024 * 1024);

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
      if (new_nbs.count(p) == 0) new_nbs[p].insert(p);

    reqs.resize(2 * new_nbs.size());
    sendcount.resize(new_nbs.size());
    recvcount.resize(new_nbs.size());
    std::size_t req_idx = 0;  // Counter for reqs
    std::size_t msg_idx = 0;  // Counter for send/recvcount
    for (const auto& [p, pnbs] : new_nbs) {
      sendcount[msg_idx] = pnbs.size();
      MPI_CHECK(MPI_Irecv(&(recvcount[msg_idx]), 1, Dune::MPITraits<std::size_t>::getType(), p, 1, comm, &(reqs[req_idx++])));
      MPI_CHECK(MPI_Isend(&(sendcount[msg_idx++]), 1, Dune::MPITraits<std::size_t>::getType(), p, 1, comm, &(reqs[req_idx++])));
    }
    MPI_CHECK(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE));

    new_nbs_data.resize(new_nbs.size());
    new_nbs_data_recv.resize(new_nbs.size());
    req_idx = 0;
    msg_idx = 0;
    for (const auto& [p, pnbs] : new_nbs) {
      new_nbs_data[msg_idx].assign(pnbs.begin(), pnbs.end());
      new_nbs_data_recv[msg_idx].resize(recvcount[msg_idx]);

      MPI_CHECK(MPI_Irecv(new_nbs_data_recv[msg_idx].data(), static_cast<int>(recvcount[msg_idx]), MPI_INT, p, 2, comm, &(reqs[req_idx++])));
      MPI_CHECK(MPI_Isend(new_nbs_data[msg_idx].data(), static_cast<int>(sendcount[msg_idx]), MPI_INT, p, 2, comm, &(reqs[req_idx++])));
      msg_idx++;
    }
    MPI_CHECK(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE));

    msg_idx = 0;
    for (const auto& [p, pnbs] : new_nbs) {
      (void)p; // Silence warning
      for (const auto& q : new_nbs_data_recv[msg_idx++])
        if (q != rank) nbs_set.insert(q);
    }

    ext_indexset.beginResize();
    for (std::size_t idx = index_set_sizes[round]; idx < index_set_sizes[round + 1]; ++idx) ext_indexset.add(ltg[idx], {idx, Dune::OwnerOverlapCopyAttributeSet::copy, true});
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
