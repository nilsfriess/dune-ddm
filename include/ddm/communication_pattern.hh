#pragma once

#include "communication_nodes.hh"
#include "dune/ddm/logger.hh"
#include "neighbours.hh"

#include <mpi.h>
#include <set>
#include <unordered_map>
#include <vector>

namespace ddm {
/**
 * @brief The topology of the data exchange between the owners ("roots") and the copies ("leaves")
 *        of an index: which indices are exchanged with which peer.
 *
 * The pattern is built from a roots array (one entry per entry of the local vector) that describes,
 * for every locally stored index, on which rank it is owned and which global id it carries. From
 * that it derives, for the broadcast (owner -> copies) and for the reduction (all holders <-> all
 * holders) operation, the list of local indices exchanged with each neighbour.
 */
class CommunicationPattern {
public:
  /// The local indices exchanged with one peer.
  struct PeerIndices {
    std::vector<int> send_idx; ///< indices we send to the peer
    std::vector<int> recv_idx; ///< indices we receive from the peer
  };

  using IndexMap = std::unordered_map<int, PeerIndices>;

  CommunicationPattern(MPI_Comm comm_, const std::vector<CommunicationNodes>& roots)
  {
    // Duplicate first: identify_neighbours() receives from any source, so it must not run on the caller's
    // communicator, where it could pick up unrelated messages
    MPI_Comm_dup(comm_, &comm);
    neighbours_ = detail::identify_neighbours(comm, roots);
    logger::trace_all("CommunicationPattern() neighbours: {}", logger::join(neighbours_));

    // Global ids are unique across all ranks, so the id alone identifies an entry of our roots
    // array. Both plan builders need to translate ids they receive back into local indices, so
    // just build a global-to-local map here.
    std::unordered_map<std::int64_t, int> gid_to_local;
    gid_to_local.reserve(roots.size());
    for (std::size_t i = 0; i < roots.size(); ++i) gid_to_local[roots[i].gid] = (int)i;

    build_broadcast_plan(roots, gid_to_local); // this must go first because the reduction plan uses the broadcast plan
    build_reduction_plan(roots, gid_to_local);
  }

  CommunicationPattern(const CommunicationPattern&) = delete;
  CommunicationPattern& operator=(const CommunicationPattern&) = delete;

  ~CommunicationPattern() { MPI_Comm_free(&comm); }

  /// Indices of the broadcast (owner -> copies) operation.
  const IndexMap& broadcast_indices() const { return broadcast_idxs; }

  /// Indices of the reduction (all holders <-> all holders) operation.
  const IndexMap& reduction_indices() const { return reduction_idxs; }

  /// The communicator the pattern was built on. This is a private duplicate, so exchanges using it
  /// cannot be confused with any other traffic on the communicator that was passed in.
  MPI_Comm communicator() const { return comm; }

  const std::vector<int>& neighbours() const { return neighbours_; }

private:
  /** Builds the plan for the broadcast (root -> leaves) operation
   *
   *  The \p roots array is a list (of the same size as our local index set) that
   *  contains pairs {r, g} where r is a MPI rank number and g is the global id of the
   *  index. If r == rank, then we are the owner of that index. Otherwise, we store a
   *  copy of an index that is owned by r.
   *
   *  For a broadcast operation, we need to know which ranks own copies of indices that
   *  we own. In other words, we need to invert the mapping induced by the roots array.
   *  This is what this method does.
   */
  void build_broadcast_plan(const std::vector<CommunicationNodes>& roots, const std::unordered_map<std::int64_t, int>& gid_to_local)
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Communication", "broadcast plan")};

    int rank;
    MPI_Comm_rank(comm, &rank);

    std::unordered_map<int, int> nleaves; // How many leaves do we have that attach to a node on the corresponding rank
    // We'll send a message to every neighbour (even if it's zero)
    for (auto neighbour : neighbours_) nleaves[neighbour] = 0;

    for (const auto& root : roots) {
      if (rank == root.rank) continue;

      nleaves.at(root.rank)++; // Use .at() here to catch a bug: if root.rank is not in nleaves, then neighbours is wrong
    }

    // Post the receives and sends for the leaf counts
    std::unordered_map<int, int> leave_count;
    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * neighbours_.size());
    for (auto neighbour : neighbours_) MPI_Irecv(&(leave_count[neighbour]), 1, MPI_INT, neighbour, 1, comm, &reqs.emplace_back());
    for (auto neighbour : neighbours_) MPI_Isend(&nleaves[neighbour], 1, MPI_INT, neighbour, 1, comm, &reqs.emplace_back());
    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Now send the actual leave data (here we now ignore ranks that don't have roots for our leaves)
    reqs.resize(0);
    std::unordered_map<int, std::vector<std::int64_t>> leaves;
    for (auto neighbour : neighbours_) {
      if (leave_count[neighbour] > 0) {
        leaves[neighbour].resize(leave_count[neighbour]);
        MPI_Irecv(leaves[neighbour].data(), leave_count[neighbour], Dune::MPITraits<std::int64_t>::getType(), neighbour, 2, comm, &reqs.emplace_back());
      }
    }

    std::unordered_map<int, std::vector<int>> recv_indices;         // The indices of the leaves in *our* local numbering (this is not communicated,
                                                                    // we just need this to know where to put the remote data later)
    std::unordered_map<int, std::vector<std::int64_t>> leaves_data; // The indices of the leaves in global numbering
    int count = 0;
    for (const auto& root : roots) {
      if (rank != root.rank) {
        if (!leaves_data.contains(root.rank)) {
          recv_indices[root.rank].reserve(nleaves[root.rank]);
          leaves_data[root.rank].reserve(nleaves[root.rank]);
        }

        recv_indices[root.rank].push_back(count);
        leaves_data[root.rank].push_back(root.gid);
      }
      count++;
    }

    for (const auto& [peer, data] : leaves_data) MPI_Isend(data.data(), (int)data.size(), Dune::MPITraits<std::int64_t>::getType(), peer, 2, comm, &reqs.emplace_back());
    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // The indices in the leaves map use global ids. We need to convert them into local numbering on
    // the current rank. Every id we receive here is one we own, so it must be in our roots array.
    std::unordered_map<int, std::vector<int>> leaves_local_numbering;
    for (const auto& [peer, indices] : leaves) {
      auto& local = leaves_local_numbering[peer];
      local.resize(indices.size());
      for (std::size_t j = 0; j < indices.size(); ++j) {
        auto it = gid_to_local.find(indices[j]);
        if (it == gid_to_local.end()) {
          // If we reach this, rank `peer` believes we own an index that we don't even hold.
          // This is a bug, so we can abort here.
          logger::error_all("Rank {} claims we own index {}, which is not in our local roots array", peer, indices[j]);
          MPI_Abort(comm, 1);
        }
        local[j] = it->second;
      }
    }

    // Now convert to the format that the plan expects
    for (auto&& [peer, indices] : recv_indices) broadcast_idxs[peer].recv_idx = std::move(indices);
    for (auto&& [peer, indices] : leaves_local_numbering) broadcast_idxs[peer].send_idx = std::move(indices);
  }

  /** Builds the plan for a reduction operation
   *
   *  The reduction is a symmetric exchange: for every pair of ranks sharing an index,
   *  both sides send their values to each other and add the received values locally.
   *  Because begin() gathers all send buffers before end() scatters anything, every
   *  holder of a shared index ends up with the sum of all holders' pre-exchange values
   *  (same argument as ISTL's addOwnerCopyToOwnerCopy). Thus the post-condition is that
   *  all ranks holding a shared index agree on the (summed) value.
   *
   *  For every pair of ranks (A, B), the plan must therefore contain the FULL set of
   *  indices they share, in any role:
   *  - indices A holds copies of that B owns (leaf edge, from A's broadcast recv_idx),
   *  - indices A owns that B holds copies of (owner edge, from A's broadcast send_idx),
   *  - indices a third rank owns that both A and B hold copies of (sibling edge).
   *
   *  The first two are already available locally from the broadcast plan on both sides.
   *  For the sibling edges, the owners "introduce" their leaves to each other: every
   *  owner knows (from build_broadcast_plan) which of its indices each leaf copies, and
   *  for each multi-holder index it sends each holder the list of the other holders.
   *
   *  To guarantee that both ends of an edge agree on the order in which index values are
   *  packed into the buffers, all per-peer index lists are sorted by (owner rank, global
   *  id) — a key that is globally consistent for a given shared index and known to
   *  every rank via its roots array.
   */
  void build_reduction_plan(const std::vector<CommunicationNodes>& roots, const std::unordered_map<std::int64_t, int>& gid_to_local)
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Communication", "reduction plan")};

    int rank;
    MPI_Comm_rank(comm, &rank);

    // On the owner, broadcast_idxs[p].send_idx holds the owner-local ids of all indices that leaf
    // p copies. Invert it to find, per owned index, all ranks holding copies of it.
    std::unordered_map<int, std::set<int>> holders; // my local idx -> ranks holding copies of it
    for (const auto& [peer, indices] : broadcast_idxs)
      for (auto idx : indices.send_idx) holders[idx].insert(peer);

    // For each leaf p and each index i it copies, the owner sends every other holder of i.
    // Payload: flattened pairs (sibling_rank, global_id). Peers are always neighbours
    // (they are exactly the leaves that reported to us in build_broadcast_plan).
    std::unordered_map<int, std::vector<int>> intros_siblings;
    std::unordered_map<int, std::vector<std::int64_t>> intros_gids;
    for (const auto& [peer, indices] : broadcast_idxs) {
      for (auto idx : indices.send_idx) {
        auto it = holders.find(idx);
        if (it == holders.end()) continue;
        for (auto sibling : it->second) {
          if (sibling == peer) continue;
          intros_siblings[peer].emplace_back(sibling);
          intros_gids[peer].emplace_back(roots[idx].gid);
        }
      }
    }

    // Exchange introduction counts
    std::unordered_map<int, int> nsend; // number of pairs we send per neighbour
    for (auto n : neighbours_) nsend[n] = 0;
    for (const auto& [peer, data] : intros_siblings) nsend.at(peer) = (int)data.size();

    std::unordered_map<int, int> nrecv;
    std::vector<MPI_Request> reqs;
    reqs.reserve(4 * neighbours_.size());
    for (auto neighbour : neighbours_) MPI_Irecv(&(nrecv[neighbour]), 1, MPI_INT, neighbour, 1, comm, &reqs.emplace_back());
    for (auto neighbour : neighbours_) MPI_Isend(&nsend[neighbour], 1, MPI_INT, neighbour, 1, comm, &reqs.emplace_back());
    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Exchange introduction data
    reqs.clear();
    std::unordered_map<int, std::vector<int>> recv_intros_siblings;
    std::unordered_map<int, std::vector<std::int64_t>> recv_intros_gids;
    for (auto neighbour : neighbours_) {
      if (nrecv[neighbour] > 0) {
        recv_intros_siblings[neighbour].resize(nrecv[neighbour]);
        recv_intros_gids[neighbour].resize(nrecv[neighbour]);
        MPI_Irecv(recv_intros_siblings[neighbour].data(), nrecv[neighbour], MPI_INT, neighbour, 2, comm, &reqs.emplace_back());
        MPI_Irecv(recv_intros_gids[neighbour].data(), nrecv[neighbour], Dune::MPITraits<std::int64_t>::getType(), neighbour, 3, comm, &reqs.emplace_back());
      }
    }
    for (const auto& [peer, data] : intros_siblings)
      if (not data.empty()) MPI_Isend(data.data(), (int)data.size(), MPI_INT, peer, 2, comm, &reqs.emplace_back());
    for (const auto& [peer, data] : intros_gids)
      if (not data.empty()) MPI_Isend(data.data(), (int)data.size(), Dune::MPITraits<std::int64_t>::getType(), peer, 3, comm, &reqs.emplace_back());

    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Introductions always come from the owner of the index in question, so the sender must be the
    // owner rank we recorded for it. Each pair adds one sibling to one of my copy indices.
    std::unordered_map<int, std::set<int>> siblings; // my local idx -> sibling ranks
    for (const auto& [peer, data] : recv_intros_gids) {
      for (std::size_t k = 0; k < data.size(); k++) {
        auto it = gid_to_local.find(data[k]);
        if (it == gid_to_local.end() or roots[it->second].rank != peer) {
          // If we reach this, we didn't find the remote root index in our roots list, or we
          // disagree with the sender about who owns it. This is a bug, so we can abort here.
          logger::error_all("Did not find remote root index {} of rank {} in local roots array", data[k], peer);
          MPI_Abort(comm, 1);
        }
        siblings[it->second].insert(recv_intros_siblings[peer][k]);
      }
    }

    // For every peer, collect all local indices shared with it in any role (see above),
    // keyed by (owner rank, global id) so both ends sort identically.
    std::unordered_map<int, std::map<std::pair<int, std::int64_t>, int>> shared; // peer -> (sort key -> local idx)
    for (const auto& [peer, indices] : broadcast_idxs) {
      for (auto idx : indices.recv_idx) shared[peer][{roots[idx].rank, roots[idx].gid}] = idx; // copies owned by peer
      for (auto idx : indices.send_idx) shared[peer][{rank, roots[idx].gid}] = idx;            // owned indices copied by peer
    }
    for (const auto& [idx, sibs] : siblings)
      for (auto s : sibs) shared[s][{roots[idx].rank, roots[idx].gid}] = idx;

    for (const auto& [peer, entries] : shared) {
      auto& idxs = reduction_idxs[peer];
      idxs.recv_idx.reserve(entries.size());
      for (const auto& [key, idx] : entries) idxs.recv_idx.push_back(idx);
      // Reduction is symmetric: we send the same indices we receive.
      idxs.send_idx = idxs.recv_idx;
    }
  }

  IndexMap broadcast_idxs; ///< Indices to communicate from owners to copies (= roots to leaves)
  IndexMap reduction_idxs; ///< Indices to communicate between all holders of a shared index

  MPI_Comm comm{};
  std::vector<int> neighbours_;
};
} // namespace ddm
