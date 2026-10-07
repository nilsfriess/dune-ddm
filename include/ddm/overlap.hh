#pragma once

#include "ddm/check.hh"
#include "ddm/communication.hh"
#include "ddm/communication_nodes.hh"
#include "ddm/communication_pattern.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/sparse_exchange.hh"
#include "dune/ddm/logger.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <dune/common/parametertree.hh>
#include <map>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ddm {
// An overlapping index set, as created by extend_overlap()
struct Overlap {
  std::shared_ptr<Communication> comm; ///< communication on the overlapping index set
  std::vector<int> layer;              ///< for every overlapping index: 0 for the original ones, k for the ones added in round k
  int layers = 0;                      ///< the number of layers the index set was extended by (the largest possible value of layer)
  Index n_original = 0;                ///< the original indices are the first n_original ones, in their original order
};

namespace detail {
// Tags for the messages of the overlap setup, distinct from the ones the exchanges of a Communication use
inline constexpr int overlap_counts_tag = 100;
inline constexpr int overlap_data_tag = 101;
inline constexpr int overlap_request_tag = 102;

/** For each index in idxs[peer].send_idx, this function sends a list to every peer and receives one per
 *  index in idxs[peer].recv_idx. The lists are built using the gather function that is passed as an argument
 *  and gather(i, out) must append the list that corresponds to local index i to the std::vector out.
 *
 *  Returns for every peer the received lists, in the order of idxs[peer].recv_idx. This uses the ReductionPlan
 *  of the Communication class since this is basically a reduction (with the reduction operation bein list
 *  concatenation instead of addition).
 *
 */
template <class U, class Gather>
std::unordered_map<int, std::vector<std::vector<U>>> exchange_lists(MPI_Comm comm, const CommunicationPattern::IndexMap& idxs, Gather&& gather)
{
  static_assert(std::is_trivially_copyable_v<U>, "the lists are sent as bytes");

  struct Outgoing {
    std::vector<int> counts; // The lengths of all lists we send to a peer
    std::vector<U> data;     // The flat list data that we send to a peer
  };
  std::unordered_map<int, Outgoing> out;
  std::unordered_map<int, std::vector<int>> in_counts; // The lengths of the lists we receive
  std::vector<MPI_Request> reqs;

  // First the length of every list ...
  for (const auto& [peer, indices] : idxs) {
    auto& o = out[peer];
    for (auto i : indices.send_idx) {
      const auto before = o.data.size();
      gather(i, o.data);
      o.counts.push_back(static_cast<int>(o.data.size() - before));
    }

    if (!indices.recv_idx.empty()) {
      auto& counts = in_counts[peer];
      // From the reduction plan we already know how many lists (= how many counts) we will receive
      counts.resize(indices.recv_idx.size());
      MPI_Irecv(counts.data(), static_cast<int>(counts.size()), MPI_INT, peer, overlap_counts_tag, comm, &reqs.emplace_back());
    }
    if (!o.counts.empty()) MPI_Isend(o.counts.data(), static_cast<int>(o.counts.size()), MPI_INT, peer, overlap_counts_tag, comm, &reqs.emplace_back());
  }
  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  reqs.clear();

  // ... then the lists themselves
  std::unordered_map<int, std::vector<U>> in_data;
  for (const auto& [peer, counts] : in_counts) {
    auto& data = in_data[peer];
    data.resize(std::accumulate(counts.begin(), counts.end(), 0));
    if (!data.empty()) MPI_Irecv(data.data(), static_cast<int>(data.size() * sizeof(U)), MPI_BYTE, peer, overlap_data_tag, comm, &reqs.emplace_back());
  }
  for (auto& [peer, o] : out)
    if (!o.data.empty()) MPI_Isend(o.data.data(), static_cast<int>(o.data.size() * sizeof(U)), MPI_BYTE, peer, overlap_data_tag, comm, &reqs.emplace_back());
  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  std::unordered_map<int, std::vector<std::vector<U>>> result;
  for (const auto& [peer, counts] : in_counts) {
    auto& lists = result[peer];
    lists.reserve(counts.size());
    auto it = in_data[peer].begin();
    for (auto count : counts) {
      lists.emplace_back(it, it + count);
      it += count;
    }
  }
  return result;
}

} // namespace detail

/** Extends the index set of comm by `layers` layers of neighbours in `graph`. Collective, all ranks must pass the same
 *  number of layers.
 *
 *  graph is a graph on the local indices of comm (e.g. the pattern of the local matrix). Like the local matrices of an
 *  additive Mat, the local graphs are summed up: two indices are neighbours if they are neighbours in the graph of any
 *  rank. Layer k consists of the indices whose distance to the original index set is k.
 *
 *  The original indices keep their local numbers, the new ones are appended layer by layer.
 *
 *  A Communication is completely determined by its roots array (owner rank and global id of every local index), so
 *  all this function does is append the (owner, gid) pairs of the new indices to the roots array. A new index keeps
 *  its owner. The neighbours and exchange plans are then computed by the constructor of the new Communication.
 */
inline Overlap extend_overlap(const Communication& comm, const Pattern& graph, int layers)
{
  Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Overlap", "extend")};

  const auto& roots = comm.roots();
  const auto n = static_cast<Index>(roots.size());
  DDM_CHECK(layers >= 0, "extend_overlap: number of layers must not be negative, got {}", layers);
  DDM_CHECK(graph.rows() == n && graph.cols() == n, "extend_overlap: graph is {}x{}, but the index set has size {}", graph.rows(), graph.cols(), n);

  const auto& pattern = comm.communication_pattern();
  MPI_Comm mpi_comm = pattern.communicator();

  std::unordered_map<std::int64_t, Index> global_to_local;
  global_to_local.reserve(roots.size());
  for (Index i = 0; i < n; ++i) global_to_local.emplace(roots[i].gid, i);

  std::vector<CommunicationNodes> ext_roots = roots; // this will be expanded below
  std::vector<int> layer(roots.size(), 0);           // after extension, this will have as many entries as there are local indices in the overlapping index set
                                                     // and each entry contains the "round" this index was added. So layer[i] == 0 are the original ones,
                                                     // and layer[i] == layers are the ones at the boundary of the overlapping subdomain.

  if (layers > 0) {
    // Every rank only has the edges it assembled itself, so no rank knows the complete row of a shared index: the
    // global graph is the union of the local graphs. We first build the complete (global) rows of our original
    // indices by collecting the local rows from all ranks holding the index.
    //
    // The rows store neighbours as (owner, gid): local numbers mean nothing on other ranks, and the owner is needed
    // to append the neighbour to the roots array and to ask for its row in later rounds.
    const auto local_row = [&](int i, std::vector<CommunicationNodes>& out) {
      for (auto j : graph.row(i)) out.push_back(roots[j]);
    };

    // For each of our original indices i, store a list of {gid, owner} of row[i]. Some of the rows might be incomplete
    // because peers hold part of them. They are completed in the next step below.
    std::vector<std::vector<CommunicationNodes>> rows(roots.size());
    for (Index i = 0; i < n; ++i) local_row(static_cast<int>(i), rows[i]);

    // Add the local rows of the other holders. The `reduction plan` of the Communication class contains exactly the information
    // that is necessary to exchange data at shared indices, so we use it here. Since the Communication object can only communicate
    // scalars, the actual data exchange happens in exchange_lists.
    // TODO(20261006-072703): Consider extending the Communication class to exchange lists
    const auto& shared_indices = pattern.reduction_indices();
    for (const auto& [peer, lists] : detail::exchange_lists<CommunicationNodes>(mpi_comm, shared_indices, local_row)) {
      // `lists` is a vector of vectors: for the current peer, we receive a list per shared_indices[peer].recv_idx.
      const auto& recv_idx = shared_indices.at(peer).recv_idx;
      for (std::size_t k = 0; k < lists.size(); ++k) {
        auto& row = rows[recv_idx[k]];                           // get the correct row ...
        row.insert(row.end(), lists[k].begin(), lists[k].end()); // ... and complete it with the data we received
      }
    }

    // Some of the communicated entries are already known to us, so remove the duplicates
    const auto by_gid = [](const auto& a, const auto& b) { return a.gid < b.gid; };
    const auto same_gid = [](const auto& a, const auto& b) { return a.gid == b.gid; };
    for (auto& row : rows) {
      std::sort(row.begin(), row.end(), by_gid);
      row.erase(std::unique(row.begin(), row.end(), same_gid), row.end());
    }

    // Breadth-first search, starting from all original indices
    std::vector<Index> frontier(roots.size());
    std::iota(frontier.begin(), frontier.end(), Index{0});
    for (int round = 1; round <= layers; ++round) {

      // This next block is where new indices are added to ext_roots. Everything after that only prepares the next round iteration
      // so that the following block knows all the necessary information. In the last iteration, we're done after this next block.
      std::vector<Index> added;
      for (auto i : frontier) {
        for (const auto& nb : rows[i]) {
          if (global_to_local.contains(nb.gid)) continue; // in the original set, an earlier layer or added in this round

          const Index local = static_cast<Index>(ext_roots.size()); // new indices are appended, so their local index is just ext_roots.size()
          global_to_local[nb.gid] = local;
          added.push_back(local);
          ext_roots.push_back(nb);
          layer.push_back(round);
        }
      }
      frontier = std::move(added);
      if (round == layers) break; // Everything below fills the rows array for the new `frontier` in the next round. If there is no next round, we can stop here.

      // The next round needs the rows of the indices we just added. Due to the exchange_lists step before the loop
      // we know for certain that the owner of an index by now has the complete row. So we prepare, per owner, a
      // `request` list of global_ids whose corresponding row list we want from that owner. This is a Sparse Data
      // Exchange problem in the sense of Hoefler et al. (2010) and we implement it using their NBX algorithm.
      std::map<int, std::vector<std::int64_t>> requests;
      for (auto i : frontier) {
        const auto& node = ext_roots[i];
        requests[node.rank].push_back(node.gid);
      }

      // Now the NBX exchange; after this, incoming[q] contains a list of gids that rank q wants from us
      const auto incoming = sparse_exchange(mpi_comm, detail::overlap_request_tag, requests);

      // Afterwards both sides know each other: the owners answer every requester, and every requester expects an
      // answer from every owner it asked. The answer is the length of each requested row (in the order of the
      // request), followed by all rows concatenated, so it consists of two messages.
      std::map<int, std::vector<int>> answer_counts;
      std::map<int, std::vector<CommunicationNodes>> answer_rows;
      for (const auto& [requester, gids] : incoming) {
        auto& counts = answer_counts[requester];
        auto& data = answer_rows[requester];
        for (auto gid : gids) {
          const auto it = global_to_local.find(gid);
          DDM_ASSERT(it != global_to_local.end() && it->second < n, "extend_overlap: rank {} asked for the row of gid {}, which is not owned here", requester, gid);
          const auto& row = rows[it->second];
          counts.push_back(static_cast<int>(row.size()));
          data.insert(data.end(), row.begin(), row.end());
        }
      }

      // Send both messages of every answer right away
      std::vector<MPI_Request> send_reqs;
      for (const auto& [requester, counts] : answer_counts) {
        const auto& data = answer_rows[requester];
        MPI_Isend(counts.data(), static_cast<int>(counts.size()), MPI_INT, requester, detail::overlap_counts_tag, mpi_comm, &send_reqs.emplace_back());
        MPI_Isend(data.data(), static_cast<int>(data.size() * sizeof(CommunicationNodes)), MPI_BYTE, requester, detail::overlap_data_tag, mpi_comm, &send_reqs.emplace_back());
      }

      // We asked for one row per gid, so we know how many counts to expect ...
      std::map<int, std::vector<int>> counts;
      std::vector<MPI_Request> count_reqs;
      for (const auto& [owner, gids] : requests) {
        auto& owner_counts = counts[owner];
        owner_counts.resize(gids.size());
        MPI_Irecv(owner_counts.data(), static_cast<int>(owner_counts.size()), MPI_INT, owner, detail::overlap_counts_tag, mpi_comm, &count_reqs.emplace_back());
      }
      MPI_Waitall(static_cast<int>(count_reqs.size()), count_reqs.data(), MPI_STATUSES_IGNORE);

      // ... and the counts tell us the size of the rows
      std::map<int, std::vector<CommunicationNodes>> data;
      std::vector<MPI_Request> data_reqs;
      for (const auto& [owner, owner_counts] : counts) {
        auto& owner_data = data[owner];
        owner_data.resize(std::accumulate(owner_counts.begin(), owner_counts.end(), 0));
        MPI_Irecv(owner_data.data(), static_cast<int>(owner_data.size() * sizeof(CommunicationNodes)), MPI_BYTE, owner, detail::overlap_data_tag, mpi_comm, &data_reqs.emplace_back());
      }
      MPI_Waitall(static_cast<int>(data_reqs.size()), data_reqs.data(), MPI_STATUSES_IGNORE);
      MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(), MPI_STATUSES_IGNORE);

      // Cut the concatenated rows of each owner back into the rows of the requested gids
      rows.resize(ext_roots.size());
      for (const auto& [owner, gids] : requests) {
        const auto& owner_counts = counts.at(owner);
        const auto& owner_data = data.at(owner);
        auto it = owner_data.begin();
        for (std::size_t k = 0; k < gids.size(); ++k) {
          auto& row = rows[global_to_local.at(gids[k])];
          row.assign(it, it + owner_counts[k]);
          it += owner_counts[k];
        }
      }
    }
  }

  return {std::make_shared<Communication>(mpi_comm, ext_roots), std::move(layer), layers, n};
}

/** Returns the restriction of the global matrix A to the overlapping index set, i.e. the rows of the global matrix
 *  for the indices in the set, without the columns outside the set. The local matrix is created with config (see
 *  create_local_mat()). Collective.
 *
 *  ovlp must have been created from A's communication. The local matrix of A must support host_csr().
 */
template <class T>
std::shared_ptr<LocalMat<T>> overlapping_matrix(const Dune::ParameterTree& config, const Mat<T>& A, const Overlap& ovlp)
{
  Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Overlap", "matrix")};

  DDM_CHECK(A.communication() != nullptr, "overlapping_matrix: the matrix has no communication");
  DDM_CHECK(A.rows() == ovlp.n_original, "overlapping_matrix: matrix has {} rows, but the overlap was created for {} indices", A.rows(), ovlp.n_original);

  const auto csr = A.local().host_csr();
  const auto& roots = ovlp.comm->roots();
  const auto n = ovlp.n_original;
  const auto n_ext = static_cast<Index>(roots.size());

  std::unordered_map<std::int64_t, Index> gid_to_local;
  gid_to_local.reserve(roots.size());
  for (Index i = 0; i < n_ext; ++i) gid_to_local.emplace(roots[i].gid, i);

  // Our own contributions are the rows of the local matrix
  std::vector<std::vector<std::pair<Index, T>>> rows(roots.size());
  for (Index i = 0; i < n; ++i)
    for (auto k = csr.row_ptr[i]; k < csr.row_ptr[i + 1]; ++k) rows[i].emplace_back(csr.cols[k], csr.values[k]);

  // Every rank that holds an index in its original index set has a contribution to its row. It holds the index in the
  // overlapping index set as well, so it shares the index with every other rank that needs the row.
  struct Entry {
    std::int64_t gid;
    T value;
  };
  const auto& pattern = ovlp.comm->communication_pattern();
  const auto& shared = pattern.reduction_indices();
  const auto local_row = [&](int i, std::vector<Entry>& out) {
    if (i >= n) return; // no contribution for indices we only hold in the overlap
    for (auto k = csr.row_ptr[i]; k < csr.row_ptr[i + 1]; ++k) out.push_back({roots[csr.cols[k]].gid, csr.values[k]});
  };
  for (const auto& [peer, lists] : detail::exchange_lists<Entry>(pattern.communicator(), shared, local_row)) {
    const auto& recv_idx = shared.at(peer).recv_idx;
    for (std::size_t k = 0; k < lists.size(); ++k) {
      for (const auto& e : lists[k]) {
        const auto it = gid_to_local.find(e.gid);
        if (it != gid_to_local.end()) rows[recv_idx[k]].emplace_back(it->second, e.value);
      }
    }
  }

  Pattern ovlp_pattern(n_ext, n_ext);
  for (Index i = 0; i < n_ext; ++i)
    for (const auto& [j, _] : rows[i]) ovlp_pattern.add(i, j);
  ovlp_pattern.finalize();

  auto A_ovlp = create_local_mat<T>(config, ovlp_pattern);
  std::vector<Index> cols;
  std::vector<T> values;
  for (Index i = 0; i < n_ext; ++i) {
    cols.clear();
    values.clear();
    for (const auto& [j, v] : rows[i]) {
      cols.push_back(j);
      values.push_back(v);
    }
    A_ovlp->add_values(std::span(&i, 1), cols, values);
  }
  A_ovlp->assemble();
  return A_ovlp;
}
} // namespace ddm
