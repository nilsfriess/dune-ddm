#pragma once

#include "dune/ddm/backend/backend.hh"
#include "dune/ddm/backend/host/backend.hh"
#include "logger.hh"
#include "types.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <dune/common/exceptions.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/mpitraits.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <exception>
#include <map>
#include <memory>
#include <mpi.h>
#include <set>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ddm {
/// The MPI rank of the owner of an index and a globally unique id identifying that index.
struct CommunicationNodes {
  int rank;
  std::int64_t gid;
};

namespace detail {
inline std::vector<int> identify_neighbours(MPI_Comm comm, const std::vector<CommunicationNodes>& roots)
{
  Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Communication", "neighbours")};

  int size{};
  int rank{};
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  std::vector<int> v(size, 0);

  // Write 1 at locations of roots we know (unless we're the root); also save the ranks we reference
  std::set<int> root_ranks;
  for (const auto& r : roots) {
    if (r.rank != rank) {
      v[r.rank] = 1;
      root_ranks.insert(r.rank);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, v.data(), size, MPI_INT, MPI_SUM, comm);

  // Now v[rank] holds the number of incoming senders; post as many MPI_ANY_SOURCE receives as there are senders
  std::vector<MPI_Request> recv_reqs(v[rank]);
  std::vector<int> recv_buf(v[rank], 1); // one dedicated buffer per receive: overlapping buffers are a formal MPI violation
  for (int i = 0; i < v[rank]; ++i) MPI_Irecv(recv_buf.data() + i, 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, &recv_reqs[i]);

  // Post the corresponding sends
  std::vector<MPI_Request> send_reqs(root_ranks.size());
  int i = 0;
  int ping = 1; // payload is ignored; the message itself is what carries the information
  for (auto root_rank : root_ranks) MPI_Issend(&ping, 1, MPI_INT, root_rank, 0, comm, &send_reqs[i++]);

  // Wait for the sends and receives to finish
  std::vector<MPI_Status> statuses(v[rank]);
  MPI_Waitall((int)send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall((int)recv_reqs.size(), recv_reqs.data(), statuses.data());

  // Read off the senders rank numbers
  std::vector<int> neighbours(v[rank] + root_ranks.size());
  auto it = std::copy(root_ranks.begin(), root_ranks.end(), neighbours.begin());
  std::transform(statuses.begin(), statuses.end(), it, [](const auto& s) { return s.MPI_SOURCE; });
  std::sort(neighbours.begin(), neighbours.end());
  neighbours.erase(std::unique(neighbours.begin(), neighbours.end()), neighbours.end());
  return neighbours;
}
} // namespace detail

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
      : neighbours_(detail::identify_neighbours(comm_, roots))
  {
    MPI_Comm_dup(comm_, &comm);
    logger::trace_all("CommunicationPattern() neighbours: {}", logger::join(neighbours_));

    // Global ids are unique across all ranks, so the id alone identifies an entry of our roots
    // array. Both plan builders need to translate ids they receive back into local indices.
    std::unordered_map<std::int64_t, int> gid_to_local;
    gid_to_local.reserve(roots.size());
    for (std::size_t i = 0; i < roots.size(); ++i) gid_to_local[roots[i].gid] = (int)i;

    build_broadcast_plan(roots, gid_to_local);
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

    // ---- Owner-side discovery ---------------------------------------------------------------
    // On the owner, broadcast_idxs[p].send_idx holds the owner-local ids of all indices that leaf
    // p copies. Invert it to find, per owned index, all ranks holding copies of it.
    std::unordered_map<int, std::set<int>> holders; // my local idx -> ranks holding copies of it
    for (const auto& [peer, indices] : broadcast_idxs)
      for (auto idx : indices.send_idx) holders[idx].insert(peer);

    // ---- Owner -> leaf introductions --------------------------------------------------------
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

    // ---- Exchange introduction counts (tag 1) ----------------------------------------------
    std::unordered_map<int, int> nsend; // number of pairs we send per neighbour
    for (auto n : neighbours_) nsend[n] = 0;
    for (const auto& [peer, data] : intros_siblings) nsend.at(peer) = (int)data.size();

    std::unordered_map<int, int> nrecv;
    std::vector<MPI_Request> reqs;
    reqs.reserve(4 * neighbours_.size());
    for (auto neighbour : neighbours_) MPI_Irecv(&(nrecv[neighbour]), 1, MPI_INT, neighbour, 1, comm, &reqs.emplace_back());
    for (auto neighbour : neighbours_) MPI_Isend(&nsend[neighbour], 1, MPI_INT, neighbour, 1, comm, &reqs.emplace_back());
    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // ---- Exchange introduction data (tag 2) -------------------------------------------------
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

    // ---- Translate received introductions to local indices ---------------------------------
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

    // ---- Assemble the symmetric per-peer index lists ---------------------------------------
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
      logger::trace_all("reduction plan peer {}: indices [{}]", peer, logger::join(idxs.recv_idx));
      // Reduction is symmetric: we send the same indices we receive.
      idxs.send_idx = idxs.recv_idx;
    }
  }

  IndexMap broadcast_idxs; ///< Indices to communicate from owners to copies (= roots to leaves)
  IndexMap reduction_idxs; ///< Indices to communicate between all holders of a shared index

  MPI_Comm comm{};
  std::vector<int> neighbours_;
};

namespace detail {
/// Which of a CommunicationPattern's two plans an exchange runs on.
enum class PlanKind : std::uint8_t {
  Broadcast,
  Reduction,
};

/** The device-side mirror of one CommunicationPattern::IndexMap, together with the message buffers
 *  and the state of an ongoing exchange.
 *
 *  This is the only part of a Communication that has to know a type: the send and receive buffers
 *  persist between begin() and end(), so they must be typed. The type they need is the element type
 *  of the data being exchanged and the backend it lives on — not the vector class it came from.
 *
 *  Only one exchange may be in flight at a time: the per-peer buffers are allocated once and reused,
 *  so a second begin() before the matching end() would overwrite data still in use.
 */
template <class Backend, class T>
class ExchangeState {
public:
  using Context = typename Backend::context_type;

  /** Sends the values of \p data at all send indices of the plan to the peers holding copies of
   *  them, and starts receiving the values of all recv indices from their owners.
   *
   *  The exchange overlaps with the caller's computation; \p data must stay valid and must not be
   *  reallocated before the matching end() call.
   */
  void begin(const CommunicationPattern::IndexMap& host_idxs, MPI_Comm pcomm, Context new_ctx, T* data, ReductionOperation op)
  {
    if (busy()) DUNE_THROW(Dune::InvalidStateException, "an exchange on this plan is still in flight");

    // The context (e.g. a sycl::queue) is taken from the caller's data and cached: buffers may
    // outlive the vector they were first created for
    if (ctx_set && new_ctx != ctx) DUNE_THROW(Dune::InvalidStateException, "vectors used with the same Communication must live on the same context (e.g. queue)");
    ctx = new_ctx;
    ctx_set = true;

    if (!indices_on_device) {
      upload_indices(host_idxs);
      indices_on_device = true;
    }

    auto mpi_type = Dune::MPITraits<T>::getType();
    requests.reserve(2 * idxs.size());

    // Post the receives
    for (const auto& [peer, indices] : idxs) {
      if (indices.recv_idx.empty()) continue;
      if (!recv_bufs.contains(peer)) recv_bufs[peer] = Backend::template make_buffer<T>(ctx, indices.recv_idx.size());
      MPI_Irecv(recv_bufs[peer].data(), (int)indices.recv_idx.size(), mpi_type, peer, 4, pcomm, &requests.emplace_back());
    }

    // Pack every send buffer first ...
    for (const auto& [peer, indices] : idxs) {
      if (indices.send_idx.empty()) continue;
      if (!send_bufs.contains(peer)) send_bufs[peer] = Backend::template make_buffer<T>(ctx, indices.send_idx.size());
      Backend::gather(ctx, data, indices.send_idx, send_bufs[peer].data());
    }

    // ... then wait for the gathers to complete before handing the buffers to MPI. On an
    // accelerator backend gather() only enqueues a kernel, and MPI knows nothing about the queue,
    // so without this the sends could read buffers that have not been written yet.
    Backend::sync(ctx);

    // Post the sends
    for (const auto& [peer, indices] : idxs) {
      if (indices.send_idx.empty()) continue;
      MPI_Isend(send_bufs[peer].data(), (int)indices.send_idx.size(), mpi_type, peer, 4, pcomm, &requests.emplace_back());
    }

    target = data;
    reduction_op = op;
  }

  /** Completes the exchange started by begin(): blocks until all data has been exchanged and the
   *  received values have been written into the data that was passed to begin().
   */
  void end()
  {
    if (not busy()) return; // nothing was started, or it has already been completed

    MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    for (const auto& [peer, indices] : idxs) {
      if (indices.recv_idx.empty()) continue;
      switch (reduction_op) {
        case ReductionOperation::None: Backend::scatter(ctx, recv_bufs[peer].data(), indices.recv_idx, target); break;
        case ReductionOperation::Addition: Backend::template scatter_reduce<ReductionOperation::Addition>(ctx, recv_bufs[peer].data(), indices.recv_idx, target); break;
      }
    }

    // The scatters are enqueued, but end() promises that the data is in place when it returns
    Backend::sync(ctx);

    requests.clear();
    target = nullptr;
  }

  bool busy() const { return target != nullptr; }

private:
  template <class U>
  using BackendBuffer = typename Backend::template buffer_type<U>;

  struct Indices {
    BackendBuffer<int> send_idx; ///< indices we send to peer
    BackendBuffer<int> recv_idx; ///< indices we receive from peer
  };

  void upload_indices(const CommunicationPattern::IndexMap& host_idxs)
  {
    for (const auto& [peer, host] : host_idxs) idxs.emplace(peer, Indices{Backend::make_buffer_from_host(ctx, host.send_idx), Backend::make_buffer_from_host(ctx, host.recv_idx)});
  }

  std::unordered_map<int, Indices> idxs;

  bool indices_on_device = false;
  T* target = nullptr;                                        ///< where end() writes the received values; also marks an exchange as in flight
  ReductionOperation reduction_op = ReductionOperation::None; ///< how end() combines them with what is already there

  std::unordered_map<int, BackendBuffer<T>> send_bufs; ///< gather buffers for outgoing data, reused between exchanges
  std::unordered_map<int, BackendBuffer<T>> recv_bufs; ///< buffers for incoming data, reused between exchanges

  std::vector<MPI_Request> requests;
  Context ctx{}; ///< context the buffers were allocated with, cached from the data at first begin()
  bool ctx_set = false;
};

/// Type-erased handle to an Exchanger, so that one Communication can hold exchangers for any number
/// of (backend, element type) combinations, and so that an Exchange can complete an exchange
/// without naming the type it runs on.
struct ExchangerBase {
  ExchangerBase() = default;
  ExchangerBase(const ExchangerBase&) = delete;
  ExchangerBase& operator=(const ExchangerBase&) = delete;
  virtual ~ExchangerBase() = default;

  /// Completes the exchange in flight on \p plan. Reached from ~Exchange(), so it must not throw.
  virtual void finish(PlanKind plan) noexcept = 0;
};

/// The broadcast and reduction exchanges for one (backend, element type) combination, carried out
/// on the topology of a shared CommunicationPattern.
template <class Backend, class T>
class Exchanger : public ExchangerBase {
public:
  using Context = typename Backend::context_type;

  explicit Exchanger(std::shared_ptr<const CommunicationPattern> pattern_)
      : pattern(std::move(pattern_))
      , broadcast_begin_event{Logger::get().registerOrGetEvent("Communication", "broadcast begin")}
      , broadcast_wait_event{Logger::get().registerOrGetEvent("Communication", "broadcast wait")}
      , reduce_begin_event{Logger::get().registerOrGetEvent("Communication", "reduce begin")}
      , reduce_wait_event{Logger::get().registerOrGetEvent("Communication", "reduce wait")}
  {
  }

  ~Exchanger() override
  {
    // An Exchange keeps its exchanger alive until it has been waited for, so getting here with an
    // exchange still in flight means one was started without an Exchange to complete it.
    if (broadcast_state.busy() or reduction_state.busy()) logger::error("Communication was destroyed while an exchange was in flight");
  }

  void broadcast_begin(Context ctx, T* data)
  {
    Logger::ScopedLog sl{broadcast_begin_event};
    broadcast_state.begin(pattern->broadcast_indices(), pattern->communicator(), ctx, data, ReductionOperation::None);
  }

  void reduce_begin(Context ctx, T* data, ReductionOperation op)
  {
    Logger::ScopedLog sl{reduce_begin_event};
    reduction_state.begin(pattern->reduction_indices(), pattern->communicator(), ctx, data, op);
  }

  void finish(PlanKind plan) noexcept override
  {
    // This is where the exchange actually costs time: begin() only posts the messages, the
    // MPI_Waitall and the unpacking happen here.
    Logger::ScopedLog sl{plan == PlanKind::Broadcast ? broadcast_wait_event : reduce_wait_event};

    // There is nobody to report a failure to: we are on the way out of ~Exchange(). MPI's default
    // error handler aborts rather than returns, so in practice this catches Backend::sync().
    try {
      if (plan == PlanKind::Broadcast) broadcast_state.end();
      else reduction_state.end();
    }
    catch (const std::exception& e) {
      logger::error("failed to complete an exchange: {}", e.what());
    }
    catch (...) {
      logger::error("failed to complete an exchange");
    }
  }

private:
  std::shared_ptr<const CommunicationPattern> pattern;
  ExchangeState<Backend, T> broadcast_state;
  ExchangeState<Backend, T> reduction_state;

  // Registered once per exchanger, so that the hot path only dereferences a pointer. The events
  // themselves are shared by name with every other exchanger (and pre-registered by Communication).
  Logger::Event* broadcast_begin_event{nullptr}; ///< packing and posting of a broadcast
  Logger::Event* broadcast_wait_event{nullptr};  ///< waiting for and unpacking a broadcast
  Logger::Event* reduce_begin_event{nullptr};    ///< packing and posting of a reduction
  Logger::Event* reduce_wait_event{nullptr};     ///< waiting for and unpacking a reduction
};
} // namespace detail

/** @brief A data exchange that has been started and has not been completed yet.
 *
 *  Returned by Communication::broadcast() and Communication::reduce(). wait() blocks until the
 *  exchanged values are in place, and the destructor calls wait(), so discarding the handle turns
 *  the call into a blocking one:
 *
 *  @code
 *  auto exchange = comm.reduce(v);  // starts the exchange and returns
 *  ...                              // overlap some computation with it
 *  exchange.wait();                 // v is consistent from here on
 *
 *  comm.reduce(v);                  // the temporary dies at the semicolon, so this blocks
 *  @endcode
 *
 *  The handle keeps everything the exchange needs alive, so it may outlive the Communication it
 *  came from. It must not outlive the data that was passed to the call that produced it.
 *
 *  A started exchange cannot be abandoned: it is collective, so every rank has to reach the wait.
 *  In particular, an exception thrown between the call and the wait completes the exchange from the
 *  destructor while unwinding, which deadlocks unless the other ranks get there as well.
 */
class Exchange {
public:
  /// Blocks until the exchange has completed and its values have been written. Idempotent.
  void wait()
  {
    if (!exchanger) return;
    exchanger->finish(plan);
    exchanger.reset();
  }

  ~Exchange() { wait(); }

  /// Moving hands the exchange over: the moved-from handle has nothing left to wait for.
  Exchange(Exchange&&) noexcept = default;

  Exchange& operator=(Exchange&& other) noexcept
  {
    if (this != &other) {
      wait(); // whatever we were holding still has to be completed
      exchanger = std::move(other.exchanger);
      plan = other.plan;
    }
    return *this;
  }

  Exchange(const Exchange&) = delete;
  Exchange& operator=(const Exchange&) = delete;

private:
  friend class Communication;

  Exchange(std::shared_ptr<detail::ExchangerBase> exchanger_, detail::PlanKind plan_)
      : exchanger(std::move(exchanger_))
      , plan(plan_)
  {
  }

  std::shared_ptr<detail::ExchangerBase> exchanger; ///< null once waited for or moved from
  detail::PlanKind plan;
};

/**
 * @brief Exchange of data between the owners ("roots") and the copies ("leaves") of an index.
 *
 * A Communication broadcasts the values of all indices from their owners to all ranks holding
 * copies of them, and reduces the values of shared indices across all their holders.
 *
 * The class is not templated on a vector type: it holds the (vector-agnostic) CommunicationPattern
 * and creates, on demand, one exchanger per backend and element type it is used with. A single
 * Communication can therefore serve `std::vector<int>`, `Dune::BlockVector<FieldVector<double,1>>`
 * and `ddm::Sycl::Vec<double>` alike, and the expensive pattern is built exactly once.
 *
 * Both exchanges are non-blocking and return an Exchange handle; discarding it makes the call
 * blocking. Only one broadcast and one reduction can be in flight at a time, because the per-peer
 * message buffers are allocated once and reused.
 *
 * A vector passed here needs a `backend_traits` specialisation and `data()`; the element type is
 * deduced from `data()` and must have a `Dune::MPITraits` specialisation.
 */
class Communication {
  struct Communicator {
    Communicator(MPI_Comm comm)
    {
      MPI_Comm_size(comm, &size_);
      MPI_Comm_rank(comm, &rank_);
    }

    int size() const { return size_; }
    int rank() const { return rank_; }

    int size_{};
    int rank_{};
  };

public:
  Communication(MPI_Comm comm, const std::vector<CommunicationNodes>& roots)
      : pattern(std::make_shared<const CommunicationPattern>(comm, roots))
      , c(comm)
      , owner_mask(roots.size())
      , dot_local_event{Logger::get().registerOrGetEvent("Communication", "dot (local)")}
      , dot_allreduce_event{Logger::get().registerOrGetEvent("Communication", "dot (allreduce)")}
  {
    std::transform(roots.begin(), roots.end(), owner_mask.begin(), [&](const auto& r) {
      // Return 1 if we're the owner, zero otherwise
      return r.rank == c.rank() ? 1 : 0;
    });

    // The exchangers are created lazily, on the first exchange of a given backend and element type,
    // and register these events themselves. Logger::report() is collective and reduces over the
    // events in registration order, so pin that order here, where we are on a collective path
    // anyway. The exchangers pick the same events up again via registerOrGetEvent().
    Logger::get().registerOrGetEvent("Communication", "broadcast begin");
    Logger::get().registerOrGetEvent("Communication", "broadcast wait");
    Logger::get().registerOrGetEvent("Communication", "reduce begin");
    Logger::get().registerOrGetEvent("Communication", "reduce wait");
  }

  // /// Builds a Communication on an existing pattern, sharing its topology and its communicator.
  // explicit Communication(std::shared_ptr<const CommunicationPattern> pattern_)
  //     : pattern(std::move(pattern_))
  // {
  // }

  Communication(const Communication&) = delete;
  Communication& operator=(const Communication&) = delete;

  // The exchangers are held behind pointers, so moving does not invalidate the buffers an ongoing
  // exchange handed to MPI.
  Communication(Communication&&) = default;
  Communication& operator=(Communication&&) = default;

  /** Broadcasts the values of all indices from their owners to all ranks holding copies of them.
   *
   *  Returns a handle to the started exchange: call wait() on it once the values are needed, or
   *  discard it to make the call blocking. v must stay valid and must not be reallocated until the
   *  exchange has been waited for.
   *
   *  Throws if another broadcast on this Communication is still in flight.
   */
  template <class Vector>
  Exchange broadcast(Vector& v) const
  {
    using Backend = backend::backend_of_t<Vector>;
    auto exchanger = exchanger_for<Vector>();
    exchanger->broadcast_begin(Backend::context(v), v.data());
    return {std::move(exchanger), detail::PlanKind::Broadcast};
  }

  /** Reduces the values of shared indices across all their holders: the values at all shared
   *  indices are sent around to owners and copy holders and combined there with \p op.
   *
   *  Returns a handle to the started exchange: call wait() on it once the values are needed, or
   *  discard it to make the call blocking. v must stay valid and must not be reallocated until the
   *  exchange has been waited for.
   *
   *  Throws if another reduction on this Communication is still in flight.
   */
  template <class Vector>
  Exchange reduce(Vector& v, ReductionOperation op = ReductionOperation::Addition) const
  {
    using Backend = backend::backend_of_t<Vector>;
    auto exchanger = exchanger_for<Vector>();
    exchanger->reduce_begin(Backend::context(v), v.data(), op);
    return {std::move(exchanger), detail::PlanKind::Reduction};
  }

  // Compatibility functions for DUNE
  template <class Vector>
  void copyOwnerToAll(const Vector& v, Vector& w) const
  {
    if (&v != &w) DUNE_THROW(Dune::Exception, "The compatibility method copyOwnerToAll is only supported when destination and source vector coincide");

    broadcast(v);
  }

  template <class Vector>
  void addOwnerCopyToOwnerCopy(const Vector& v, Vector& w) const
  {
    if (&v != &w) DUNE_THROW(Dune::Exception, "The compatibility method copyOwnerToAll is only supported when destination and source vector coincide");

    reduce(v);
  }

  template <class Vector>
  void dot(const Vector& v, const Vector& w, typename Vector::field_type& result) const
  {
    // TOOD: This assumes that the vector is ddm::Sycl::Vec
    static Vector mask = Vector::from_host_vector(v.queue(), owner_mask);

    Logger::get().startEvent(dot_local_event);
    result = v.masked_dot(mask, w);
    Logger::get().endEvent(dot_local_event);

    Logger::get().startEvent(dot_allreduce_event);
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, Dune::MPITraits<typename Vector::field_type>::getType(), MPI_SUM, pattern->communicator());
    Logger::get().endEvent(dot_allreduce_event);
  }

  template <class Vector>
  typename Vector::field_type norm(const Vector& v) const
  {
    typename Vector::field_type res{};
    dot(v, v, res);

    using std::sqrt;
    return sqrt(res);
  }

  Communicator communicator() const { return c; }

  const CommunicationPattern& communication_pattern() const { return *pattern; }

  /// The pattern, to build a second Communication on the same topology without redoing the setup.
  std::shared_ptr<const CommunicationPattern> shared_pattern() const { return pattern; }

private:
  /// The exchanger for the backend and element type of \p Vector, created on first use. Shared
  /// rather than owned: an Exchange holds on to it, so that waiting for an exchange stays valid
  /// even if this Communication is destroyed first.
  template <class Vector>
  auto exchanger_for() const
  {
    using Exchanger = detail::Exchanger<backend::backend_of_t<Vector>, backend::element_of_t<Vector>>;

    auto key = std::type_index(typeid(Exchanger));
    auto it = exchangers.find(key);
    if (it == exchangers.end()) it = exchangers.emplace(key, std::make_shared<Exchanger>(pattern)).first;
    return std::static_pointer_cast<Exchanger>(it->second);
  }

  std::shared_ptr<const CommunicationPattern> pattern;
  mutable std::unordered_map<std::type_index, std::shared_ptr<detail::ExchangerBase>> exchangers;

  Communicator c;
  std::vector<std::uint8_t> owner_mask;

  // The local part and the collective are timed separately: the Allreduce is the synchronisation
  // point, so it is where load imbalance elsewhere in the solver shows up.
  Logger::Event* dot_local_event{nullptr};     ///< the masked local dot product
  Logger::Event* dot_allreduce_event{nullptr}; ///< the MPI_Allreduce that combines it
};

/** Builds the roots array for a Dune::OwnerOverlapCopyCommunication.
 *
 *  It is filled entirely from data that ISTL already holds, no communication is needed:
 *
 *  - The global id of a local index is the global index of its entry in the parallel index set.
 *  - The owner rank of an index we own ourselves is our own rank; the index set records this in
 *    the attribute of the local index.
 *  - For the remaining indices the owner is found in the remote indices: a RemoteIndex stores
 *    the attribute the shared index carries *on the remote rank*, so the owner of a copy is
 *    exactly the neighbour whose remote attribute is `owner`.
 *
 *  @pre oocc.remoteIndices() has been built (rebuild<...>()) and every shared index is visible
 *       in it, i.e. it was rebuilt with ignorePublic == true or all shared indices are flagged
 *       public. Otherwise the owner of a copy cannot be determined and we throw.
 */
template <class T1, class T2>
std::vector<CommunicationNodes> make_roots_from_dune(const Dune::OwnerOverlapCopyCommunication<T1, T2>& oocc)
{
  Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Communication", "roots from Dune")};

  using Attribute = Dune::OwnerOverlapCopyAttributeSet;

  const auto& pis = oocc.indexSet();
  const int rank = oocc.communicator().rank();

  std::vector<CommunicationNodes> roots(pis.size(), CommunicationNodes{-1, -1});

  for (const auto& pair : pis) {
    auto local = pair.local().local();
    roots[local].gid = (std::int64_t)pair.global();
    if (pair.local().attribute() == Attribute::owner) roots[local].rank = rank;
  }

  // For an OwnerOverlapCopyCommunication the source and the destination index set of the remote
  // indices are the same object, so the two lists of the per-peer pair are the same list. We use
  // .second (the receive list) here.
  for (const auto& [peer, lists] : oocc.remoteIndices()) {
    for (const auto& remote : *lists.second) {
      if (remote.attribute() != Attribute::owner) continue;

      auto local = remote.localIndexPair().local().local();
      if (roots[local].rank != -1)
        DUNE_THROW(Dune::InvalidStateException, "local index " << local << " (global " << roots[local].gid << ") is claimed by more than one owner (" << roots[local].rank << " and " << peer << ")");
      roots[local].rank = peer;
    }
  }

  for (std::size_t i = 0; i < roots.size(); ++i)
    if (roots[i].rank == -1)
      DUNE_THROW(Dune::InvalidStateException, "no owner found for local index " << i << " (global " << roots[i].gid << "); are the remote indices built and is the index flagged public?");

  return roots;
}

/** Builds a Communication from a Dune::OwnerOverlapCopyCommunication. See make_roots_from_dune()
 *  for how the roots array is derived and what is required of \p oocc.
 */
template <class T1, class T2>
Communication make_communication_from_dune(const Dune::OwnerOverlapCopyCommunication<T1, T2>& oocc)
{
  return Communication(oocc.communicator(), make_roots_from_dune(oocc));
}
} // namespace ddm
