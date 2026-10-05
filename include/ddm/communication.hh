#pragma once

#include "communication_pattern.hh"
#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/vec/exchange_plan.hh"
#include "ddm/vec/vec.hh"
#include "dune/ddm/logger.hh"

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
#include <optional>
#include <string_view>
#include <typeindex>
#include <utility>
#include <vector>

namespace ddm {
namespace detail {

/// Which of a CommunicationPattern's two plans an exchange runs on.
enum class PlanKind : std::uint8_t {
  Broadcast,
  Reduction,
};

/** The state of one kind of exchange (broadcast or reduction) for one backend and element type:
 *  the exchange plan holding the message buffers, and the requests of an ongoing exchange.
 *
 *  The plan is created by the vector passed to the first begin(), so that the buffers live in the
 *  memory space of that vector, and reused afterwards.
 *
 *  Only one exchange may be in flight at a time: the per-peer buffers are allocated once and reused,
 *  so a second begin() before the matching end() would overwrite data still in use.
 */
template <class T>
class ExchangeState {
public:
  /// \p idxs must outlive this object
  explicit ExchangeState(const CommunicationPattern::IndexMap& idxs_)
      : idxs(idxs_)
  {
  }

  /** Sends the values of \p v at all send indices of the plan to the peers, and starts receiving
   *  the values of all recv indices from them.
   *
   *  \p v must stay valid until the matching end() call.
   */
  void begin(MPI_Comm pcomm, VecImpl<T>& v, ReductionOperation op)
  {
    if (busy()) DUNE_THROW(Dune::InvalidStateException, "an exchange on this plan is still in flight");
    if (!plan) plan = v.make_exchange_plan(idxs);

    const auto mpi_type = Dune::MPITraits<T>::getType();

    // Post the receives
    for (const auto& [peer, buffer] : plan->recv_buffers()) MPI_Irecv(buffer.data(), (int)buffer.size(), mpi_type, peer, 4, pcomm, &requests.emplace_back());

    // Post the sends
    for (const auto& [peer, buffer] : plan->pack(v)) MPI_Isend(buffer.data(), (int)buffer.size(), mpi_type, peer, 4, pcomm, &requests.emplace_back());

    target = &v;
    reduction_op = op;
  }

  /** Completes the exchange started by begin(): blocks until all data has been exchanged and the
   *  received values have been written into the vector that was passed to begin().
   */
  void end()
  {
    if (not busy()) return;
    MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    plan->unpack(*target, reduction_op);

    requests.clear();
    target = nullptr;
  }

  bool busy() const { return not requests.empty(); }

private:
  const CommunicationPattern::IndexMap& idxs;
  std::unique_ptr<ExchangePlan<T>> plan;

  VecImpl<T>* target = nullptr; ///< where end() writes the received values
  ReductionOperation reduction_op = ReductionOperation::None;

  std::vector<MPI_Request> requests;
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

/// The broadcast and reduction exchanges for one element_type
template <class T>
class Exchanger : public ExchangerBase {
public:
  explicit Exchanger(std::shared_ptr<const CommunicationPattern> pattern_)
      : pattern(std::move(pattern_))
      , broadcast_state(pattern->broadcast_indices())
      , reduction_state(pattern->reduction_indices())
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

  void broadcast_begin(Vec<T>& v)
  {
    Logger::ScopedLog sl{broadcast_begin_event};
    broadcast_state.begin(pattern->communicator(), v.impl(), ReductionOperation::None);
  }

  void reduce_begin(Vec<T>& v, ReductionOperation op)
  {
    Logger::ScopedLog sl{reduce_begin_event};
    reduction_state.begin(pattern->communicator(), v.impl(), op);
  }

  /// The dot product of v and w restricted to the indices owned by this rank, as given by owner_mask
  T owned_dot(const Vec<T>& v, const Vec<T>& w, const std::vector<std::uint8_t>& owner_mask)
  {
    // The mask and the temporary are created from the first vector, so that they have its backend
    if (!mask) {
      mask.emplace(v);
      auto view = mask->host_view(write);
      for (std::size_t i = 0; i < view.size(); ++i) view[i] = owner_mask[i] ? T{1} : T{0};
    }
    if (!masked) masked.emplace(v);

    mask->pointwise_mult(v, *masked);
    return masked->dot(w);
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
  ExchangeState<T> broadcast_state;
  ExchangeState<T> reduction_state;

  std::optional<Vec<T>> mask;   ///< 1 at the owned indices, 0 elsewhere
  std::optional<Vec<T>> masked; ///< temporary for owned_dot()

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
 * Communication can therefore serve vectors of different backends and element types alike, and
 * the expensive pattern is built exactly once.
 *
 * Both exchanges are non-blocking and return an Exchange handle; discarding it makes the call
 * blocking. Only one broadcast and one reduction can be in flight at a time, because the per-peer
 * message buffers are allocated once and reused.
 *
 * The vectors passed here must have one entry per entry of the roots array, and their element type
 * must have a `Dune::MPITraits` specialisation.
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
    Logger::get().registerOrGetEvent("Communication", "all-holders begin");
    Logger::get().registerOrGetEvent("Communication", "all-holders wait");
  }

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
  template <class T>
  Exchange broadcast(Vec<T>& v) const
  {
    check_size(v, "broadcast()");
    auto exchanger = exchanger_for(v);
    exchanger->broadcast_begin(v);
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
  template <class T>
  Exchange reduce(Vec<T>& v, ReductionOperation op = ReductionOperation::Addition) const
  {
    check_size(v, "reduce()");
    auto exchanger = exchanger_for(v);
    exchanger->reduce_begin(v, op);
    return {std::move(exchanger), detail::PlanKind::Reduction};
  }

  /// The global dot product of v and w, every index is counted once (on its owner). Collective.
  template <class T>
  void dot(const Vec<T>& v, const Vec<T>& w, T& result) const
  {
    check_size(v, "dot()");
    check_size(w, "dot()");

    Logger::get().startEvent(dot_local_event);
    result = exchanger_for(v)->owned_dot(v, w, owner_mask);
    Logger::get().endEvent(dot_local_event);

    Logger::get().startEvent(dot_allreduce_event);
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, Dune::MPITraits<T>::getType(), MPI_SUM, pattern->communicator());
    Logger::get().endEvent(dot_allreduce_event);
  }

  template <class T>
  T norm(const Vec<T>& v) const
  {
    T res{};
    dot(v, v, res);

    using std::sqrt;
    return sqrt(res);
  }

  Communicator communicator() const { return c; }

  // Convenience methods to access size and rank
  [[nodiscard]] int rank() const { return c.rank(); }
  [[nodiscard]] int size() const { return c.size(); }

  const CommunicationPattern& communication_pattern() const { return *pattern; }

  /// The pattern, to build a second Communication on the same topology without redoing the setup.
  std::shared_ptr<const CommunicationPattern> shared_pattern() const { return pattern; }

private:
  template <class T>
  void check_size(const Vec<T>& v, std::string_view op) const
  {
    DDM_CHECK(v.size() == static_cast<Index>(owner_mask.size()), "communication: {} expects a vector of size {}, got {}", op, owner_mask.size(), v.size());
  }

  /// The exchanger for the backend and element type of \p Vector, created on first use. Shared
  /// rather than owned: an Exchange holds on to it, so that waiting for an exchange stays valid
  /// even if this Communication is destroyed first.
  template <class T>
  auto exchanger_for(const Vec<T>& v) const
  {
    using Exchanger = detail::Exchanger<T>;
    ExchangerKey key{v.backend(), typeid(T)};
    auto it = exchangers.find(key);
    if (it == exchangers.end()) it = exchangers.emplace(key, std::make_shared<Exchanger>(pattern)).first;
    return std::static_pointer_cast<Exchanger>(it->second);
  }

  std::shared_ptr<const CommunicationPattern> pattern; // Holds the broadcast and reduce plan

  using ExchangerKey = std::pair<BackendId, std::type_index>;
  mutable std::map<ExchangerKey, std::shared_ptr<detail::ExchangerBase>> exchangers;

  Communicator c;
  std::vector<std::uint8_t> owner_mask;

  // The local part and the collective are timed separately: the Allreduce is the synchronisation
  // point, so it is where load imbalance elsewhere in the solver shows up.
  Logger::Event* dot_local_event{nullptr};     ///< the masked local dot product
  Logger::Event* dot_allreduce_event{nullptr}; ///< the MPI_Allreduce that combines it
};
} // namespace ddm
