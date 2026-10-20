#pragma once

#include "ddm/check.hh"

#include <cstddef>
#include <map>
#include <mpi.h>
#include <type_traits>
#include <vector>

namespace ddm {
/** Sends send[p] to rank p for every p in send and returns what the other ranks sent to us, by sender. Collective.
 *
 *  Unlike an exchange between known neighbours, the receivers do not need to know who sends to them: this is the
 *  "dynamic sparse data exchange" of Hoefler, Siebert, Lumsdaine, "Scalable Communication Protocols for Dynamic Sparse
 *  Data Exchange" (PPoPP 2010), implemented with their NBX algorithm. It costs the messages themselves plus one
 *  non-blocking barrier, i.e. O(log P), and no data of size O(P).
 *
 *  Empty vectors are sent as well (the message itself can carry the information), so the result has an entry for
 *  every rank that has us in its send map. Data for ourselves is copied. U is sent as bytes, so it must be trivially
 *  copyable.
 *
 *  comm must not be used concurrently for other messages with the same tag, because we receive from any source. When
 *  the function returns, all messages of this call have been received, so consecutive calls cannot interfere.
 */
template <class U>
std::map<int, std::vector<U>> sparse_exchange(MPI_Comm comm, int tag, const std::map<int, std::vector<U>>& send)
{
  static_assert(std::is_trivially_copyable_v<U>, "the data is sent as bytes");

  int rank{};
  MPI_Comm_rank(comm, &rank);

  std::map<int, std::vector<U>> recv;

  // Synchronous sends complete only once the receiver has started to receive the message. So when all of them are
  // complete, everything we sent has arrived.
  std::vector<MPI_Request> send_reqs;
  send_reqs.reserve(send.size());
  for (const auto& [p, data] : send) {
    if (p == rank) {
      recv[p] = data;
      continue;
    }
    MPI_Issend(data.data(), static_cast<int>(data.size() * sizeof(U)), MPI_BYTE, p, tag, comm, &send_reqs.emplace_back());
  }

  // Receive whatever arrives. Once our own sends are complete, we enter a non-blocking barrier. When it completes,
  // all ranks' sends are complete, so all messages have been received. We have to keep receiving while waiting for
  // the barrier, because the sends of the other ranks only complete when we receive their messages.
  MPI_Request barrier = MPI_REQUEST_NULL;
  bool in_barrier = false;
  while (true) {
    int arrived = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &arrived, &status);
    if (arrived) {
      int bytes = 0;
      MPI_Get_count(&status, MPI_BYTE, &bytes);
      DDM_ASSERT(bytes % static_cast<int>(sizeof(U)) == 0, "sparse_exchange: received {} bytes from rank {}, which is not a multiple of the element size {}", bytes, status.MPI_SOURCE, sizeof(U));
      DDM_ASSERT(!recv.contains(status.MPI_SOURCE), "sparse_exchange: received two messages from rank {}", status.MPI_SOURCE);

      auto& data = recv[status.MPI_SOURCE];
      data.resize(static_cast<std::size_t>(bytes) / sizeof(U));
      MPI_Recv(data.data(), bytes, MPI_BYTE, status.MPI_SOURCE, tag, comm, MPI_STATUS_IGNORE);
    }

    if (!in_barrier) {
      int sent = 0;
      MPI_Testall(static_cast<int>(send_reqs.size()), send_reqs.data(), &sent, MPI_STATUSES_IGNORE);
      if (sent) {
        MPI_Ibarrier(comm, &barrier);
        in_barrier = true;
      }
    }
    else {
      int done = 0;
      MPI_Test(&barrier, &done, MPI_STATUS_IGNORE);
      if (done) break;
    }
  }
  return recv;
}
} // namespace ddm
