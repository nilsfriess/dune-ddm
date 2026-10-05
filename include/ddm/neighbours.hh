#pragma once

#include "ddm/communication_nodes.hh"
#include "dune/ddm/logger.hh"

#include <mpi.h>
#include <set>
#include <vector>

namespace ddm::detail {
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
  std::vector<int> recv_buf(v[rank], 1);
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
} // namespace ddm::detail
