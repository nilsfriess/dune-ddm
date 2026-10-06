#pragma once

#include "ddm/communication_nodes.hh"
#include "ddm/sparse_exchange.hh"
#include "dune/ddm/logger.hh"

#include <algorithm>
#include <map>
#include <mpi.h>
#include <vector>

namespace ddm::detail {
// Tag of the messages of identify_neighbours()
inline constexpr int neighbours_tag = 0;

/** Returns the ranks we exchange data with: the owners of the indices we hold copies of, and the ranks that hold
 *  copies of indices we own. We know the former from the roots array; we learn the latter by sending a message to
 *  every owner we reference. Collective.
 */
inline std::vector<int> identify_neighbours(MPI_Comm comm, const std::vector<CommunicationNodes>& roots)
{
  Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Communication", "neighbours")};

  int rank{};
  MPI_Comm_rank(comm, &rank);

  // The messages are empty, receiving one is what carries the information
  std::map<int, std::vector<char>> owners;
  for (const auto& r : roots)
    if (r.rank != rank) owners[r.rank];

  const auto referencing = sparse_exchange(comm, neighbours_tag, owners);

  std::vector<int> neighbours;
  for (const auto& [p, _] : owners) neighbours.push_back(p);
  for (const auto& [p, _] : referencing) neighbours.push_back(p);
  std::sort(neighbours.begin(), neighbours.end());
  neighbours.erase(std::unique(neighbours.begin(), neighbours.end()), neighbours.end());
  return neighbours;
}
} // namespace ddm::detail
