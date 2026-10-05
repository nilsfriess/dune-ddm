#pragma once

#include <cstdint>

namespace ddm {
/// The MPI rank of the owner of an index and a globally unique id identifying that index.
struct CommunicationNodes {
  int rank;
  std::int64_t gid;
};
} // namespace ddm
