#pragma once

#include <string_view>

namespace ddm {
// Identifies the data layout/memory space of an object. Objects can only be combined (e.g. in Mat::mv) if their
// backends match, which lets an implementation static_cast its arguments to its own concrete types.
enum class BackendId { istl };

constexpr std::string_view to_string(BackendId backend)
{
  switch (backend) {
    case BackendId::istl: return "istl";
  }
  return "unknown";
}
} // namespace ddm
