#pragma once

namespace ddm {
// How a host view (see MultiVec::host_view() and Vec::host_view()) accesses the data. For backends that keep their
// data in device memory this decides what has to be copied: read copies to the host, write copies back on release,
// read_write does both.
enum class Access { read, write, read_write };

// Tags to select the access mode of host_view() at compile time, so that read views are const
struct read_t {};
struct write_t {};
struct read_write_t {};

inline constexpr read_t read{};
inline constexpr write_t write{}; // the previous contents are not loaded, every entry must be overwritten
inline constexpr read_write_t read_write{};
} // namespace ddm
