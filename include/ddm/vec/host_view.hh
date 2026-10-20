#pragma once

#include "ddm/check.hh"

#include <cstddef>
#include <exception>
#include <iostream>
#include <span>
#include <type_traits>
#include <utility>

namespace ddm {
template <class T>
class Vec;

// How a view accesses the data of a vector. For backends that keep their data in device memory this decides what has
// to be copied: read copies to the host, write copies back on release, read_write does both.
enum class Access { read, write, read_write };

// Tags to select the access mode of Vec::host_view() at compile time, so that read views are const
struct read_t {};
struct write_t {};
struct read_write_t {};

inline constexpr read_t read{};
inline constexpr write_t write{};       // the previous contents are not loaded, every entry must be overwritten
inline constexpr read_write_t read_write{};

// Scoped access to the data of a Vec in host memory, U is T for writable and const T for read-only views. Released
// (i.e. written back if needed) on destruction or by an explicit call to release(). While a view is open, the vector
// cannot be used otherwise.
template <class U>
class HostView {
  using T = std::remove_const_t<U>;

public:
  using value_type = T;
  using element_type = U;

  HostView(const HostView&) = delete;
  HostView& operator=(const HostView&) = delete;
  HostView& operator=(HostView&&) = delete;

  HostView(HostView&& other) noexcept
      : vec_(std::exchange(other.vec_, nullptr))
      , data_(other.data_)
      , mode_(other.mode_)
  {
  }

  ~HostView()
  {
    // Destructors must not throw, call release() explicitly to handle errors (e.g. a failing copy back to the device)
    try {
      release();
    }
    catch (const std::exception& e) {
      std::cerr << "ddm: releasing a host view failed: " << e.what() << std::endl;
      std::terminate();
    }
  }

  // Ends the access, the view must not be used afterwards
  void release()
  {
    if (vec_ == nullptr) return;
    std::exchange(vec_, nullptr)->release_host_view(mode_);
  }

  U& operator[](std::size_t i) const
  {
    DDM_ASSERT(vec_ != nullptr, "host view: used after release()");
    DDM_ASSERT(i < data_.size(), "host view: index {} out of range for size {}", i, data_.size());
    return data_[i];
  }

  std::size_t size() const { return data_.size(); }
  U* data() const { return data_.data(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
  std::span<U> span() const { return data_; }

private:
  friend class Vec<T>;

  HostView(const Vec<T>* vec, std::span<U> data, Access mode)
      : vec_(vec)
      , data_(data)
      , mode_(mode)
  {
  }

  const Vec<T>* vec_;
  std::span<U> data_;
  Access mode_;
};
} // namespace ddm
