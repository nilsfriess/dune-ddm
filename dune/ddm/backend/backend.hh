#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace ddm::backend {
template <class T>
struct backend_traits {};

template <class T>
concept HasBackend = requires { typename backend_traits<T>::type; };

template <class T>
struct backend_of {
  using type = backend_traits<T>::type;
};

template <class T>
using backend_of_t = backend_of<std::remove_cvref_t<T>>::type;

/// The type of the elements a container stores, deduced from what data() points at.
///
/// This is deliberately not `T::value_type`: it is exactly the type the backend's gather/scatter
/// primitives operate on, and it works uniformly for containers that name their scalar differently
/// (ddm::Sycl::Vec calls it field_type) or not at all.
template <class T>
using element_of_t = std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<T&>().data())>>;

template <class T, class Backend>
class Buffer {
public:
  Buffer() = default;

  Buffer(typename Backend::context_type ctx, std::size_t n)
      : ctx(ctx)
      , p(Backend::template malloc<T>(ctx, n))
      , n(n)
  {
  }

  ~Buffer()
  {
    if (p) Backend::free(ctx, p);
  }

  T* data() { return p; }
  const T* data() const { return p; }

  std::size_t size() const { return n; }
  bool empty() const { return n == 0; }

  Buffer(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept
      : ctx(std::move(other.ctx))
      , p(other.p)
      , n(other.n)
  {
    other.p = nullptr;
    other.n = 0;
  }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& other) noexcept
  {
    if (this != &other) {
      if (p) Backend::free(ctx, p);
      ctx = std::move(other.ctx);
      p = other.p;
      n = other.n;
      other.p = nullptr;
      other.n = 0;
    }
    return *this;
  }

private:
  typename Backend::context_type ctx;
  T* p = nullptr;
  std::size_t n = 0;
};
} // namespace ddm::backend
