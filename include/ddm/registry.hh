#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"

#include <dune/common/parametertree.hh>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace ddm {
// Registers the built-in implementations. Called by all create_* functions, so users only need to call it if they
// access a registry directly. Safe to call more than once.
void initialize();

// Maps type names to factories for one interface (Mat<T>, VecImpl<T>, ...). Result is the (smart) pointer type the
// factories return. The factory receives the config subtree of the object (so implementations can read their own
// options) followed by Args.
//
// A factory is registered either for any backend or for one specific backend. When creating an object for a backend,
// the factory for that backend is preferred, the one for any backend is the fallback. This lets e.g. "ilu" pick the
// implementation that matches the backend of the matrix, and lets a backend provide a specialized version of a
// generic implementation.
template <class Result, class... Args>
class Registry {
public:
  using Factory = std::function<Result(const Dune::ParameterTree&, Args...)>;

  static Registry& instance()
  {
    static Registry registry;
    return registry;
  }

  // Registers a factory that works for any backend
  void add(std::string name, Factory factory) { add(std::move(name), std::nullopt, std::move(factory)); }

  // Registers a factory for one backend, or for any backend if backend is std::nullopt
  void add(std::string name, std::optional<BackendId> backend, Factory factory)
  {
    auto& by_backend = factories_[name];
    DDM_CHECK(!by_backend.contains(backend), "registry: type '{}' registered twice for backend {}", name, backend_name(backend));
    by_backend.emplace(backend, std::move(factory));
  }

  // Creates the type named by config["type"], or default_type if the key is missing, using the factory that is
  // registered for any backend. `what` is used in error messages.
  Result create(std::string_view what, const Dune::ParameterTree& config, std::string_view default_type, Args... args) const
  {
    return create_for_backend(what, config, default_type, std::nullopt, std::move(args)...);
  }

  // As create(), but prefers the factory registered for `backend` over the one registered for any backend
  Result create_for_backend(std::string_view what, const Dune::ParameterTree& config, std::string_view default_type, std::optional<BackendId> backend, Args... args) const
  {
    const auto type = config.get("type", std::string(default_type));
    const auto it = factories_.find(type);
    DDM_CHECK(it != factories_.end(), "{}: unknown type '{}', available types: {}", what, type, names());

    const auto& by_backend = it->second;
    auto factory = backend ? by_backend.find(backend) : by_backend.end();
    if (factory == by_backend.end()) factory = by_backend.find(std::nullopt);
    DDM_CHECK(factory != by_backend.end(), "{}: type '{}' is not available for backend {} (available for: {})", what, type, backend_name(backend), backend_names(by_backend));

    auto obj = factory->second(config, std::move(args)...);
    DDM_CHECK(obj != nullptr, "{}: factory for type '{}' returned nullptr", what, type);
    return obj;
  }

private:
  using ByBackend = std::map<std::optional<BackendId>, Factory>;

  static std::string_view backend_name(std::optional<BackendId> backend) { return backend ? to_string(*backend) : "any"; }

  std::string names() const
  {
    std::string result;
    for (const auto& [name, _] : factories_) result += (result.empty() ? "" : ", ") + name;
    return result;
  }

  static std::string backend_names(const ByBackend& by_backend)
  {
    std::string result;
    for (const auto& [backend, _] : by_backend) {
      if (!result.empty()) result += ", ";
      result += backend_name(backend);
    }
    return result;
  }

  std::map<std::string, ByBackend, std::less<>> factories_;
};
} // namespace ddm
