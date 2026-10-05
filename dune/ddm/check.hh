#pragma once

#include <cstdlib>
#include <dune/common/exceptions.hh>
#include <format>
#include <iostream>
#include <utility>

inline void todo_impl(const char* file, int line, const char* message)
{
  std::cerr << file << ":" << line << ": TODO: " << message << std::endl;
  std::abort();
}

#define TODO(message) todo_impl(__FILE__, __LINE__, message)

template <class... Args>
inline void assert_impl(const char* file, int line, bool condition, std::format_string<Args...> fmt, Args&&... args)
{
  if (!condition) [[unlikely]] {
    std::cerr << file << ":" << line << ": CHECK failed: " << std::format(fmt, std::forward<Args>(args)...) << std::endl;
    std::abort();
  }
}

// DDM_ASSERT is for assertions that reveal a bug if they fire. It calls std::abort() if the assertion fails
#define DDM_ASSERT(cond, ...) assert_impl(__FILE__, __LINE__, (cond), __VA_ARGS__)

template <class... Args>
inline void check_impl(const char* file, int line, bool condition, std::format_string<Args...> fmt, Args&&... args)
{
  if (!condition) [[unlikely]]
    DUNE_THROW(Dune::InvalidStateException, "\n") << file << ":" << line << ": CHECK failed: " << std::format(fmt, std::forward<Args>(args)...);
}

// DDM_CHECK is for throws and exception instead
#define DDM_CHECK(cond, ...) check_impl(__FILE__, __LINE__, (cond), __VA_ARGS__)
