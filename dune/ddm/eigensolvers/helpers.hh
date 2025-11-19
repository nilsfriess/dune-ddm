#pragma once

#include <ostream>
#include <string>
#include <utility>

// Helper to convert floats to string because std::to_string outputs with low precision
template <typename T>
std::string to_string_with_precision(const T a_value)
{
  std::ostringstream out;
  out << a_value;
  return std::move(out).str();
}
