#pragma once

#include <iostream>

#include <dune/common/version.hh>

inline void hello() {
  std::cout << "DUNE Common version: " << DUNE_COMMON_VERSION << std::endl;
}