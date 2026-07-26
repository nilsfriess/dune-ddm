cmake_minimum_required(VERSION 3.25)
project(dune-ddm
        VERSION 0.0.1
        LANGUAGES CXX C)

include(CheckFunctionExists)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COLOR_DIAGNOSTICS ON)

# ---------------------------------------------------------------------------
# Superbuild
#
# When enabled (the default), the required DUNE modules are built in-tree from
# the git submodules in extern/, so no dunecontrol invocation is needed:
#
#     git clone --recursive <url> && cd dune-ddm && mkdir build && cd build
#     cmake .. && make
#
# Disable with -DDUNE_DDM_SUPERBUILD=OFF to instead build against DUNE modules
# provided externally (e.g. via dunecontrol or a system installation).
# ---------------------------------------------------------------------------
option(DUNE_DDM_SUPERBUILD
  "Build the required DUNE modules in-tree from the submodules in extern/ (no dunecontrol needed)"
  ON)

if(DUNE_DDM_SUPERBUILD)
  # Make imported targets created in one module visible to the others and let
  # find_package() resolve modules that were added as subdirectories.
  set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

  # Let find_package() locate the in-tree build-tree configs (build/extern/<module>).
  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/extern)

  # Python bindings are off by default to keep a plain `cmake ..` friction-free.
  # Override with -DDUNE_ENABLE_PYTHONBINDINGS=ON if you need them.
  set(DUNE_ENABLE_PYTHONBINDINGS OFF CACHE BOOL "Enable Python bindings for the DUNE modules")

  # Add the DUNE modules in dependency order. Each module configures itself and
  # exports its targets, which the following modules (and dune-ddm) pick up.
  #
  # The SYSTEM keyword (CMake >= 3.25) marks every target created in these
  # subdirectories as a system target, so their interface include directories
  # are consumed via -isystem. That silences the dependencies' header warnings
  # under -Wall -Wextra while keeping warnings for dune-ddm's own code.
  add_subdirectory(extern/dune-common         SYSTEM)
  add_subdirectory(extern/dune-geometry       SYSTEM)
  add_subdirectory(extern/dune-uggrid         SYSTEM)
  add_subdirectory(extern/dune-grid           SYSTEM)
  add_subdirectory(extern/dune-localfunctions SYSTEM)
  add_subdirectory(extern/dune-istl           SYSTEM)
  add_subdirectory(extern/dune-typetree       SYSTEM)
  add_subdirectory(extern/dune-functions      SYSTEM)
  add_subdirectory(extern/dune-pdelab         SYSTEM)
endif()

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/extern)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/extern/dune-common/cmake/modules)

if(NOT (dune-common_DIR OR dune-common_ROOT OR
      "${CMAKE_PREFIX_PATH}" MATCHES ".*dune-common.*"))
  string(REPLACE  ${PROJECT_NAME} dune-common dune-common_DIR
    ${PROJECT_BINARY_DIR})
endif()

#find dune-common and set the module path
find_package(dune-common REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules"
  ${dune-common_MODULE_PATH})

#include the dune macros
include(DuneMacros)

# start a dune project with information from dune.module
dune_project()
dune_enable_all_packages()

if (dune-common_VERSION VERSION_GREATER_EQUAL 2.10)
  dune_add_library(duneddm INTERFACE EXPORT_NAME DDM)
else()
  dune_add_library(duneddm INTERFACE EXPORT_NAME Dune::DDM)
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# SuiteSparse (UMFPACK/CHOLMOD) is a public dependency: dune-ddm's headers
# include <umfpack.h> unconditionally. The find_package call lives in
# cmake/modules/DuneDdmMacros.cmake so that it also runs in downstream consumers
# of the installed Dune::DDM target (Dune auto-includes that file). Here we only
# attach the imported target to the exported interface, so its include dirs and
# libraries propagate to anything linking Dune::DDM.
target_link_libraries(duneddm INTERFACE SuiteSparse::SuiteSparse)

# optionally find STRUMPACK
find_package(STRUMPACK CONFIG)

find_package(Eigen3 NO_MODULE REQUIRED)

# Install dune-ddm's own cmake modules (DuneDdmMacros.cmake) for consumers.
add_subdirectory(cmake/modules)

# Set up external dependencies that are included as git submodules
add_subdirectory(extern)

add_subdirectory(examples)
add_subdirectory(dune)

enable_testing()
add_subdirectory(tests)

finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
