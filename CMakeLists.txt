cmake_minimum_required(VERSION 3.25)
project(dune-ddm
        VERSION 0.0.1
        LANGUAGES CXX C)

include(CheckFunctionExists)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COLOR_DIAGNOSTICS ON)

# ---------------------------------------------------------------------------
# Sanitizers
#
# -DDDM_SANITIZE=address,undefined instruments every target in this build tree
# (dune-ddm, the examples, the tests and, in a superbuild, the DUNE modules in
# extern/). The value is passed verbatim to -fsanitize=, so any combination the
# compiler accepts works, e.g. `address`, `address,undefined`, `thread`.
#
# The options are set here, before any add_subdirectory(), so that they are
# inherited by every subdirectory. Leave the variable empty to build without
# instrumentation.
# ---------------------------------------------------------------------------
set(DDM_SANITIZE "" CACHE STRING
  "Comma-separated list of sanitizers to enable for all targets (e.g. address,undefined); empty disables instrumentation")

if(DDM_SANITIZE)
  include(CheckCXXSourceCompiles)

  # -fsanitize has to be given to the compiler *and* the linker, so check that a
  # trivial program both compiles and links with it. The cache variable holding
  # the result is derived from the requested sanitizer list, so changing
  # DDM_SANITIZE re-runs the check instead of reusing a stale answer.
  string(MAKE_C_IDENTIFIER "DDM_SANITIZE_SUPPORTED_${DDM_SANITIZE}" DDM_SANITIZE_CHECK_VAR)
  set(CMAKE_REQUIRED_FLAGS "-fsanitize=${DDM_SANITIZE}")
  set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=${DDM_SANITIZE}")
  check_cxx_source_compiles("int main() { return 0; }" ${DDM_SANITIZE_CHECK_VAR})
  unset(CMAKE_REQUIRED_FLAGS)
  unset(CMAKE_REQUIRED_LINK_OPTIONS)

  if(NOT ${DDM_SANITIZE_CHECK_VAR})
    message(FATAL_ERROR
      "DDM_SANITIZE=${DDM_SANITIZE}: the compiler does not support -fsanitize=${DDM_SANITIZE}")
  endif()

  # -g and -fno-omit-frame-pointer are what turn the sanitizer reports into
  # readable stack traces; they come after the per-config flags, so they also
  # apply to e.g. a Release build.
  add_compile_options(-fsanitize=${DDM_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${DDM_SANITIZE})

  message(STATUS "Sanitizers enabled for all targets: ${DDM_SANITIZE}")
endif()

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

# AdaptiveCpp (SYCL). Found here rather than in a subdirectory because
# add_sycl_to_target() reads ACPP_COMPILER_LAUNCH_RULE, a plain variable that
# adaptivecpp-config.cmake sets in the scope of this find_package() call. A
# function resolves variables in its caller's scope, so a find_package() in
# examples/ would leave the variable empty in the sibling tests/ directory:
# add_sycl_to_target() would then silently set an empty compiler launch rule and
# the target would be built by plain clang++ while still linking libacpp-rt,
# which fails at runtime rather than at build time.
set(ACPP_TARGETS generic)
find_package(AdaptiveCpp REQUIRED)

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

# ---------------------------------------------------------------------------
# Warnings
#
# DUNE_DDM_ENABLE_WARNINGS=ON (the default) turns on a curated warning set for
# dune-ddm's own code: the library headers, the examples and the tests.
#
# The options are deliberately set *here*, after the extern/ subdirectories have
# been added, so they apply only to targets created below (examples, dune, doc,
# tests) and never to the vendored DUNE modules or LuaJIT. Together with the
# SYSTEM keyword on the superbuild subdirectories -- which makes the dependency
# headers arrive via -isystem -- this keeps the warnings restricted to code we
# actually maintain.
#
# Only clang and gcc are supported; on any other compiler the flags are skipped.
# ---------------------------------------------------------------------------
option(DUNE_DDM_ENABLE_WARNINGS
  "Enable a curated set of compiler warnings for dune-ddm's own targets" ON)

if(DUNE_DDM_ENABLE_WARNINGS)
  set(ddm_warnings
    -Wall
    -Wextra
    -Wcast-align         # a cast that increases the required alignment
    -Wcast-qual          # a cast that drops const/volatile
    -Wextra-semi         # stray ';' after a member function definition
    -Wformat=2           # format string checking, including security issues
    -Wimplicit-fallthrough
    -Wnon-virtual-dtor   # polymorphic base without a virtual destructor
    -Wnull-dereference
    -Woverloaded-virtual # a base class overload set hidden by a derived override
    -Wshadow)            # NB: on clang this misses a constructor parameter
                         # shadowing a member; gcc's -Wshadow catches it

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND ddm_warnings
      -Wduplicated-branches
      -Wduplicated-cond
      -Wlogical-op
      -Wmisleading-indentation)
  elseif(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "DUNE_DDM_ENABLE_WARNINGS: unsupported compiler "
      "'${CMAKE_CXX_COMPILER_ID}', no warning flags added")
    set(ddm_warnings "")
  endif()

  # One add_compile_options() call per flag: wrapping the whole list in a single
  # generator expression would make CMake reinterpret the ';' separators.
  foreach(w IN LISTS ddm_warnings)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${w}>")
  endforeach()

  if(ddm_warnings)
    message(STATUS "Warnings enabled for dune-ddm's own targets (DUNE_DDM_ENABLE_WARNINGS=ON)")
  endif()
endif()

add_subdirectory(examples)
add_subdirectory(dune)
add_subdirectory(doc)

enable_testing()
add_subdirectory(tests)

finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
