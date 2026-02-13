cmake_minimum_required(VERSION 3.24)
project(dune-ddm
        VERSION 0.0.1
        LANGUAGES CXX C)

include(CheckFunctionExists)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COLOR_DIAGNOSTICS ON)

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
find_package(SuiteSparse REQUIRED COMPONENTS UMFPACK CHOLMOD)

# optionally find STRUMPACK
find_package(STRUMPACK CONFIG)

find_package(Eigen3 NO_MODULE REQUIRED)

# Set up external dependencies that are included as git submodules
add_subdirectory(extern)

add_subdirectory(examples)
add_subdirectory(dune)

finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
