cmake_minimum_required(VERSION 3.24)
project(dune-ddm
        VERSION 0.0.1
        LANGUAGES CXX C Fortran)

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
if (TARGET Eigen3::Eigen)
  target_include_directories(duneddm INTERFACE
                                     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/spectra-1.2.0/include>
                                     $<INSTALL_INTERFACE:extern/spectra-1.2.0/include>)
  target_compile_definitions(duneddm INTERFACE DUNE_DDM_HAVE_SPECTRA)
else()
  message(STATUS "Eigen not found, Spectra eigensolver will not be available")
endif()

target_include_directories(duneddm INTERFACE
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/taskflow>
                                   $<INSTALL_INTERFACE:extern/taskflow>)
target_compile_definitions(duneddm INTERFACE DUNE_DDM_HAVE_TASKFLOW=1)

# Build vendored LuaJIT
add_subdirectory(extern/luajit)

# Link LuaJIT using full paths for both build and install trees
# This ensures the library can be found when dune-ddm is used from either location
target_include_directories(duneddm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/luajit/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dune-ddm/luajit>
)

# Use absolute path for build tree, relative for install tree
target_link_libraries(duneddm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/luajit/src/libluajit.a>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}/libdune-ddm-luajit.a>
)

# Add system libraries needed by LuaJIT
if(UNIX AND NOT APPLE)
  target_link_libraries(duneddm INTERFACE dl m)
elseif(APPLE)
  target_link_libraries(duneddm INTERFACE m)
endif()

# Ensure luajit is built before anything that links against duneddm
add_dependencies(duneddm luajit)

target_compile_definitions(duneddm INTERFACE DUNE_DDM_HAVE_LUAJIT)

add_subdirectory(examples)
add_subdirectory(dune)

finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
