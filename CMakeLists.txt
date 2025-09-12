cmake_minimum_required(VERSION 3.24)
project(dune-ddm
        VERSION 0.0.1
        LANGUAGES CXX C)

set(CMAKE_COLOR_DIAGNOSTICS ON)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/extern)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/extern/dune-common/cmake/modules)

option(DUNE_DDM_SUPERBUILD "Use this project as a superbuild" OFF)
if (${DUNE_DDM_SUPERBUILD})
  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/extern)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/extern/dune-common/cmake/modules)

  message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

  set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

  add_subdirectory(extern/dune-common SYSTEM)
  add_subdirectory(extern/dune-geometry SYSTEM)
  add_subdirectory(extern/dune-uggrid SYSTEM)
  add_subdirectory(extern/dune-grid SYSTEM)
  add_subdirectory(extern/dune-istl SYSTEM)
  add_subdirectory(extern/dune-localfunctions SYSTEM)
  add_subdirectory(extern/dune-typetree SYSTEM)
  add_subdirectory(extern/dune-functions SYSTEM)
  add_subdirectory(extern/dune-pdelab SYSTEM)
else()
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
  if (dune-common_VERSION VERSION_GREATER_EQUAL 2.10)
    dune_add_library(duneddm EXPORT_NAME DDM)
  else()
    dune_add_library(duneddm EXPORT_NAME Dune::DDM)
  endif()

  target_link_libraries(duneddm PUBLIC ${DUNE_LIBS})
  dune_enable_all_packages()
endif()

find_package(MPI REQUIRED)
find_package(SuiteSparse REQUIRED COMPONENTS UMFPACK CHOLMOD)
# optionally find STRUMPACK
find_package(STRUMPACK CONFIG)

include(FetchContent)

message(STATUS "Trying to download Spectra (C++ ARPACK implementation based on Eigen)")
find_package(Eigen3 REQUIRED)
FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra
  GIT_TAG        v1.0.1
)
FetchContent_MakeAvailable(Spectra)

option(DUNE_DDM_WITH_TASKFLOW "Download the taskflow library for asynchronous execution" OFF)
if (${DUNE_DDM_WITH_TASKFLOW})
   message(STATUS "Trying to download Taskflow (C++ Parallel Programming Library)")
   set(TF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
   set(TF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
   FetchContent_Declare(
     Taskflow
     GIT_REPOSITORY https://github.com/taskflow/taskflow.git
     GIT_TAG        v3.8.0
   )
   FetchContent_MakeAvailable(Taskflow)
   target_link_libraries(duneddm PUBLIC Taskflow)
   target_compile_definitions(duneddm PUBLIC DUNE_DDM_HAVE_TASKFLOW=1)
endif()

option(DUNE_DDM_WITH_BLOPEX "Download and enable BLOPEX" OFF)
if(${DUNE_DDM_WITH_BLOPEX})
  message(STATUS "Trying to download BLOPEX (implementation of LOBPCG)")
  FetchContent_Declare(
    blopex_all
    GIT_REPOSITORY https://github.com/nilsfriess/blopex
    GIT_COMMIT     master
  )
  FetchContent_MakeAvailable(blopex_all)
  add_subdirectory(${blopex_all_SOURCE_DIR}/blopex_abstract)

  set(BLOPEX_LIB BLOPEX)
endif()

target_sources(duneddm PRIVATE dune/ddm/eigensolvers/eigensolvers.cc)
target_include_directories(duneddm PRIVATE ${Spectra_SOURCE_DIR}/include)
add_dune_suitesparse_flags(duneddm)
add_dune_arpackpp_flags(duneddm)
add_dune_eigen_flags(duneddm)
add_dune_superlu_flags(duneddm)
add_dune_mpi_flags(duneddm)
add_dune_parmetis_flags(duneddm)

# add_library(eigensolvers src/eigensolvers.cc)
# target_include_directories(eigensolvers PUBLIC ${PROJECT_SOURCE_DIR}/include)
# target_link_libraries(eigensolvers PRIVATE Dune::ISTL spdlog::spdlog Spectra ${BLOPEX_LIB} MPI::MPI_C)
# if (${DUNE_DDM_WITH_BLOPEX})
#   target_compile_definitions(eigensolvers PUBLIC DUNE_DDM_HAVE_BLOPEX)
# endif()
# if (${STRUMPACK_FOUND})
#   target_link_libraries(eigensolvers PUBLIC STRUMPACK::strumpack)
#   target_compile_definitions(eigensolvers PUBLIC DUNE_DDM_HAVE_STRUMPACK)
#   message(STATUS "Found STRUMPACK: ${STRUMPACK_DIR}")
# else()
#   message(STATUS "Could not find STRUMPACK")
# endif()
# # add_dune_eigen_flags(eigensolvers)
# add_dune_suitesparse_flags(eigensolvers)
# add_dune_mpi_flags(eigensolvers)

add_subdirectory(examples)
add_subdirectory(dune)

if(${DUNE_DDM_SUPERBUILD})
  add_library(dune-ddm INTERFACE)
  target_include_directories(dune-ddm PUBLIC INTERFACE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(dune-ddm PUBLIC INTERFACE Dune::ISTL eigensolvers)
else()
  # if (${STRUMPACK_FOUND})
  #   target_compile_definitions(duneddm PUBLIC DUNE_DDM_HAVE_STRUMPACK=1)
  #   target_link_libraries(duneddm PUBLIC STRUMPACK::strumpack)
  # endif()
  # finalize the dune project, e.g. generating config.h etc.
  finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
endif()
