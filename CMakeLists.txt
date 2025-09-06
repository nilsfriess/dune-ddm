cmake_minimum_required(VERSION 3.24)
project(dune-ddm 
        VERSION 0.0.1
        LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COLOR_DIAGNOSTICS ON)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/extern)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/extern/dune-common/cmake/modules)

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

find_package(MPI REQUIRED)
find_package(SuiteSparse REQUIRED COMPONENTS UMFPACK CHOLMOD)
# optionally find STRUMPACK
find_package(STRUMPACK CONFIG)

add_subdirectory(extern/dune-common SYSTEM)
add_subdirectory(extern/dune-geometry SYSTEM)
add_subdirectory(extern/dune-uggrid SYSTEM)
add_subdirectory(extern/dune-grid SYSTEM)
add_subdirectory(extern/dune-istl SYSTEM)
add_subdirectory(extern/dune-localfunctions SYSTEM)
add_subdirectory(extern/dune-typetree SYSTEM)
add_subdirectory(extern/dune-functions SYSTEM)
add_subdirectory(extern/dune-pdelab SYSTEM)

include(CTest)

include(FetchContent)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.15.1
)
FetchContent_MakeAvailable(spdlog)

message(STATUS "Trying to download Spectra (C++ ARPACK implementation based on Eigen)")
find_package(Eigen3 REQUIRED)
FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra
  GIT_TAG        v1.0.1
)
FetchContent_MakeAvailable(Spectra)

message(STATUS "Trying to download Taskflow (C++ Parallel Programming Library)")
set(TF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  Taskflow
  GIT_REPOSITORY https://github.com/taskflow/taskflow.git
  GIT_TAG        v3.8.0
)
FetchContent_MakeAvailable(Taskflow)

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

add_library(eigensolvers src/eigensolvers.cc)
target_include_directories(eigensolvers PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(eigensolvers PRIVATE Dune::ISTL spdlog::spdlog Spectra ${BLOPEX_LIB} MPI::MPI_C)
if (${DUNE_DDM_WITH_BLOPEX})
  target_compile_definitions(eigensolvers PUBLIC DUNE_DDM_HAVE_BLOPEX)
endif()
if (${STRUMPACK_FOUND})
  target_link_libraries(eigensolvers PUBLIC STRUMPACK::strumpack)
  target_compile_definitions(eigensolvers PUBLIC DUNE_DDM_HAVE_STRUMPACK)
  message(STATUS "Found STRUMPACK: ${STRUMPACK_DIR}")
else()
  message(STATUS "Could not find STRUMPACK")
endif()
add_dune_eigen_flags(eigensolvers)
add_dune_suitesparse_flags(eigensolvers)
add_dune_mpi_flags(eigensolvers)

add_library(dune-ddm INTERFACE)
target_include_directories(dune-ddm PUBLIC INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dune-ddm PUBLIC INTERFACE Dune::ISTL spdlog::spdlog eigensolvers)
if (${STRUMPACK_FOUND})
  target_compile_definitions(dune-ddm PUBLIC DUNE_DDM_HAVE_STRUMPACK)
  target_link_libraries(dune-ddm PUBLIC INTERFACE STRUMPACK::strumpack)
endif()

add_subdirectory(examples)
add_subdirectory(tests)

