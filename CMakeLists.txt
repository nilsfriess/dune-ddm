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

add_library(dune-ddm INTERFACE)
target_include_directories(dune-ddm INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dune-ddm PUBLIC INTERFACE Dune::ISTL spdlog::spdlog)

add_subdirectory(examples)
add_subdirectory(tests)

