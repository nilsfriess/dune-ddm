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

message(STATUS "Trying to download Spectra (C++ ARPACK implementation based on Eigen)")
find_package(Eigen3 REQUIRED)
FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra
  GIT_TAG        v1.0.1
)
FetchContent_MakeAvailable(Spectra)

message(STATUS "Trying to download BLOPEX (implementation of LOBPCG)")
FetchContent_Declare(
  blopex_all
  GIT_REPOSITORY https://github.com/nilsfriess/blopex
  GIT_COMMIT     master
)
FetchContent_MakeAvailable(blopex_all)
add_subdirectory(${blopex_all_SOURCE_DIR}/blopex_abstract)

# add_custom_command(
#         OUTPUT ${blopex_all_SOURCE_DIR}/blopex_abstract/lib/libBLOPEX.a
#         COMMAND cd ${blopex_all_SOURCE_DIR}/blopex_abstract && make clean && make CFLAGS='-O3')
# add_custom_target(BLOPEX_BUILD DEPENDS ${blopex_all_SOURCE_DIR}/blopex_abstract/lib/libBLOPEX.a)
# add_library(BLOPEX STATIC IMPORTED GLOBAL)
# set_target_properties(BLOPEX
#     PROPERTIES
#     IMPORTED_LOCATION ${blopex_all_SOURCE_DIR}/blopex_abstract/lib/libBLOPEX.a
#     INTERFACE_INCLUDE_DIRECTORIES ${blopex_all_SOURCE_DIR}/blopex_abstract/include
# )
# add_dependencies(BLOPEX BLOPEX_BUILD)

add_library(eigensolvers src/eigensolvers.cc)
target_include_directories(eigensolvers PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(eigensolvers PRIVATE Dune::ISTL spdlog::spdlog Spectra BLOPEX MPI::MPI_C)
add_dune_eigen_flags(eigensolvers)
add_dune_mpi_flags(eigensolvers)

add_library(dune-ddm INTERFACE)
target_include_directories(dune-ddm PUBLIC INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dune-ddm PUBLIC INTERFACE Dune::ISTL spdlog::spdlog eigensolvers)

add_subdirectory(examples)
add_subdirectory(tests)

