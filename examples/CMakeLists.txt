find_package(Eigen3 REQUIRED)

include(FetchContent)
message(STATUS "Trying to download Spectra (C++ ARPACK alternative)")
FetchContent_Declare(
  Spectra
  GIT_REPOSITORY https://github.com/yixuan/spectra
  GIT_TAG        v1.0.1
)
FetchContent_MakeAvailable(Spectra)

add_executable(poisson poisson.cc)
target_link_libraries(poisson PUBLIC dune-ddm Dune::PDELab Spectra)
add_dune_all_flags(poisson)
