# Header-only: installing just copies the headers and the CMake config.
ddm_build_dependency(eigen ${DDM_EXTERN_DIR}/eigen [=[
set(BUILD_TESTING OFF CACHE BOOL "")
set(EIGEN_BUILD_DOC OFF CACHE BOOL "")
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "")
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "")
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "")
]=])
