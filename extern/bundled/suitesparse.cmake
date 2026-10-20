# CHOLMOD is required for UMFPACK's METIS ordering. It uses SuiteSparse's own
# METIS copy with renamed symbols, so it does not clash with the bundled METIS.
ddm_build_dependency(suitesparse ${DDM_EXTERN_DIR}/suitesparse [=[
set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd;camd;colamd;ccolamd;cholmod;umfpack" CACHE STRING "")
set(BUILD_STATIC_LIBS OFF CACHE BOOL "")
set(BUILD_TESTING OFF CACHE BOOL "")
set(SUITESPARSE_DEMOS OFF CACHE BOOL "")
set(SUITESPARSE_USE_CUDA OFF CACHE BOOL "")
set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "")
set(BLA_VENDOR "@DDM_DEPS_BLA_VENDOR@" CACHE STRING "")
set(BLAS_LIBRARIES "@BLAS_LIBRARIES@" CACHE STRING "")
set(LAPACK_LIBRARIES "@LAPACK_LIBRARIES@" CACHE STRING "")
]=])
