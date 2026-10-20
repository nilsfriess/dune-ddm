# STRUMPACK requires a Fortran compiler, even without MPI. Skip it if there is
# none, since it is optional.
include(CheckLanguage)
check_language(Fortran)
if(NOT CMAKE_Fortran_COMPILER)
  message(STATUS "No Fortran compiler found, skipping bundled STRUMPACK")
  return()
endif()

# Built without MPI, which would also require ScaLAPACK. Scotch and ZFP are
# disabled so that no optional system libraries get picked up.
ddm_build_dependency(strumpack ${DDM_EXTERN_DIR}/STRUMPACK [=[
set(CMAKE_Fortran_COMPILER "@CMAKE_Fortran_COMPILER@" CACHE FILEPATH "")
set(STRUMPACK_USE_MPI OFF CACHE BOOL "")
set(TPL_ENABLE_SCOTCH OFF CACHE BOOL "")
set(TPL_ENABLE_ZFP OFF CACHE BOOL "")
set(TPL_METIS_PREFIX "@DDM_DEPS_PREFIX@" CACHE PATH "")
set(TPL_BLAS_LIBRARIES "@BLAS_LIBRARIES@" CACHE STRING "")
set(TPL_LAPACK_LIBRARIES "@LAPACK_LIBRARIES@" CACHE STRING "")
]=])
