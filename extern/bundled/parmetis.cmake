# Pass MPI explicitly instead of relying on the compiler wrapper.
find_package(MPI REQUIRED COMPONENTS C)

ddm_build_dependency(parmetis ${DDM_EXTERN_DIR}/parmetis [=[
set(GKLIB_PATH "@source_dir@/headers" CACHE PATH "")
set(METIS_PATH "@DDM_DEPS_PREFIX@" CACHE PATH "")
set(MPI_INCLUDE_PATH "@MPI_C_INCLUDE_DIRS@" CACHE STRING "")
set(MPI_LIBRARIES "@MPI_C_LIBRARIES@" CACHE STRING "")
set(SHARED ON CACHE BOOL "")
]=])
