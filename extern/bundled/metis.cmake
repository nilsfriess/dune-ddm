ddm_build_dependency(metis ${DDM_EXTERN_DIR}/metis [=[
set(GKLIB_PATH "@source_dir@/GKlib" CACHE PATH "")
set(SHARED ON CACHE BOOL "")
set(MATH_LIB m CACHE STRING "")
]=])
