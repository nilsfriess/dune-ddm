# Builds METIS, ParMETIS, SuiteSparse, Eigen and STRUMPACK from the submodules
# in extern/ and installs them into build/extern/install. This happens at
# configure time because the DUNE modules search for these libraries while
# being configured. Build logs are written to build/extern/<name>-build.log.
option(DUNE_DDM_BUNDLED_DEPS
  "Build METIS, ParMETIS, SuiteSparse, Eigen and STRUMPACK from the submodules in extern/ while configuring"
  ON)

if(NOT DUNE_DDM_BUNDLED_DEPS)
  return()
endif()

set(DDM_EXTERN_DIR ${CMAKE_CURRENT_LIST_DIR})
set(DDM_DEPS_PREFIX ${CMAKE_BINARY_DIR}/extern/install)
cmake_host_system_information(RESULT DDM_DEPS_JOBS QUERY NUMBER_OF_LOGICAL_CORES)

# ddm_build_dependency(<name> <source_dir> <cache>)
#
# Configures, builds and installs <source_dir> into DDM_DEPS_PREFIX. <cache> holds
# set(... CACHE ...) lines (with @VAR@ substitution) that are passed via cmake -C.
function(ddm_build_dependency name source_dir cache)
  set(build_dir ${CMAKE_BINARY_DIR}/extern/${name}-build)
  set(init_cache ${CMAKE_BINARY_DIR}/extern/${name}-init.cmake)
  set(log ${CMAKE_BINARY_DIR}/extern/${name}-build.log)

  string(CONFIGURE [=[
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_C_COMPILER "@CMAKE_C_COMPILER@" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "@CMAKE_CXX_COMPILER@" CACHE FILEPATH "")
set(CMAKE_INSTALL_PREFIX "@DDM_DEPS_PREFIX@" CACHE PATH "")
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "")
# So the installed libraries find each other without LD_LIBRARY_PATH.
set(CMAKE_INSTALL_RPATH "@DDM_DEPS_PREFIX@/lib" CACHE STRING "")
set(BUILD_SHARED_LIBS ON CACHE BOOL "")
]=] contents @ONLY)
  string(CONFIGURE "${cache}" cache @ONLY)
  string(APPEND contents "${cache}")

  set(old_contents "")
  if(EXISTS ${init_cache})
    file(READ ${init_cache} old_contents)
  endif()

  file(WRITE ${log} "")

  # cmake -C does not overwrite existing cache entries, so start from a fresh
  # cache when the options change.
  if(NOT contents STREQUAL old_contents OR NOT EXISTS ${build_dir}/CMakeCache.txt)
    message(STATUS "Configuring bundled ${name} (log: ${log})")
    file(REMOVE ${build_dir}/CMakeCache.txt)
    file(WRITE ${init_cache}.new "${contents}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} -C ${init_cache}.new -S ${source_dir} -B ${build_dir}
      OUTPUT_VARIABLE output
      ERROR_VARIABLE output
      RESULT_VARIABLE result)
    file(APPEND ${log} "${output}")
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Configuring bundled ${name} failed, see ${log}")
    endif()
    file(RENAME ${init_cache}.new ${init_cache})
  endif()

  # Clear MAKEFLAGS so the nested build does not inherit make's jobserver.
  message(STATUS "Building bundled ${name} (log: ${log})")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS
      ${CMAKE_COMMAND} --build ${build_dir} --target install --parallel ${DDM_DEPS_JOBS}
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
    RESULT_VARIABLE result)
  file(APPEND ${log} "${output}")
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Building bundled ${name} failed, see ${log}")
  endif()
endfunction()

# Used by SuiteSparse and STRUMPACK, and found again later by dune-common, so
# everything uses the same BLAS/LAPACK (select with -DBLA_VENDOR=...). Read
# BLA_VENDOR first, since FindBLAS sets it to "All" when it is empty.
if(BLA_VENDOR)
  set(DDM_DEPS_BLA_VENDOR ${BLA_VENDOR})
else()
  set(DDM_DEPS_BLA_VENDOR Generic)
endif()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# ParMETIS and STRUMPACK need METIS, so the order matters.
include(${DDM_EXTERN_DIR}/bundled/metis.cmake)
include(${DDM_EXTERN_DIR}/bundled/parmetis.cmake)
include(${DDM_EXTERN_DIR}/bundled/suitesparse.cmake)
include(${DDM_EXTERN_DIR}/bundled/eigen.cmake)
include(${DDM_EXTERN_DIR}/bundled/strumpack.cmake)

list(PREPEND CMAKE_PREFIX_PATH ${DDM_DEPS_PREFIX})
