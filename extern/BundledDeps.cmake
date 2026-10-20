# ---------------------------------------------------------------------------
# Bundled third-party libraries
#
# When enabled (the default), METIS, ParMETIS, SuiteSparse (UMFPACK/CHOLMOD) and
# Eigen are built from the git submodules in extern/ and installed into
# build/extern/install. The only libraries the machine has to provide are then
# MPI and a BLAS/LAPACK.
#
# The libraries are built while CMake configures, not at `make` time: the DUNE
# modules search for them with find_path()/find_library() (and read version
# macros from their headers) inside add_subdirectory(extern/dune-common) and
# friends, so they have to be installed before those calls run. Prepending the
# install prefix to CMAKE_PREFIX_PATH is then enough for the existing Find
# modules and DuneDdmMacros.cmake to pick them up like system installations.
#
# The first configure takes a minute or two longer. Later configures only check
# that the builds are up to date. The output of each library's last build is in
# build/extern/<name>-build.log.
#
# Disable with -DDUNE_DDM_BUNDLED_DEPS=OFF to use installations found on the
# system instead.
# ---------------------------------------------------------------------------
option(DUNE_DDM_BUNDLED_DEPS
  "Build METIS, ParMETIS, SuiteSparse and Eigen from the submodules in extern/ while configuring"
  ON)

if(NOT DUNE_DDM_BUNDLED_DEPS)
  return()
endif()

set(DDM_DEPS_PREFIX ${CMAKE_BINARY_DIR}/extern/install)
cmake_host_system_information(RESULT DDM_DEPS_JOBS QUERY NUMBER_OF_LOGICAL_CORES)

# ddm_build_dependency(<name> <source_dir> <cache>)
#
# Configures, builds and installs the CMake project in <source_dir> into
# DDM_DEPS_PREFIX. <cache> holds set(... CACHE ...) lines for the project's
# options, with @VAR@ references substituted. It is passed as an initial-cache
# file (cmake -C), which keeps list values such as SUITESPARSE_ENABLE_PROJECTS
# intact. The configure step only re-runs when the cache contents change; the
# build step runs on every configure but is incremental.
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
# Lets the installed libraries find each other (libumfpack -> libcholmod,
# libparmetis -> libmetis) without LD_LIBRARY_PATH.
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

  # cmake -C only seeds entries that are not in the cache yet, so a changed
  # configuration starts from a fresh cache. The init-cache file is written only
  # after a successful configure, so a failed one is retried next time.
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

  # As for LuaJIT, clear MAKEFLAGS so that a re-configure triggered from make
  # does not hand its jobserver to the nested build.
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

# ParMETIS is given MPI's include directories and libraries rather than the MPI
# compiler wrapper, which also works where the C compiler is the wrapper itself.
find_package(MPI REQUIRED COMPONENTS C)

# SuiteSparse needs BLAS (UMFPACK, CHOLMOD) and LAPACK (CHOLMOD). It is handed the
# libraries found here; dune-common's FindSuiteSparse later runs the same search,
# so UMFPACK and the dune-ddm targets use the same BLAS. Select an implementation
# with -DBLA_VENDOR=... The vendor is read before the search because FindBLAS
# sets BLA_VENDOR to "All" when it is empty.
if(BLA_VENDOR)
  set(DDM_DEPS_BLA_VENDOR ${BLA_VENDOR})
else()
  set(DDM_DEPS_BLA_VENDOR Generic)
endif()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

ddm_build_dependency(metis ${CMAKE_CURRENT_LIST_DIR}/metis [=[
set(GKLIB_PATH "@source_dir@/GKlib" CACHE PATH "")
set(SHARED ON CACHE BOOL "")
set(MATH_LIB m CACHE STRING "")
]=])

ddm_build_dependency(parmetis ${CMAKE_CURRENT_LIST_DIR}/parmetis [=[
set(GKLIB_PATH "@source_dir@/headers" CACHE PATH "")
set(METIS_PATH "@DDM_DEPS_PREFIX@" CACHE PATH "")
set(MPI_INCLUDE_PATH "@MPI_C_INCLUDE_DIRS@" CACHE STRING "")
set(MPI_LIBRARIES "@MPI_C_LIBRARIES@" CACHE STRING "")
set(SHARED ON CACHE BOOL "")
]=])

# CHOLMOD is needed even though dune-ddm only calls UMFPACK: UMFPACK's METIS
# ordering (UMFPACK_ORDERING_METIS) goes through CHOLMOD's partition module. That
# uses SuiteSparse's own copy of METIS with renamed symbols, so it does not clash
# with the METIS built above.
ddm_build_dependency(suitesparse ${CMAKE_CURRENT_LIST_DIR}/suitesparse [=[
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

# Eigen is header-only; with the tests, docs, demos and its own BLAS/LAPACK
# turned off, installing it just copies the headers and its CMake config.
ddm_build_dependency(eigen ${CMAKE_CURRENT_LIST_DIR}/eigen [=[
set(BUILD_TESTING OFF CACHE BOOL "")
set(EIGEN_BUILD_DOC OFF CACHE BOOL "")
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "")
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "")
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "")
]=])

list(PREPEND CMAKE_PREFIX_PATH ${DDM_DEPS_PREFIX})
