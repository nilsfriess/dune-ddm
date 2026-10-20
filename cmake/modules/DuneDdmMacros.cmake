# Module-specific setup for dune-ddm.
#
# Dune includes this file via dune_process_dependency_macros() both while
# configuring dune-ddm itself and in every downstream module that calls
# find_package(dune-ddm). Locating dune-ddm's mandatory external dependencies
# here (rather than only in the top-level CMakeLists.txt) means the imported
# targets they define also exist wherever the exported Dune::DDM target is
# consumed, so that target's interface can simply reference them by name.

# SuiteSparse is a *public* dependency: dune-ddm's headers
# (dune/ddm/eigensolvers/umfpack.hh and .../spectra.hh) include <umfpack.h>
# unconditionally. dune-common only searches for SuiteSparse as an optional
# component, so we require it explicitly here.
find_package(SuiteSparse REQUIRED COMPONENTS UMFPACK CHOLMOD)

# STRUMPACK is optional. If it is found, Dune::DDM links it publicly, so it is
# searched for here for the same reason as SuiteSparse.
find_package(STRUMPACK CONFIG QUIET)

# STRUMPACK's target links METIS::metis, which its config creates through
# find_dependency(METIS). CMake skips that call if an identical one already ran
# in this directory (dune-pdelab's config makes one), so create the target from
# DUNE's METIS target instead.
if(TARGET STRUMPACK::strumpack AND TARGET METIS::METIS AND NOT TARGET METIS::metis)
  add_library(METIS::metis INTERFACE IMPORTED)
  target_link_libraries(METIS::metis INTERFACE METIS::METIS)
endif()

# STRUMPACK's target adds -Wno-overloaded-virtual to everything that links it,
# which would turn that warning off for dune-ddm's own code. The warnings in its
# headers are silenced where they are included instead.
if(TARGET STRUMPACK::strumpack)
  get_target_property(_ddm_strumpack_options STRUMPACK::strumpack INTERFACE_COMPILE_OPTIONS)
  if(_ddm_strumpack_options)
    list(FILTER _ddm_strumpack_options EXCLUDE REGEX "-Wno-overloaded-virtual")
    set_target_properties(STRUMPACK::strumpack PROPERTIES INTERFACE_COMPILE_OPTIONS "${_ddm_strumpack_options}")
  endif()
  unset(_ddm_strumpack_options)
endif()
