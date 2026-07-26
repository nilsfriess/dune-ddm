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
