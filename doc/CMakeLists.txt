# Doxygen targets for dune-ddm.
#
# These are thin wrappers around doc/Doxyfile: they only run doxygen from the
# source tree, so `doxygen doc/Doxyfile` by hand does exactly the same thing.
# The targets are deliberately not attached to DUNE's `doc` target, which
# builds the documentation of every vendored module in extern/ and takes
# considerably longer.

find_package(Doxygen)

if(NOT TARGET Doxygen::doxygen)
  message(STATUS "Doxygen not found -- the 'docs' target is unavailable")
  return()
endif()

set(DOXYFILE ${PROJECT_SOURCE_DIR}/doc/Doxyfile)

# make docs -- build the HTML documentation into doc/html
add_custom_target(docs
  COMMAND Doxygen::doxygen ${DOXYFILE}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Building dune-ddm documentation with doxygen"
  VERBATIM)

add_custom_command(TARGET docs POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo
    "Documentation written to ${PROJECT_SOURCE_DIR}/doc/html/index.html"
  VERBATIM)

# make docs-coverage -- report what is still undocumented.
#
# WARN_IF_UNDOCUMENTED has no effect while EXTRACT_ALL is on, so the same
# Doxyfile is replayed with EXTRACT_ALL disabled and the HTML/graph output
# switched off, which makes it a few-second run.
add_custom_target(docs-coverage
  COMMAND ${CMAKE_COMMAND} -E env sh -c
    "{ cat ${DOXYFILE}; \
       echo 'EXTRACT_ALL = NO'; \
       echo 'GENERATE_HTML = NO'; \
       echo 'GENERATE_TAGFILE ='; \
       echo 'HAVE_DOT = NO'; \
       echo 'QUIET = YES'; \
       echo 'WARN_LOGFILE = doc/undocumented.log'; \
     } | $<TARGET_FILE:Doxygen::doxygen> -"
  COMMAND ${CMAKE_COMMAND} -E cat ${PROJECT_SOURCE_DIR}/doc/undocumented.log
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Listing undocumented entities in dune/ddm"
  VERBATIM)
