# Build LuaJIT as an external project
include(ExternalProject)

set(LUAJIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/luajit)
set(LUAJIT_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/luajit-build)
set(LUAJIT_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/luajit-install)

ExternalProject_Add(luajit_external
  SOURCE_DIR ${LUAJIT_SOURCE_DIR}
  BUILD_IN_SOURCE 0
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${CMAKE_COMMAND} -E env
    "MAKEFLAGS="  # Clear parallel make flags to avoid issues
    make -C ${LUAJIT_SOURCE_DIR}
    PREFIX=${LUAJIT_INSTALL_DIR}
    BUILDMODE=static
    "CC=${CMAKE_C_COMPILER}"
  INSTALL_COMMAND ${CMAKE_COMMAND} -E env
    make -C ${LUAJIT_SOURCE_DIR}
    PREFIX=${LUAJIT_INSTALL_DIR}
    BUILDMODE=static
    install
  BUILD_BYPRODUCTS
    ${LUAJIT_INSTALL_DIR}/lib/libluajit-5.1.a
)

# Imported target for the bundled LuaJIT. This is an *examples-only* dependency:
# nothing under dune/ddm/ uses Lua, so LuaJIT is deliberately NOT part of the
# duneddm interface (otherwise every consumer of Dune::DDM would be forced to
# link a library it never uses). Example targets that need Lua link `luajit`
# directly; see examples/CMakeLists.txt.
#
# Linking this imported target also propagates the build-order dependency on
# luajit_external (via add_dependencies below) and the in-tree header path, so
# no extra wiring is needed.
add_library(luajit STATIC IMPORTED GLOBAL)
add_dependencies(luajit luajit_external)

# luajit_external only creates the include directory at build time, but CMake
# requires an imported target's INTERFACE_INCLUDE_DIRECTORIES to exist already at
# configure time. Create it now so configuration succeeds on a clean build; the
# ExternalProject populates it with the actual headers when it is built.
file(MAKE_DIRECTORY ${LUAJIT_INSTALL_DIR}/include/luajit-2.1)

set_target_properties(luajit PROPERTIES
  IMPORTED_LOCATION ${LUAJIT_INSTALL_DIR}/lib/libluajit-5.1.a
  INTERFACE_INCLUDE_DIRECTORIES ${LUAJIT_INSTALL_DIR}/include/luajit-2.1
)

# LuaJIT requires libm and libdl on Linux.
if(UNIX AND NOT APPLE)
  set_property(TARGET luajit APPEND PROPERTY INTERFACE_LINK_LIBRARIES m dl)
endif()

# Spectra - header-only eigenvalue solver library
# Add include directories directly to duneddm to avoid export issues
if(TARGET Eigen3::Eigen)
  target_include_directories(duneddm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/spectra/include>
    $<INSTALL_INTERFACE:extern/spectra/include>
  )
  target_compile_definitions(duneddm INTERFACE DUNE_DDM_HAVE_SPECTRA=1)
  target_link_libraries(duneddm INTERFACE Eigen3::Eigen)

  # Install the header tree so the $<INSTALL_INTERFACE:...> path above actually
  # exists in the install tree (otherwise consumers of the exported Dune::DDM
  # target hit a "non-existent path" error).
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/spectra/include/
    DESTINATION extern/spectra/include)
else()
  message(STATUS "Eigen not found, Spectra eigensolver will not be available")
endif()

# TRL - header-only randomized linear algebra library
# Requires Eigen3 and OpenMP (already found in root CMakeLists.txt)
target_include_directories(duneddm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/trl/include>
  $<INSTALL_INTERFACE:extern/trl/include>
)
target_compile_definitions(duneddm INTERFACE DUNE_DDM_HAVE_TRL=1)

# Install the header tree to match the $<INSTALL_INTERFACE:...> path above.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/trl/include/
  DESTINATION extern/trl/include)

# dune-pdelab does not install the headers under dune/pdelab/test (they are
# treated as test-only), but downstream code (e.g. dune-heart's rwdata.hh)
# includes some of them, such as l2difference.hh. Install that header tree into
# the same include destination as the rest of pdelab so the includes resolve
# against the installed module. Only relevant for the in-tree superbuild.
if(DUNE_DDM_SUPERBUILD)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dune-pdelab/dune/pdelab/test/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/pdelab/test
    FILES_MATCHING PATTERN "*.hh")
endif()
