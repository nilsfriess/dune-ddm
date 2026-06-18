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

# Create an imported target for LuaJIT. This is used for build-tree consumers:
# it provides the correct build ordering against luajit_external and the
# in-tree header path.
add_library(luajit STATIC IMPORTED GLOBAL)
add_dependencies(luajit luajit_external)

set_target_properties(luajit PROPERTIES
  IMPORTED_LOCATION ${LUAJIT_INSTALL_DIR}/lib/libluajit-5.1.a
  INTERFACE_INCLUDE_DIRECTORIES ${LUAJIT_INSTALL_DIR}/include/luajit-2.1
)

# Install the static library and headers so consumers of the *installed*
# Dune::DDM target can find them. The imported target above only exists in the
# superbuild's own build tree, so the install tree needs real files to link.
install(FILES ${LUAJIT_INSTALL_DIR}/lib/libluajit-5.1.a
  DESTINATION extern/luajit/lib)
install(DIRECTORY ${LUAJIT_INSTALL_DIR}/include/luajit-2.1
  DESTINATION extern/luajit/include)

# Link LuaJIT into duneddm. In the build tree we link the imported target (so
# luajit_external builds first and the in-tree include dir is picked up); in the
# install tree the imported target is not part of the exported target set, so we
# point at the installed static library by absolute path instead. Linking the
# bare target name would otherwise be exported verbatim and degrade to
# "-lluajit", which the consumer's linker cannot resolve.
target_link_libraries(duneddm INTERFACE
  $<BUILD_INTERFACE:luajit>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/extern/luajit/lib/libluajit-5.1.a>
)
target_include_directories(duneddm INTERFACE
  $<INSTALL_INTERFACE:extern/luajit/include/luajit-2.1>
)

# LuaJIT requires libm and libdl on Linux. These are needed in both the build
# and install interfaces, so attach them to duneddm directly rather than only to
# the imported target (which does not participate in the install tree).
if(UNIX AND NOT APPLE)
  target_link_libraries(duneddm INTERFACE m dl)
endif()

target_compile_definitions(duneddm INTERFACE DUNE_DDM_HAVE_LUAJIT=1)

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
