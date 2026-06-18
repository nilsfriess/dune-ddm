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

# Create an imported target for LuaJIT
add_library(luajit STATIC IMPORTED GLOBAL)
add_dependencies(luajit luajit_external)

set_target_properties(luajit PROPERTIES
  IMPORTED_LOCATION ${LUAJIT_INSTALL_DIR}/lib/libluajit-5.1.a
  INTERFACE_INCLUDE_DIRECTORIES ${LUAJIT_INSTALL_DIR}/include/luajit-2.1
)

# LuaJIT requires libm and libdl on Linux
if(UNIX AND NOT APPLE)
  set_property(TARGET luajit APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES m dl)
endif()

# Make duneddm depend on LuaJIT so all users automatically get it
target_link_libraries(duneddm INTERFACE luajit)
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
