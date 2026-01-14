# CMakeLists.txt for LuaJIT
# This builds LuaJIT as an external project using its native Makefile build system

cmake_minimum_required(VERSION 3.24)
project(LuaJIT C)

include(ExternalProject)
include(GNUInstallDirs)

set(LUAJIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LUAJIT_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/luajit-build)
set(LUAJIT_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/install)

# Determine the library name based on platform
if(APPLE)
  set(LUAJIT_LIB_NAME "libluajit.a")
else()
  set(LUAJIT_LIB_NAME "libluajit.a")
endif()

# Build LuaJIT using its native Makefile
# Use USES_TERMINAL to show build output and pass jobserver to sub-make
ExternalProject_Add(
  luajit_build
  SOURCE_DIR ${LUAJIT_SOURCE_DIR}
  BINARY_DIR ${LUAJIT_BUILD_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${CMAKE_COMMAND} -E env "MAKEFLAGS=${MAKEFLAGS}"
    ${CMAKE_MAKE_PROGRAM} -C ${LUAJIT_SOURCE_DIR}/src BUILDMODE=static
  INSTALL_COMMAND ""
  BUILD_ALWAYS TRUE
  USES_TERMINAL_BUILD TRUE
  BUILD_BYPRODUCTS 
    ${LUAJIT_SOURCE_DIR}/src/libluajit.a
    ${LUAJIT_SOURCE_DIR}/src/luajit
)

# Create an imported library target for build time
add_library(luajit STATIC IMPORTED GLOBAL)
add_dependencies(luajit luajit_build)

# On some platforms, LuaJIT needs libdl and libm
if(UNIX AND NOT APPLE)
  set(LUAJIT_SYSTEM_LIBS "dl;m")
elseif(APPLE)
  set(LUAJIT_SYSTEM_LIBS "m")
else()
  set(LUAJIT_SYSTEM_LIBS "")
endif()

# Set library location
set_target_properties(luajit PROPERTIES
  IMPORTED_LOCATION ${LUAJIT_SOURCE_DIR}/src/libluajit.a
  INTERFACE_INCLUDE_DIRECTORIES ${LUAJIT_SOURCE_DIR}/src
  INTERFACE_LINK_LIBRARIES "${LUAJIT_SYSTEM_LIBS}"
)

# Export variables for parent scope
set(LUAJIT_INCLUDE_DIR ${LUAJIT_SOURCE_DIR}/src PARENT_SCOPE)
set(LUAJIT_LIBRARY ${LUAJIT_SOURCE_DIR}/src/libluajit.a PARENT_SCOPE)
set(LUAJIT_SYSTEM_LIBS "${LUAJIT_SYSTEM_LIBS}" PARENT_SCOPE)

# Install the static library and headers
install(FILES ${LUAJIT_SOURCE_DIR}/src/libluajit.a
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RENAME libdune-ddm-luajit.a
)

install(FILES 
  ${LUAJIT_SOURCE_DIR}/src/lua.h
  ${LUAJIT_SOURCE_DIR}/src/lualib.h
  ${LUAJIT_SOURCE_DIR}/src/lauxlib.h
  ${LUAJIT_SOURCE_DIR}/src/luajit.h
  ${LUAJIT_SOURCE_DIR}/src/luaconf.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune-ddm/luajit
)
