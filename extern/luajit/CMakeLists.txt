# CMakeLists.txt for LuaJIT
# This builds LuaJIT as an external project using its native Makefile build system

cmake_minimum_required(VERSION 3.24)
project(LuaJIT C)

include(ExternalProject)

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
ExternalProject_Add(
  luajit_build
  SOURCE_DIR ${LUAJIT_SOURCE_DIR}
  BINARY_DIR ${LUAJIT_BUILD_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make -C ${LUAJIT_SOURCE_DIR}/src BUILDMODE=static
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS 
    ${LUAJIT_SOURCE_DIR}/src/libluajit.a
    ${LUAJIT_SOURCE_DIR}/src/luajit
)

# Create an imported library target
add_library(luajit STATIC IMPORTED GLOBAL)
add_dependencies(luajit luajit_build)

# Set library location
set_target_properties(luajit PROPERTIES
  IMPORTED_LOCATION ${LUAJIT_SOURCE_DIR}/src/libluajit.a
  INTERFACE_INCLUDE_DIRECTORIES ${LUAJIT_SOURCE_DIR}/src
)

# On some platforms, LuaJIT needs libdl and libm
if(UNIX AND NOT APPLE)
  set_target_properties(luajit PROPERTIES
    INTERFACE_LINK_LIBRARIES "dl;m"
  )
elseif(APPLE)
  set_target_properties(luajit PROPERTIES
    INTERFACE_LINK_LIBRARIES "m"
  )
endif()

# Export variables for parent scope
set(LUAJIT_INCLUDE_DIR ${LUAJIT_SOURCE_DIR}/src PARENT_SCOPE)
set(LUAJIT_LIBRARY ${LUAJIT_SOURCE_DIR}/src/libluajit.a PARENT_SCOPE)
