cmake_minimum_required(VERSION 3.24)
project(dune-ddm 
        VERSION 0.0.1
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COLOR_DIAGNOSTICS ON)

file(GLOB DUNE_MODULES LIST_DIRECTORIES true CONFIGURE_DEPENDS extern/*)
foreach(DUNE_MODULE ${DUNE_MODULES})
    get_filename_component(DUNE_MODULE_NAME ${DUNE_MODULE} NAME)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/extern/${DUNE_MODULE_NAME})
endforeach()

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

add_subdirectory(extern/dune-common SYSTEM)
add_subdirectory(extern/dune-geometry SYSTEM)
add_subdirectory(extern/dune-uggrid SYSTEM)
add_subdirectory(extern/dune-grid SYSTEM)
add_subdirectory(extern/dune-istl SYSTEM)
add_subdirectory(extern/dune-localfunctions SYSTEM)
add_subdirectory(extern/dune-typetree SYSTEM)
add_subdirectory(extern/dune-functions SYSTEM)
add_subdirectory(extern/dune-pdelab SYSTEM)

include(CTest)

add_library(dune-ddm INTERFACE)
target_include_directories(dune-ddm INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(dune-ddm PUBLIC INTERFACE Dune::Common Dune::ISTL)

add_subdirectory(examples)
add_subdirectory(tests)

