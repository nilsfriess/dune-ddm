if (${dune-pdelab_FOUND})
  option(DUNE_DDM_BUILD_EXAMPLES "Try to build the examples" On)
  if (${DUNE_DDM_BUILD_EXAMPLES})
    find_package(Lua)
    if(Lua_FOUND AND NOT TARGET Lua::Lua)
      add_library(Lua::Lua INTERFACE IMPORTED)
      set_target_properties(
        Lua::Lua
        PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LUA_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${LUA_LIBRARIES}"
      )
    endif()

    # Poisson example
    add_executable(poisson EXCLUDE_FROM_ALL poisson.cc)
    target_link_libraries(poisson PUBLIC duneddm -llapacke)
    add_dune_blas_lapack_flags(poisson)
    target_link_dune_default_libraries(poisson)
    dune_symlink_to_source_files(FILES poisson.ini)

    # Linear elasticity example
    add_executable(linearelasticity EXCLUDE_FROM_ALL linearelasticity.cc)
    target_link_libraries(linearelasticity PUBLIC duneddm -llapacke)
    if (${Lua_FOUND})
      target_link_libraries(linearelasticity PUBLIC Lua::Lua)
      target_compile_definitions(linearelasticity PUBLIC DUNE_DDM_HAVE_LUA)
      dune_symlink_to_source_files(FILES coefficient.lua)
    endif()
    add_dune_blas_lapack_flags(linearelasticity)
    target_link_dune_default_libraries(linearelasticity)
    dune_symlink_to_source_files(FILES linearelasticity.ini)

    # Nonlinear Poisson example
    add_executable(nonlinearpoisson EXCLUDE_FROM_ALL nonlinearpoisson.cc)
    target_link_libraries(nonlinearpoisson PUBLIC duneddm)
    add_dune_blas_lapack_flags(nonlinearpoisson)
    target_link_dune_default_libraries(nonlinearpoisson)
    dune_symlink_to_source_files(FILES nonlinearpoisson.ini)

    add_executable(convectiondiffusiondg EXCLUDE_FROM_ALL convectiondiffusiondg.cc)
    target_link_libraries(convectiondiffusiondg PUBLIC duneddm Lua::Lua)
    add_dune_blas_lapack_flags(convectiondiffusiondg)
    target_link_dune_default_libraries(convectiondiffusiondg)
    dune_symlink_to_source_files(FILES convectiondiffusiondg.ini)

    # Debug overlap example
    add_executable(debug_overlap EXCLUDE_FROM_ALL debug_overlap.cc)
    target_link_libraries(debug_overlap PUBLIC duneddm)
    add_dune_blas_lapack_flags(debug_overlap)
    target_link_dune_default_libraries(debug_overlap)

    # Test for Neumann nullspace (A * 1 = 0)
    add_executable(test_neumann_nullspace EXCLUDE_FROM_ALL test_neumann_nullspace.cc)
    target_link_libraries(test_neumann_nullspace PUBLIC duneddm)
    add_dune_blas_lapack_flags(test_neumann_nullspace)
    target_link_dune_default_libraries(test_neumann_nullspace)

    # Simple nonoverlapping DG test
    add_executable(test_dg_nonoverlapping EXCLUDE_FROM_ALL test_dg_nonoverlapping.cc)
    target_link_libraries(test_dg_nonoverlapping PUBLIC duneddm)
    target_link_dune_default_libraries(test_dg_nonoverlapping)

    # Test unified framework
    add_executable(test_framework EXCLUDE_FROM_ALL test_framework.cc)
    target_link_libraries(test_framework PUBLIC duneddm -llapacke)
    add_dune_blas_lapack_flags(test_framework)
    target_link_dune_default_libraries(test_framework)
    dune_symlink_to_source_files(FILES test_framework.ini)
  
    # add_executable(pdelab_example EXCLUDE_FROM_ALL pdelab_example.cc)
    # target_link_dune_default_libraries(pdelab_example)
    # target_link_libraries(pdelab_example PUBLIC duneddm)
    # dune_symlink_to_source_files(FILES pdelab_example.ini)
    # dune_symlink_to_source_files(FILES tutorial06.ini)
  endif()
else()
  message(WARNING "PDELab not found, not building examples")
endif()
