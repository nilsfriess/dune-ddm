if (${dune-pdelab_FOUND})
  option(DUNE_DDM_BUILD_EXAMPLES "Try to build the examples" On)
  if (${DUNE_DDM_BUILD_EXAMPLES})
    # Nonlinear Poisson example
    add_executable(nonlinearpoisson EXCLUDE_FROM_ALL nonlinearpoisson.cc)
    target_link_libraries(nonlinearpoisson PUBLIC duneddm)
    add_dune_blas_lapack_flags(nonlinearpoisson)
    target_link_dune_default_libraries(nonlinearpoisson)
    dune_symlink_to_source_files(FILES nonlinearpoisson.ini)

    # PDELab example demonstrating the unified DDM framework
    add_executable(pdelab_example EXCLUDE_FROM_ALL pdelab_example.cc)
    dune_symlink_to_source_files(FILES
      coefficient.lua
      convection_diffusion_coefficient.lua
      layers_coefficient_3d.lua
      layers_coefficient.lua
      poisson_coefficient.lua
      symmetric_convection_diffusion_coefficient.lua
    )
    target_link_libraries(pdelab_example PUBLIC duneddm)
    # pdelab_example is the only example that uses Lua (via the *_problems.hh
    # headers). LuaJIT is an examples-only dependency, so it is linked here
    # rather than through the duneddm interface.
    target_link_libraries(pdelab_example PRIVATE luajit)
    target_link_dune_default_libraries(pdelab_example)
    dune_symlink_to_source_files(FILES pdelab_example.ini)

    add_executable(matrix_market_test EXCLUDE_FROM_ALL matrix_market_test.cc)
    target_link_libraries(matrix_market_test PUBLIC duneddm)
    target_link_dune_default_libraries(matrix_market_test)
    dune_symlink_to_source_files(FILES matrix_market_test.ini)
  endif()
else()
  message(WARNING "PDELab not found, not building examples")
endif()
