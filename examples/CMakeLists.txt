if (${dune-pdelab_FOUND})
  option(DUNE_DDM_BUILD_EXAMPLES "Try to build the examples" On)
  if (${DUNE_DDM_BUILD_EXAMPLES})
    # Poisson example
    add_executable(poisson EXCLUDE_FROM_ALL poisson.cc)
    target_link_libraries(poisson PUBLIC duneddm)
    add_dune_blas_lapack_flags(poisson)
    target_link_dune_default_libraries(poisson)
    dune_symlink_to_source_files(FILES poisson.ini)

    # Nonlinear Poisson example
    add_executable(nonlinearpoisson EXCLUDE_FROM_ALL nonlinearpoisson.cc)
    target_link_libraries(nonlinearpoisson PUBLIC duneddm)
    add_dune_blas_lapack_flags(nonlinearpoisson)
    target_link_dune_default_libraries(nonlinearpoisson)
    dune_symlink_to_source_files(FILES nonlinearpoisson.ini)

    # PDELab example demonstrating the unified DDM framework
    add_executable(pdelab_example EXCLUDE_FROM_ALL pdelab_example.cc)
    dune_symlink_to_source_files(FILES poisson_coefficient.lua
                                       convection_diffusion_coefficient.lua
                                       symmetric_convection_diffusion_coefficient.lua
    )
    target_link_libraries(pdelab_example PUBLIC duneddm)
    target_link_dune_default_libraries(pdelab_example)
    dune_symlink_to_source_files(FILES pdelab_example.ini)
  endif()
else()
  message(WARNING "PDELab not found, not building examples")
endif()
