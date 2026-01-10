if (${dune-pdelab_FOUND})
  option(DUNE_DDM_BUILD_EXAMPLES "Try to build the examples" On)
  if (${DUNE_DDM_BUILD_EXAMPLES})
    # Poisson example
    add_executable(poisson EXCLUDE_FROM_ALL poisson.cc)
    target_link_libraries(poisson PUBLIC duneddm -llapacke)
    add_dune_blas_lapack_flags(poisson)
    target_link_dune_default_libraries(poisson)
    dune_symlink_to_source_files(FILES poisson.ini)

    # Linear elasticity example
    add_executable(linearelasticity EXCLUDE_FROM_ALL linearelasticity.cc)
    target_link_libraries(linearelasticity PUBLIC duneddm -llapacke)
    target_compile_definitions(linearelasticity PUBLIC DUNE_DDM_HAVE_LUA)
    dune_symlink_to_source_files(FILES coefficient.lua)
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
    target_link_libraries(convectiondiffusiondg PUBLIC duneddm)
    add_dune_blas_lapack_flags(convectiondiffusiondg)
    target_link_dune_default_libraries(convectiondiffusiondg)
    dune_symlink_to_source_files(FILES convectiondiffusiondg.ini)

    # PDELab example demonstrating the unified DDM framework
    add_executable(pdelab_example EXCLUDE_FROM_ALL pdelab_example.cc)
    target_link_dune_default_libraries(pdelab_example)
    target_link_libraries(pdelab_example PUBLIC duneddm -llapacke)
    dune_symlink_to_source_files(FILES pdelab_example.ini)
  endif()
else()
  message(WARNING "PDELab not found, not building examples")
endif()
