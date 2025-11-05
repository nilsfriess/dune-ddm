if (${dune-pdelab_FOUND})
  option(DUNE_DDM_BUILD_EXAMPLES "Try to build the examples" On)
  if (${DUNE_DDM_BUILD_EXAMPLES})
    add_executable(poisson EXCLUDE_FROM_ALL poisson.cc)
    target_link_dune_default_libraries(poisson)
    target_link_libraries(poisson PUBLIC duneddm)
    
    target_link_libraries(poisson PUBLIC duneddm)
    dune_symlink_to_source_files(FILES poisson.ini)
  
    add_executable(pdelab_example EXCLUDE_FROM_ALL pdelab_example.cc)
    target_link_dune_default_libraries(pdelab_example)
    target_link_libraries(pdelab_example PUBLIC duneddm)
    dune_symlink_to_source_files(FILES pdelab_example.ini)
    dune_symlink_to_source_files(FILES tutorial06.ini)
  endif()
else()
  message(WARNING "PDELab not found, not building examples")
endif()

# add_executable(gpu_solvers gpu_solvers.cc)
# target_link_libraries(gpu_solvers PUBLIC dune-ddm Dune::PDELab)
# add_dune_all_flags(gpu_solvers)

