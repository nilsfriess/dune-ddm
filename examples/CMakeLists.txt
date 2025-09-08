if (${dune-pdelab_FOUND})
  add_executable(poisson poisson.cc)
  target_link_dune_default_libraries(poisson)
  target_link_libraries(poisson PUBLIC Taskflow Dune::DDM)
  dune_symlink_to_source_files(FILES poisson.ini)

  add_executable(pdelab_example pdelab_example.cc)
  target_link_dune_default_libraries(pdelab_example)
  target_link_libraries(pdelab_example PUBLIC Taskflow Dune::DDM)
  dune_symlink_to_source_files(FILES pdelab_example.ini)
else()
  message(WARNING "PDELab not found, not building examples")
endif()

# add_executable(gpu_solvers gpu_solvers.cc)
# target_link_libraries(gpu_solvers PUBLIC dune-ddm Dune::PDELab)
# add_dune_all_flags(gpu_solvers)

