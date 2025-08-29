add_executable(poisson poisson.cc)
target_link_libraries(poisson PUBLIC dune-ddm Dune::PDELab Taskflow)
add_dune_all_flags(poisson)

dune_symlink_to_source_files(FILES poisson.ini)
add_executable(gpu_solvers gpu_solvers.cc)
target_link_libraries(gpu_solvers PUBLIC dune-ddm Dune::PDELab STRUMPACK::strumpack)
add_dune_all_flags(gpu_solvers)

