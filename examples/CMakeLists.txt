add_executable(poisson poisson.cc)
target_link_libraries(poisson PUBLIC dune-ddm Dune::UGGrid Dune::PDELab)
add_dune_all_flags(poisson)
