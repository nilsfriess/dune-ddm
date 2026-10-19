if (${dune-pdelab_FOUND})
   # 3d linear elasticity example
   add_executable(ex1 ex1.cc)
   target_link_libraries(ex1 PUBLIC duneddm)
   target_link_dune_default_libraries(ex1)

   # 2d convection-diffusion example
   add_executable(ex2 ex2.cc)
   target_link_libraries(ex2 PUBLIC duneddm)
   target_link_dune_default_libraries(ex2)

   # 2d diffusion example with DG
   add_executable(ex3 ex3.cc)
   target_link_libraries(ex3 PUBLIC duneddm)
   target_link_dune_default_libraries(ex3)

   # 2d Helmholtz example
   add_executable(ex4 ex4.cc)
   target_link_libraries(ex4 PUBLIC duneddm)
   target_link_dune_default_libraries(ex4)

   # 2d Helmholtz example on the BP 2004 velocity model, which it reads from data/
   add_executable(ex5 ex5.cc)
   target_link_libraries(ex5 PUBLIC duneddm)
   target_link_dune_default_libraries(ex5)
   target_compile_definitions(ex5 PRIVATE DDM_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
endif()
