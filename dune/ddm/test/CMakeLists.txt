dune_declare_test_label(LABELS ddm)
link_libraries(duneddm)

####
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC PETSc>=3.17 IMPORTED_TARGET REQUIRED)
pkg_search_module(SLEPC SLEPc>=3.17 IMPORTED_TARGET REQUIRED)
####

dune_add_test(SOURCES test_eigensolver.cc NAME test_eigensolver LABELS ddm)
target_include_directories(test_eigensolver SYSTEM PUBLIC ${PETSC_INCLUDE_DIRS} ${SLEPC_INCLUDE_DIRS})
target_compile_options(test_eigensolver PUBLIC ${PETSC_CFLAGS} ${SLEPC_CFLAGS})
target_link_libraries(test_eigensolver PUBLIC PkgConfig::PETSC PkgConfig::SLEPC)

dune_add_test(SOURCES test_blocksolver.cc LABELS ddm)

dune_add_test(SOURCES test_lanczos_step.cc NAME test_lanczos_step LABELS ddm)

dune_add_test(SOURCES test_ortho_minimal.cc NAME test_ortho_minimal LABELS ddm)

dune_add_test(SOURCES debug_dot.cc NAME debug_dot LABELS ddm)

dune_symlink_to_source_files(FILES A.mtx B.mtx)
