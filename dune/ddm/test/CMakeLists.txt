dune_declare_test_label(LABELS ddm)
link_libraries(duneddm)

dune_add_test(SOURCES test_eigensolver.cc LABELS ddm)
dune_add_test(SOURCES test_blocksolver.cc LABELS ddm)

dune_symlink_to_source_files(FILES A.mtx B.mtx)
