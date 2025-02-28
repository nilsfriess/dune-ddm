enable_testing()

set(MPIEXEC_PREFLAGS "--oversubscribe")
set(DUNE_MAX_TEST_CORES 32)

dune_declare_test_label(LABELS ddm)

dune_add_test(SOURCES test_build_matrix.cc
              LINK_LIBRARIES dune-ddm
              MPI_RANKS 1 4 16 32
              TIMEOUT 100
              LABELS ddm
)

dune_add_test(SOURCES test_galerkin_coarse_matrix.cc
              LINK_LIBRARIES dune-ddm
              MPI_RANKS 1 2
              TIMEOUT 100
              LABELS ddm
)