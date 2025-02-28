set(MPIEXEC_PREFLAGS "--oversubscribe")
set(DUNE_MAX_TEST_CORES 32)
set(HAVE_MPI TRUE) # The dune_add_test macro only enables MPI tests if HAVE_MPI is set to TRUE; in Dune Common, this is set somewhere in their CMake config, we just set it here manually

dune_declare_test_label(LABELS ddm)

dune_add_test(SOURCES test_build_matrix.cc
              LINK_LIBRARIES dune-ddm
              MPI_RANKS 1 4 16 32
              TIMEOUT 100
              LABELS ddm
)

dune_add_test(SOURCES test_galerkin_coarse_matrix.cc
              LINK_LIBRARIES dune-ddm
              MPI_RANKS 4               # This test requires exactly 4 MPI ranks
              TIMEOUT 100
              LABELS ddm
)