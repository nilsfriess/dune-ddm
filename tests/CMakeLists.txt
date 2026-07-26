set(MPIEXEC_PREFLAGS "--oversubscribe")
set(DUNE_MAX_TEST_CORES 32)
set(HAVE_MPI TRUE) # The dune_add_test macro only enables MPI tests if HAVE_MPI is set to TRUE; in Dune Common, this is set somewhere in their CMake config, we just set it here manually

dune_declare_test_label(LABELS ddm)

# Test that the partition-of-unity constructions (in particular MsFEM) sum to 1 across all
# subdomains (in parallel).
dune_add_test(SOURCES test_pou.cc
  LINK_LIBRARIES duneddm
  MPI_RANKS 2 4 16
  TIMEOUT 100
  LABELS ddm
)
            
dune_add_test(SOURCES test_build_matrix.cc
  LINK_LIBRARIES duneddm
  MPI_RANKS 1 2 4 16
  TIMEOUT 100
  LABELS ddm
)

# Pure helpers (detect_dirichlet_dofs, get_parameter_tree_prefix); nothing parallel to exercise.
dune_add_test(SOURCES test_helpers.cc
  LINK_LIBRARIES duneddm
  MPI_RANKS 1
  TIMEOUT 100
  LABELS ddm
)

# distributeMatrixFrom0: owned rows must reassemble into the original matrix.
dune_add_test(SOURCES test_distribute_matrix.cc
  LINK_LIBRARIES duneddm
  MPI_RANKS 1 2 4 16
  TIMEOUT 100
  LABELS ddm
)

# GalerkinPreconditioner: the assembled coarse matrix must equal R A R^T and apply() must evaluate
# R^T (R A R^T)^-1 R, both compared against a dense reference built from the global matrix.
dune_add_test(SOURCES test_galerkin_coarse_matrix.cc
  LINK_LIBRARIES duneddm
  MPI_RANKS 1 2 4 16
  TIMEOUT 300
  LABELS ddm
)
