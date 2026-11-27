# Distributed coarse solve instead of gathering A_0 on rank 0

- STATUS: OPEN
- PRIORITY: 90
- TAGS: roadmap,coarse,mpi,performance

Measured on Helix (Marmousi Helmholtz weak scaling: 2D, complex, f ~ sqrt(P), RAS + deflated, two_sided; results in
`~/Dokumente/PhD/Conferences/Bogota/Talk/results/raw/helmholtz_*_P*`): at P=512 the coarse correction setup, which runs
entirely on rank 0, dominates the setup.

- nev 75 (8 iterations): 10.8 of 19.9 s setup (54%), coarse factorization 8.0 s, 38,400 coarse dofs.
- nev 100: 26.3 of 38.8 s (68%). nev 150: 60 of 79 s (76%), factorization 51.6 s, 76,800 coarse dofs.
- Fixed nev 150 over P = 8/32/128/512: coarse factorization 0.1 / 0.8 / 5.1 / 51.6 s.
- The coarse correction is 70-90% of every preconditioner apply at P=512 (coarse solve on rank 0 + gather/scatter).
- Peak RSS on rank 0 is 9.7-17.4 GB; the largest on any other rank is 2.6-4.7 GB.
- Not only the factorization: rank 0's setup minus factorization is 2.8 / 9.3 / 8.5 s for nev 75/100/150, and 28 s
  for ring nev 150 with the same factorization. Unexplained; profile the gather and the CSR assembly.

Plan:
- `coarse_matrix_rows()` (`galerkin.hh`) already computes A_0's block rows on every rank. Hand them to a distributed
  sparse direct solver (all ranks or a subcommunicator) instead of `MPI_Gatherv` to rank 0;
  `gather_coarse_defect()`/`scatter_coarse_solution()` become a redistribution.
- Solver backend still to decide, none verified: MUMPS via PETSc (20261126-092821); MKL's `cluster_sparse_solver` (MKL is already
  linked, needs the BLACS library for the MPI in use); STRUMPACK's MPI-distributed solver (vendored, check that it is
  built with MPI).
- Rerun the P=512 rows afterwards.

Related: 20260917-103910 (old `galerkin_product` task), 20261126-092826 (A_0 as a distributed additive `Mat`).
