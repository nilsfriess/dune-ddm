# PETSc interface stages 1-2: PetscPrec and the parallel matrix conversion

- STATUS: OPEN
- PRIORITY: 85
- TAGS: roadmap,petsc

Design from the 2026-11-10 Claude Code session (transcript `cbe6433b` under `~/.claude/projects/-home-friessn-Projects-dune-ddm/`).
Purpose: the AMG baselines (GAMG, BoomerAMG, PCHPDDM) on the same matrices, Krylov solver and stopping criterion, and
MUMPS/hypre as local, coarse (20261126-092820) and shift-invert solvers.

- `PetscPrec<T>`, registered as prec type `petsc`: owns a KSP, `ksp_type preonly` by default.
- Stage 1: sequential `PetscPrec`, plus the plumbing:
  - CMake: optional, `pkg_check_modules(PETSc IMPORTED_TARGET PETSc)`, `DUNE_DDM_HAVE_PETSC`.
  - Options: copy `config.sub("petsc")` into the options database with a unique prefix per object, then
    `KSPSetFromOptions`.
  - Init: RAII `ddm::PetscScope` right after `MPIHelper::instance`; the factories check `PetscInitialized()`.
  - Test: matches `lu`.
- Stage 2: parallel conversion.
  - Global numbering: owned indices (`owner_mask`) + `MPI_Exscan` + `Communication::broadcast` -> `ISLocalToGlobalMapping`.
  - Matrix: `MatCreateIS` with a SeqAIJ from `local().host_csr()`; `MatConvert` to AIJ (keep MATIS for BDDC);
    `update()` reuses the pattern (`MAT_REUSE_MATRIX`).
  - Vectors: gather the owned entries; back: scatter, zero the rest, `comm->broadcast`.
  - Test: `MatMult` on the converted matrix matches `Mat::apply`.
- Gotchas:
  - Complex transposed solves: PETSc's transpose is A^T, our wrappers use A^H; conjugate around `KSPSolveTranspose`.
  - One scalar type per `PETSC_ARCH`: complex Helmholtz needs a second build. Check `PetscInt` vs `Index`.
  - Build PETSc against the same MKL and METIS/ParMETIS as dune-ddm.
  - Stopping criteria differ from ISTL GMRES (norm type, relative to |b|): set them explicitly before comparing iterations.
- Later (stage 3): `PetscSolver<T>` (PCNONE / native PC / PCSHELL + MATSHELL around our preconditioners), `KSPMatSolve`
  for several right-hand sides.
- The reverse direction (dune-ddm as a PETSc PC) is part of 20261126-092824.
