# Algebraic two-level method from an assembled matrix, then dune-ddm as a PETSc PC

- STATUS: OPEN
- PRIORITY: 60
- TAGS: roadmap,algebraic,petsc

- Decide which Al Daas-Jolivet method: Al Daas-Jolivet 2022 (SISC, SPD, algebraic multilevel) or Al Daas-Jolivet-Rees
  2023 (SISC, general sparse). The latter may be what PCHPDDM's `-pc_hpddm_harmonic_overlap` implements; its local
  problems on an extra overlap layer look close to MS-GFEM's oversampling. Verify before building on it.
- Needs a grid-free entry point: a `Mat` built from a row-distributed assembled matrix, without element matrices. Today
  only the FE path exists.
- Old code to port: `algebraic_neumann.hh` (deleted in dfcf2a9; aligned with PCHPDDM `-pc_hpddm_block_splitting`, with
  an open TODO that it differs from the paper) and `overlap_extension.hh` on `origin/algebraic`.
- Caveat (2026-11-01 analysis): a general assembled SPD matrix cannot be split exactly from local information. Judge
  approximations by the eigenvalue error of the first non-kernel modes against the exact Neumann matrices ("patch
  test"). Exact for graph Laplacians, SDD matrices and A^T A.
- Compare against PCHPDDM through 20261126-092821.
- Then P4 from the 2026-10-30 ideas list: register two-level Schwarz as a PETSc PC. Needs the AIJ -> `Mat` direction of
  20261126-092821's conversion, plus this algebraic splitting or a Neumann callback like `PCHPDDMSetAuxiliaryMat`.

Related: Pillar A task "Algebraic MsGFEM experiments" (20260917-150154 in `~/Dokumente/PhD/tasks`).
