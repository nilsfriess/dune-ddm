# SYCL GPU backend, setup path first (eigensolves and cuDSS direct solves)

- STATUS: OPEN
- PRIORITY: 75
- TAGS: roadmap,gpu,backend

The flagship item of Pillar B. Gate: 20261126-092818. Builds on 20261126-092819 and 20261126-092822.

- Backend: `SyclMat` stub (`include/ddm/impl/mat/syclmat.hh`, `src/sycl/`, untracked). About 800 lines of older
  AdaptiveCpp kernels (vectors, CSR mat-vec, pack/unpack) and a working cuDSS interop were removed in dfcf2a9 and can
  be revived. One level of dispatch: `SyclMultiVec`/`SyclMat` implement the `do_*` ops with plain SYCL kernels (testable
  on the CPU backend), and vendor calls (cuBLAS, cuDSS) override them where available.
- Setup path first: the profile of two_sided ring puts 67% of the eigensolver time in the P/P^T (A_dir) solves. So local
  direct solves and block Lanczos go on the device first (cuDSS, batched over the local subdomains), then the apply.
- Mixed precision: try FP32 for local factorizations and shift-invert eigensolves; a coarse space needs a good subspace,
  not accurate eigenpairs. `SyclMat` already instantiates `float`.
- MPI: the `ExchangePlan` hook (`MultiVec::do_make_exchange_plan`) covers `Communication` exchanges. The neighbour basis
  exchange and the coarse gather/scatter still go through host views (TODO(20261126-093234) in `galerkin.hh`).
  Related: 20260913-151931.
- AMD later: HIP mirrors the CUDA libraries, but no ROCm sparse direct solver was found; STRUMPACK might be the fallback.
