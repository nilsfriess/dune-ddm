# Check complex<double> SYCL kernels and cuDSS support before committing to SYCL

- STATUS: OPEN
- PRIORITY: 100
- TAGS: roadmap,gpu

Deciding check for the GPU backend (20261126-092823). The 2026-11-24 backend discussion chose SYCL over CUDA and Kokkos on the
condition that complex numbers work on the device.

- Write a ~10-line `complex<double>` axpy/dot kernel and run it with the SYCL toolchain (AdaptiveCpp or DPC++) on the
  NVIDIA machine. SYCL 2020 does not guarantee `std::complex` in device code, and Helmholtz needs it
  (`complex<double>` is registered in `src/initialize.cc`).
- Check cuDSS: `complex<double>` support, and batched factorization/solve of many matrices with different sparsity
  patterns (several subdomains per rank, 20261126-092822).
- Note the GPU model. Consumer cards run FP64 at 1/32-1/64 of the FP32 rate, which makes mixed precision in 20261126-092823 much
  more attractive.
- If complex turns out to be painful: switch to Kokkos (Kokkos Kernels already dispatches to vendor libraries). The
  roadmap order stays the same.
