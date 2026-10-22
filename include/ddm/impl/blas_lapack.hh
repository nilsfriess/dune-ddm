#pragma once

#include <complex>

// C++ wrappers of the BLAS and LAPACK routines used by dune-ddm. The routines themselves are only declared in
// src/blas_lapack.cc: other headers declare some of them as well (e.g. SuiteSparse_config.h, included by umfpack.h),
// with signatures that depend on their configuration and would clash with our declarations. Assumes a BLAS and LAPACK
// with 32-bit integers (LP64), which FindBLAS and FindLAPACK find by default.
namespace ddm::detail::blas {
// C = alpha op(A) op(B) + beta C for column-major matrices, where op(A) is m x k and op(B) is k x n. trans_a and trans_b
// select op: 'N' for A, 'T' for A^T and 'C' for A^H (the same as 'T' for real matrices). As in BLAS, C is not read if
// beta is zero.
void gemm(char trans_a, char trans_b, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
void gemm(char trans_a, char trans_b, int m, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc);
void gemm(char trans_a, char trans_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, std::complex<float> beta,
          std::complex<float>* C, int ldc);
void gemm(char trans_a, char trans_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, std::complex<double> beta,
          std::complex<double>* C, int ldc);
} // namespace ddm::detail::blas

namespace ddm::detail::lapack {
// The eigenpairs il, ..., iu (counted from 1, in the order of increasing eigenvalues) of A x = lambda B x, where A is
// symmetric (Hermitian) and B symmetric (Hermitian) positive definite, both n x n and column-major. Only the lower
// triangles are read, and both matrices are overwritten. w (iu - il + 1 entries) gets the eigenvalues in increasing
// order, Z (n x (iu - il + 1), column-major) the eigenvectors, normalized to Z^H B Z = I. Calls ?sygvx (?hegvx for
// complex matrices) and returns its info, which is zero on success.
int sygvx(int n, float* A, float* B, int il, int iu, float* w, float* Z);
int sygvx(int n, double* A, double* B, int il, int iu, double* w, double* Z);
int sygvx(int n, std::complex<float>* A, std::complex<float>* B, int il, int iu, float* w, std::complex<float>* Z);
int sygvx(int n, std::complex<double>* A, std::complex<double>* B, int il, int iu, double* w, std::complex<double>* Z);
} // namespace ddm::detail::lapack
