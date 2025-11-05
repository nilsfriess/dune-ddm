#pragma once

#include <cblas.h>

namespace blas {

inline void gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta, double *C, const int ldc)
{
  cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta, float *C, const int ldc)
{
  cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void trsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const int M, const int N, const double alpha,
                 const double *A, const int lda, double *B, const int ldb)
{
  cblas_dtrsm(Layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}

inline void trsm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag, const int M, const int N, const float alpha,
                 const float *A, const int lda, float *B, const int ldb)
{
  cblas_strsm(Layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);
}
} // namespace blas
