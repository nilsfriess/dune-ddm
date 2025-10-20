#pragma once

#include <lapacke.h>

namespace lapacke {
inline int potrf(int matrix_layout, char uplo, int n, double* a, int lda) { return LAPACKE_dpotrf(matrix_layout, uplo, n, a, lda); }

inline int potrf(int matrix_layout, char uplo, int n, float* a, int lda) { return LAPACKE_spotrf(matrix_layout, uplo, n, a, lda); }

inline int syevd(int matrix_layout, char jobz, char uplo, int n, double* a, int lda, double* w) { return LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, a, lda, w); }

inline int syev(int matrix_layout, char jobz, char uplo, int n, double* a, int lda, double* w) { return LAPACKE_dsyev(matrix_layout, jobz, uplo, n, a, lda, w); }

inline int syevd(int matrix_layout, char jobz, char uplo, int n, float* a, int lda, float* w) { return LAPACKE_ssyevd(matrix_layout, jobz, uplo, n, a, lda, w); }

inline int gecon(int matrix_layout, char norm, int32_t n, const double* a, int32_t lda, double anorm, double* rcond) { return LAPACKE_dgecon(matrix_layout, norm, n, a, lda, anorm, rcond); }

inline int geqrf(int matrix_layout, int m, int n, double* a, int lda, double* tau) { return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau); }

inline int geqrf(int matrix_layout, int m, int n, float* a, int lda, float* tau) { return LAPACKE_sgeqrf(matrix_layout, m, n, a, lda, tau); }

inline int orgqr(int matrix_layout, int m, int n, int k, double* a, int lda, const double* tau) { return LAPACKE_dorgqr(matrix_layout, m, n, k, a, lda, tau); }

inline int orgqr(int matrix_layout, int m, int n, int k, float* a, int lda, const float* tau) { return LAPACKE_sorgqr(matrix_layout, m, n, k, a, lda, tau); }

// inline int gehrd(int matrix_layout, int32_t n, int32_t ilo, int32_t ihi, double* a, int32_t lda, double* tau) { return LAPACKE_dgehrd(matrix_layout, n, ilo, ihi, a, lda, tau); }

// inline int gehrd(int matrix_layout, int32_t n, int32_t ilo, int32_t ihi, float* a, int32_t lda, float* tau) { return LAPACKE_sgehrd(matrix_layout, n, ilo, ihi, a, lda, tau); }

} // namespace lapacke
