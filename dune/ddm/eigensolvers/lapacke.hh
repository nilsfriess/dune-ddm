#pragma once

#include <lapacke.h>

namespace lapacke {
inline int potrf(int matrix_layout, char uplo, int n, double *a, int lda) { return LAPACKE_dpotrf(matrix_layout, uplo, n, a, lda); }

inline int potrf(int matrix_layout, char uplo, int n, float *a, int lda) { return LAPACKE_spotrf(matrix_layout, uplo, n, a, lda); }

inline int syevd(int matrix_layout, char jobz, char uplo, int n, double *a, int lda, double *w) { return LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, a, lda, w); }

inline int syevd(int matrix_layout, char jobz, char uplo, int n, float *a, int lda, float *w) { return LAPACKE_ssyevd(matrix_layout, jobz, uplo, n, a, lda, w); }
} // namespace lapacke
