#include "ddm/impl/blas_lapack.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

// The Fortran BLAS and LAPACK routines, see blas_lapack.hh for why they are declared here. The LAPACK routines get the
// lengths of their character arguments as hidden trailing arguments, as gfortran passes them.
extern "C" {
void sgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const float* alpha, const float* A, const int* lda, const float* B, const int* ldb, const float* beta,
            float* C, const int* ldc);
void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const double* alpha, const double* A, const int* lda, const double* B, const int* ldb, const double* beta,
            double* C, const int* ldc);
void cgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const std::complex<float>* alpha, const std::complex<float>* A, const int* lda,
            const std::complex<float>* B, const int* ldb, const std::complex<float>* beta, std::complex<float>* C, const int* ldc);
void zgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const std::complex<double>* alpha, const std::complex<double>* A, const int* lda,
            const std::complex<double>* B, const int* ldb, const std::complex<double>* beta, std::complex<double>* C, const int* ldc);

void ssygvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, float* A, const int* lda, float* B, const int* ldb, const float* vl, const float* vu,
             const int* il, const int* iu, const float* abstol, int* m, float* w, float* Z, const int* ldz, float* work, const int* lwork, int* iwork, int* ifail, int* info, std::size_t, std::size_t,
             std::size_t);
void dsygvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, double* A, const int* lda, double* B, const int* ldb, const double* vl, const double* vu,
             const int* il, const int* iu, const double* abstol, int* m, double* w, double* Z, const int* ldz, double* work, const int* lwork, int* iwork, int* ifail, int* info, std::size_t,
             std::size_t, std::size_t);
void chegvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, std::complex<float>* A, const int* lda, std::complex<float>* B, const int* ldb, const float* vl,
             const float* vu, const int* il, const int* iu, const float* abstol, int* m, float* w, std::complex<float>* Z, const int* ldz, std::complex<float>* work, const int* lwork, float* rwork,
             int* iwork, int* ifail, int* info, std::size_t, std::size_t, std::size_t);
void zhegvx_(const int* itype, const char* jobz, const char* range, const char* uplo, const int* n, std::complex<double>* A, const int* lda, std::complex<double>* B, const int* ldb,
             const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, double* w, std::complex<double>* Z, const int* ldz, std::complex<double>* work,
             const int* lwork, double* rwork, int* iwork, int* ifail, int* info, std::size_t, std::size_t, std::size_t);
}

namespace ddm::detail::blas {
void gemm(char trans_a, char trans_b, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{
  sgemm_(&trans_a, &trans_b, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

void gemm(char trans_a, char trans_b, int m, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc)
{
  dgemm_(&trans_a, &trans_b, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

void gemm(char trans_a, char trans_b, int m, int n, int k, std::complex<float> alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb, std::complex<float> beta,
          std::complex<float>* C, int ldc)
{
  cgemm_(&trans_a, &trans_b, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

void gemm(char trans_a, char trans_b, int m, int n, int k, std::complex<double> alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, std::complex<double> beta,
          std::complex<double>* C, int ldc)
{
  zgemm_(&trans_a, &trans_b, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
} // namespace ddm::detail::blas

namespace ddm::detail::lapack {
namespace {
// See sygvx() in blas_lapack.hh. Calls LAPACK twice: to query the size of the workspace, then to compute the eigenpairs.
template <class T, class Real>
int sygvx_impl(int n, T* A, T* B, int il, int iu, Real* w, T* Z)
{
  const int itype = 1; // A x = lambda B x
  const char jobz = 'V';
  const char range = 'I';
  const char uplo = 'L';
  const int ld = std::max(1, n);
  const Real vl = 0;
  const Real vu = 0;
  const Real abstol = 0; // LAPACK's default tolerance
  int m = 0;
  int info = 0;

  // LAPACK needs room for all n eigenvalues
  std::vector<Real> w_all(ld);
  std::vector<int> iwork(5 * ld);
  std::vector<int> ifail(ld);
  std::vector<Real> rwork(7 * ld); // only used for complex matrices

  const auto call = [&](T* work, int lwork) {
    if constexpr (std::is_same_v<T, float>)
      ssygvx_(&itype, &jobz, &range, &uplo, &n, A, &ld, B, &ld, &vl, &vu, &il, &iu, &abstol, &m, w_all.data(), Z, &ld, work, &lwork, iwork.data(), ifail.data(), &info, 1, 1, 1);
    else if constexpr (std::is_same_v<T, double>)
      dsygvx_(&itype, &jobz, &range, &uplo, &n, A, &ld, B, &ld, &vl, &vu, &il, &iu, &abstol, &m, w_all.data(), Z, &ld, work, &lwork, iwork.data(), ifail.data(), &info, 1, 1, 1);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
      chegvx_(&itype, &jobz, &range, &uplo, &n, A, &ld, B, &ld, &vl, &vu, &il, &iu, &abstol, &m, w_all.data(), Z, &ld, work, &lwork, rwork.data(), iwork.data(), ifail.data(), &info, 1, 1, 1);
    else zhegvx_(&itype, &jobz, &range, &uplo, &n, A, &ld, B, &ld, &vl, &vu, &il, &iu, &abstol, &m, w_all.data(), Z, &ld, work, &lwork, rwork.data(), iwork.data(), ifail.data(), &info, 1, 1, 1);
  };

  T work_size{};
  call(&work_size, -1);
  if (info != 0) return info;
  std::vector<T> work(std::max<std::size_t>(1, static_cast<std::size_t>(std::real(work_size))));
  call(work.data(), static_cast<int>(work.size()));
  std::copy_n(w_all.begin(), m, w);
  return info;
}
} // namespace

int sygvx(int n, float* A, float* B, int il, int iu, float* w, float* Z) { return sygvx_impl(n, A, B, il, iu, w, Z); }
int sygvx(int n, double* A, double* B, int il, int iu, double* w, double* Z) { return sygvx_impl(n, A, B, il, iu, w, Z); }
int sygvx(int n, std::complex<float>* A, std::complex<float>* B, int il, int iu, float* w, std::complex<float>* Z) { return sygvx_impl(n, A, B, il, iu, w, Z); }
int sygvx(int n, std::complex<double>* A, std::complex<double>* B, int il, int iu, double* w, std::complex<double>* Z) { return sygvx_impl(n, A, B, il, iu, w, Z); }
} // namespace ddm::detail::lapack
