#pragma once

#ifdef DUNE_DDM_HAVE_MKL_PARDISO

#include "ddm/check.hh"
#include "ddm/impl/mat/istlmat.hh"
#include "ddm/impl/multivec/istlmultivec.hh"
#include "ddm/index.hh"
#include "ddm/logger.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/prec/prec.hh"
#include "ddm/vec/vec.hh"

#include <array>
#include <dune/common/parametertree.hh>
#include <memory>
#include <mkl_pardiso.h>
#include <mkl_types.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ddm {
/** Sparse direct solver PARDISO from Intel MKL. Sequential matrices only. MKL is linked without threading (see
 *  DuneDdmMacros.cmake), so PARDISO runs on one thread.
 *
 *  Config:
 *   - mtype:    PARDISO's matrix type (default: 11 for real, 13 for complex T, i.e. nonsymmetric). For the symmetric
 *               and Hermitian types (2, -2, 4, -4, 6), PARDISO is only given the upper triangle of the matrix, which
 *               still has to store all entries.
 *   - msglvl:   1 prints PARDISO's statistics
 *   - iparm<i>: sets iparm[i] (0-based, as in MKL's C documentation), overriding the defaults set below. Values that
 *               PARDISO does not accept are not caught.
 *  As for UMFPACK and STRUMPACK, there is no iterative refinement by default (iparm7 = 0), so the preconditioner is a
 *  fixed linear operator. PARDISO still refines twice when it had to perturb pivots.
 *
 *  apply_transposed() solves with A^T, or A^H for complex T (like UMFPACK).
 */
template <class T>
class PardisoPrec final : public Prec<T> {
  static constexpr bool is_complex = !std::is_floating_point_v<T>;

public:
  PardisoPrec(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A)
      : Prec<T>(std::move(A))
      , factorize_event_(Logger::get().registerOrGetEvent("PARDISO", "factorize"))
      , solve_event_(Logger::get().registerOrGetEvent("PARDISO", "solve"))
      , solve_multi_event_(Logger::get().registerOrGetEvent("PARDISO", "solve_multi"))
  {
    // TODO(20261006-192425): It should be possible to mark a preconditioner or solver as "sequential" on a higher level
    DDM_CHECK(this->mat()->sequential(), "PARDISO is a sequential solver");
    read_options(config);
    do_update();
  }

  ~PardisoPrec() override
  {
    release();
  }

private:
  void do_apply(Vec<T>& z, const Vec<T>& r) override
  {
    Logger::ScopedLog sl{solve_event_};
    solve(0, 1, as_istl(r).data().data(), as_istl(z).data().data());
  }

  void do_apply_multi(MultiVec<T>& Z, const MultiVec<T>& R) override
  {
    Logger::ScopedLog sl{solve_multi_event_};
    if (R.cols() == 0) return;
    solve(0, R.cols(), as_istl(R).data().data(), as_istl(Z).data().data());
  }

  void do_apply_transposed(Vec<T>& z, const Vec<T>& r) override
  {
    Logger::ScopedLog sl{solve_event_};
    solve(transposed_system, 1, as_istl(r).data().data(), as_istl(z).data().data());
  }

  void do_apply_transposed_multi(MultiVec<T>& Z, const MultiVec<T>& R) override
  {
    Logger::ScopedLog sl{solve_multi_event_};
    if (R.cols() == 0) return;
    solve(transposed_system, R.cols(), as_istl(R).data().data(), as_istl(Z).data().data());
  }

  // The first call analyses and factorizes the matrix, later ones only factorize it, so PARDISO reuses the analysis
  void do_update() override
  {
    Logger::ScopedLog sl{factorize_event_};
    set_matrix(this->mat()->local());
    if (n_ == 0) return;

    const MKL_INT phase = analysed_ ? 22 : 12;
    T dummy{};
    const auto error = call(phase, 1, &dummy, &dummy);
    if (error != 0) release(); // so the next update() starts with a fresh analysis
    DDM_CHECK(error == 0, "PARDISO: factorization (phase {}) failed with error {} (n = {}, nnz = {})", phase, error, n_, values_.size());
    analysed_ = true;
  }

  void do_info() const override
  {
    logger::info("PARDISO solver (mtype {})", mtype_);
  }

  // Matrix types of which PARDISO is given the upper triangle only. The structurally symmetric types 1 and 3 are not
  // among them, their values are not symmetric.
  bool upper_triangle_only() const
  {
    return mtype_ == 2 || mtype_ == -2 || mtype_ == 4 || mtype_ == -4 || mtype_ == 6;
  }

  void read_options(const Dune::ParameterTree& config)
  {
    mtype_ = config.get<MKL_INT>("mtype", is_complex ? 13 : 11);
    const bool complex_mtype = mtype_ == 3 || mtype_ == 4 || mtype_ == -4 || mtype_ == 6 || mtype_ == 13;
    DDM_CHECK(complex_mtype == is_complex, "PARDISO: mtype {} is for {} matrices", mtype_, complex_mtype ? "complex" : "real");
    msglvl_ = config.get<MKL_INT>("msglvl", 0);

    // Mostly the defaults of MKL's pardiso_unsym_c example. All other entries are 0.
    const bool scale_and_match = mtype_ == 11 || mtype_ == 13 || mtype_ == -2 || mtype_ == -4 || mtype_ == 6;
    iparm_[0] = 1;                                 // Use the values set here instead of PARDISO's defaults
    iparm_[1] = 2;                                 // Fill-in reducing ordering from METIS
    iparm_[7] = 0;                                 // No iterative refinement (see above)
    iparm_[9] = 13;                                // Perturb small pivots to 1e-13
    iparm_[10] = scale_and_match ? 1 : 0;          // Scaling
    iparm_[12] = scale_and_match ? 1 : 0;          // Weighted matching
    iparm_[20] = 1;                                // Bunch-Kaufman pivoting for symmetric indefinite matrices
    iparm_[27] = std::is_same_v<T, float> ? 1 : 0; // Precision of the matrix and vectors
    iparm_[34] = 1;                                // Zero-based indices

    for (int i = 1; i < 64; ++i) iparm_[i] = config.get<MKL_INT>("iparm" + std::to_string(i), iparm_[i]);
  }

  // Converts A to the CSR format PARDISO expects for mtype_: the upper triangle only for the symmetric types, and an
  // explicit (zero) diagonal entry in every row for those and the structurally symmetric types. Relies on the columns
  // of each row of host_csr() being sorted, which PARDISO requires as well.
  void set_matrix(const LocalMat<T>& A)
  {
    n_ = static_cast<MKL_INT>(A.rows());
    row_ptr_.assign(1, 0);
    cols_.clear();
    values_.clear();
    if (n_ == 0) return;

    const auto csr = A.host_csr();
    const bool upper_only = upper_triangle_only();
    const bool store_diagonal = upper_only || mtype_ == 1 || mtype_ == 3;
    const auto add = [&](Index j, T value) {
      cols_.push_back(static_cast<MKL_INT>(j));
      values_.push_back(value);
    };

    for (Index i = 0; i < n_; ++i) {
      bool diagonal_done = !store_diagonal;
      for (Index k = csr.row_ptr[i]; k < csr.row_ptr[i + 1]; ++k) {
        const Index j = csr.cols[k];
        if (upper_only && j < i) continue;
        if (!diagonal_done && j >= i) {
          if (j > i) add(i, T{});
          diagonal_done = true;
        }
        add(j, csr.values[k]);
      }
      if (!diagonal_done) add(i, T{});
      row_ptr_.push_back(static_cast<MKL_INT>(cols_.size()));
    }
  }

  // Solves with nrhs right-hand sides, stored one after the other in b, for x. system is 0 for A and transposed_system
  // for A^T/A^H.
  void solve(MKL_INT system, Index nrhs, const T* b, T* x)
  {
    if (n_ == 0) return;
    iparm_[11] = system;
    // With iparm[5] = 0, PARDISO does not write to b
    const auto error = call(33, static_cast<MKL_INT>(nrhs), const_cast<T*>(b), x);
    DDM_CHECK(error == 0, "PARDISO: solve failed with error {}", error);
  }

  MKL_INT call(MKL_INT phase, MKL_INT nrhs, T* b, T* x)
  {
    const MKL_INT maxfct = 1; // Only one factorization is kept
    const MKL_INT mnum = 1;   // and used
    MKL_INT perm = 0;         // Not used, iparm[4] = 0
    MKL_INT error = 0;
    pardiso(pt_.data(), &maxfct, &mnum, &mtype_, &phase, &n_, values_.data(), row_ptr_.data(), cols_.data(), &perm, &nrhs, iparm_.data(), &msglvl_, b, x, &error);
    return error;
  }

  // Frees PARDISO's memory, if it holds any (pt_ is zero until the first factorization)
  void release() noexcept
  {
    if (pt_ == decltype(pt_){}) return;
    T dummy{};
    call(-1, 1, &dummy, &dummy);
    pt_ = {};
    analysed_ = false;
  }

  static constexpr MKL_INT transposed_system = is_complex ? 1 : 2; // A^H for complex T, A^T otherwise

  Logger::Event* factorize_event_;
  Logger::Event* solve_event_;
  Logger::Event* solve_multi_event_;

  // The matrix in the CSR format of set_matrix(), PARDISO reads it in every phase
  MKL_INT n_ = 0;
  std::vector<MKL_INT> row_ptr_;
  std::vector<MKL_INT> cols_;
  std::vector<T> values_;

  std::array<void*, 64> pt_{}; // PARDISO's internal data
  std::array<MKL_INT, 64> iparm_{};
  MKL_INT mtype_ = 0;
  MKL_INT msglvl_ = 0;
  bool analysed_ = false;
};
} // namespace ddm

#endif
