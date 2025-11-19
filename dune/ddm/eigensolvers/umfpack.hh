#pragma once

#include "aligned_allocator.hh"

#include <cmath>
#include <cstdint>
#include <dune/common/exceptions.hh>
#include <dune/ddm/helpers.hh>
#include <dune/istl/umfpack.hh>
#include <experimental/simd>
#include <memory>
#include <string>
#include <umfpack.h>
#include <vector>

template <class Mat>
class UMFPackMultivecSolver {
public:
  explicit UMFPackMultivecSolver(const Mat& A, int max_refinement_iter = 3)
      : A(A)
      , A_norm(A.infinity_norm())
      , max_refinement_iter(max_refinement_iter)
  {
    solver.setOption(UMFPACK_ORDERING, UMFPACK_ORDERING_METIS);
    solver.setMatrix(A);
    init();
  }

  template <class Multivec>
  void operator()(Multivec& X, std::size_t block)
  {
    auto B = X;
    apply(X, B, block, block + 1);
  }

  template <class Multivec1, class Multivec2>
  void apply(Multivec1& xx, Multivec2& bb)
  {
    apply(xx, bb, 0, xx.blocks());
  }

  template <class Multivec1, class Multivec2>
  void apply(Multivec1& xx, Multivec2& bb, std::size_t block_from, std::size_t block_to)
  {
    auto bbefore = bb;

    const double tol = 1e-14; // More reasonable tolerance than machine epsilon
    double omega_prev = std::numeric_limits<double>::infinity();

    solve(xx, bb, block_from, block_to);
    for (int iter = 0; iter < max_refinement_iter; ++iter) {
      auto resid = xx;

      // Compute block-wise backward errors and take maximum
      double omega_max = 0.0;
      for (std::size_t block = block_from; block < block_to; ++block) {
        auto r_block = resid.block_view(block);
        auto x_block = xx.block_view(block);
        auto b_block = bbefore.block_view(block); // Read-only access to original RHS

        // Compute residual: r = b - A*x
        x_block.apply_to_mat(A, r_block); // r = A*x
        r_block *= -1.0;                  // r = -A*x
        r_block += b_block;               // r = b - A*x

        // Compute norms for this block
        auto r_norm = r_block.frobenius_norm();
        auto x_norm = x_block.frobenius_norm();
        auto b_norm = b_block.frobenius_norm();

        // Backward error for this block: ||r|| / (||A|| * ||x|| + ||b||)
        double denom = A_norm * x_norm + b_norm;
        double omega_block = r_norm / denom;

        omega_max = std::max(omega_max, omega_block);
      }

      // Stop if backward error is essentially zero (UMFPACK criterion)
      if (omega_max < tol) {
        // logger::debug("Converged: omega_max < machine epsilon");
        break;
      }

      // Stop if NaN detected
      if (std::isnan(omega_max)) {
        logger::debug("NaN detected, stopping refinement");
        break;
      }

      // Stop if insufficient improvement (UMFPACK requires 2x improvement, we do the same)
      if (iter > 0 && omega_max > omega_prev / 2.0) {
        logger::debug("Insufficient improvement, stopping refinement");
        break;
      }

      // Solve correction equation: A * delta = r
      // Note: resid now contains the actual residual r = b - A*x
      auto delta = resid;                        // Copy residual to delta
      solve(delta, resid, block_from, block_to); // Solve A * delta = r

      // Update solution: x = x + delta
      for (std::size_t block = block_from; block < block_to; ++block) {
        auto x_block = xx.block_view(block);
        auto d_block = delta.block_view(block);
        x_block += d_block;
      }

      omega_prev = omega_max;
    }

    // // Final diagnostic output
    // auto resid = xx;
    // xx.apply_to_mat(A, resid);
    // resid -= bbefore;
    // resid *= -1.0;

    // double final_omega_max = 0.0;
    // for (std::size_t block = block_from; block < block_to; ++block) {
    //   auto r_block = resid.block_view(block);
    //   auto x_block = xx.block_view(block);
    //   auto b_block = bbefore.block_view(block);
    //   double r_norm = r_block.two_norm();
    //   double x_norm = x_block.two_norm();
    //   double b_norm = b_block.two_norm();
    //   double omega_block = r_norm / (A_norm * x_norm + b_norm);
    //   final_omega_max = std::max(final_omega_max, omega_block);
    // }
    // logger::debug("Final backward error: {}", final_omega_max);
  }

  template <class Multivec1, class Multivec2>
  void solve(Multivec1& xx, Multivec2& bb, std::size_t block_from, std::size_t block_to)
  {
    if (!xx.matches_blocks(bb)) DUNE_THROW(Dune::Exception, "Vectors are incompatible");

    constexpr auto block_size = Multivec1::blocksize;
    using VReal = std::experimental::fixed_size_simd<double, block_size>;

#define row_start(v, row) std::assume_aligned<64>(&((v).data_ptr[block * xx.rows_ * block_size + (row) * block_size]))

#pragma omp parallel for schedule(static)
#if 1 // Manually vectorised variant
    for (std::size_t block = block_from; block < block_to; ++block) {
      VReal vdata;
      VReal vtemp;

      // Step 1: Row permutation and scaling - store result in xx (which becomes temporary storage)
      auto addr_write = row_start(xx, 0);
      for (std::int64_t k = 0; k < n_row; ++k) {
        vdata.copy_from(row_start(bb, P[k]), std::experimental::vector_aligned);
        vdata *= Rseff[k];
        vdata.copy_to(addr_write, std::experimental::vector_aligned);
        addr_write += block_size;
      }

      // Step 2: Forward solve L*y = (scaled and permuted b)
      // Input from xx, output to bb
      auto addr_read = row_start(xx, 0);
      addr_write = row_start(bb, 0);
      for (std::int64_t i = 0; i < n_row; ++i) {
        vdata.copy_from(addr_read, std::experimental::vector_aligned);

        for (auto k = Lp[i]; k < Lp[i + 1] - 1; ++k) {
          const double coeff = Lx[k];
          vtemp.copy_from(row_start(bb, Lj[k]), std::experimental::vector_aligned);
          vdata = std::experimental::fma(-coeff, vtemp, vdata);
        }

        vdata.copy_to(addr_write, std::experimental::vector_aligned);

        addr_read += block_size;
        addr_write += block_size;
      }

      // Step 3: Back solve U and apply column permutation
      // Input from bb, output to xx (final result)
      addr_read = row_start(bb, n_row - 1);
      for (std::int64_t j = n_row - 1; j >= 0; --j) {
        vdata.copy_from(addr_read, std::experimental::vector_aligned);

        // Multiply by precomputed inverse diagonal
        vdata *= Udiag_inv[j];

        // Update all rows above (those already processed)
        for (auto k = Up[j]; k < Up[j + 1] - 1; ++k) {
          const double coeff = Ux[k];
          vtemp.copy_from(row_start(bb, Ui[k]), std::experimental::vector_aligned);
          vtemp = std::experimental::fma(-coeff, vdata, vtemp);
          vtemp.copy_to(row_start(bb, Ui[k]), std::experimental::vector_aligned);
        }

        // Store in permuted position in final result
        vdata.copy_to(row_start(xx, Q[j]), std::experimental::vector_aligned);

        addr_read -= block_size;
      }
    }
#else // Variant that relies on autovectorisation (usually generates nearly identical assemble, occasionally even better than the other branch)
    for (std::size_t block = block_from; block < block_to; ++block) {
      alignas(64) double tmp[block_size];

      // Step 1: Row permutation and scaling - store result in xx (which becomes temporary storage)
      double* addr_write = row_start(xx, 0);
      for (std::int64_t k = 0; k < n_row; ++k) {
        auto addr_read = row_start(bb, P[k]);
        for (std::size_t i = 0; i < block_size; ++i) addr_write[i] = std::assume_aligned<64>(Rseff.data())[k] * addr_read[i];
        addr_write += block_size;
      }

      // Step 2: Forward solve L*y = (scaled and permuted b)
      // Input from xx, output to bb
      addr_write = row_start(bb, 0);
      auto addr_read = row_start(xx, 0);
      for (std::int64_t i = 0; i < n_row; ++i) {
        std::copy(addr_read, addr_read + block_size, tmp);

        for (auto k = Lp[i]; k < Lp[i + 1] - 1; ++k) {
          const double coeff = Lx[k];
          auto bp = row_start(bb, Lj[k]);
          for (std::size_t j = 0; j < block_size; ++j) tmp[j] -= coeff * bp[j];
        }

        for (std::size_t j = 0; j < block_size; ++j) addr_write[j] = tmp[j];

        addr_read += block_size;
        addr_write += block_size;
      }

      // Step 3: Back solve U and apply column permutation
      // Input from bb, output to xx (final result)
      addr_read = row_start(bb, n_row - 1);
      for (std::int64_t j = n_row - 1; j >= 0; --j) {
        for (std::size_t i = 0; i < block_size; ++i) tmp[i] = addr_read[i] * Udiag_inv[j];

        // Update all rows above (those already processed)
        for (auto k = Up[j]; k < Up[j + 1] - 1; ++k) {
          const double coeff = Ux[k];
          auto bp = row_start(bb, Ui[k]);
          for (std::size_t i = 0; i < block_size; ++i) bp[i] -= coeff * tmp[i];
        }

        // Store in permuted position in final result
        auto px = row_start(xx, Q[j]);
        for (std::size_t i = 0; i < block_size; ++i) px[i] = tmp[i];

        addr_read -= block_size;
      }
    }
#endif
#undef row_start
  }

private:
  void init()
  {
    void* Numeric = solver.getFactorization();

    auto status = umfpack_dl_get_lunz(&lnz, &unz, &n_row, &n_col, &nz_udiag, Numeric);
    if (status != UMFPACK_OK) DUNE_THROW(Dune::Exception, "UMFPACK error in umfpack_dl_get_lunz (error code: " + std::to_string(status) + ")");

    // Since we use an aligned allocator for all arrays below, we need to ensure that
    // the allocated size is divisible by the alignment. We just allocate a little bit
    // more if it isn't (which is fine, because we never query the sizes of these arrays).
    const auto next_div_by_alignment = [](auto x) {
      if (x % 64 == 0) return x;
      return x + 64 - x % 64;
    };

    Lp.resize(next_div_by_alignment(n_row + 1));
    Lj.resize(next_div_by_alignment(lnz));
    Lx.resize(next_div_by_alignment(lnz));

    Up.resize(next_div_by_alignment(n_col + 1));
    Ui.resize(next_div_by_alignment(unz));
    Ux.resize(next_div_by_alignment(unz));

    P.resize(next_div_by_alignment(n_row));
    Q.resize(next_div_by_alignment(n_col));

    Rs.resize(next_div_by_alignment(n_row));

    status = umfpack_dl_get_numeric(Lp.data(), Lj.data(), Lx.data(), Up.data(), Ui.data(), Ux.data(), P.data(), Q.data(), nullptr, &do_recip, Rs.data(), Numeric);
    if (status != UMFPACK_OK) DUNE_THROW(Dune::Exception, "UMFPACK error in umfpack_dl_get_numeric (error code: " + std::to_string(status) + ")");

    // Precompute effective scaling factors to avoid divisions in the hot path
    Rseff.resize(next_div_by_alignment(n_row));
    if (do_recip > 0)
      for (std::int64_t i = 0; i < n_row; ++i) Rseff[i] = Rs[P[i]];
    else
      for (std::int64_t i = 0; i < n_row; ++i) Rseff[i] = 1.0 / Rs[P[i]];

    // Precompute inverse of U diagonal entries for the back solve
    Udiag_inv.resize(next_div_by_alignment(n_row));
    for (std::int64_t j = 0; j < n_row; ++j) Udiag_inv[j] = 1.0 / Ux[Up[j + 1] - 1];
  }

  std::int64_t lnz{};
  std::int64_t unz{};
  std::int64_t n_row{};
  std::int64_t n_col{};
  std::int64_t nz_udiag{};

  // The L matrix
  std::vector<std::int64_t, AlignedAllocator<std::int64_t>> Lp;
  std::vector<std::int64_t, AlignedAllocator<std::int64_t>> Lj;
  std::vector<double, AlignedAllocator<double>> Lx;

  // The U matrix
  std::vector<std::int64_t, AlignedAllocator<std::int64_t>> Up;
  std::vector<std::int64_t, AlignedAllocator<std::int64_t>> Ui;
  std::vector<double, AlignedAllocator<double>> Ux;

  // The row permutation vector
  std::vector<std::int64_t, AlignedAllocator<std::int64_t>> P;

  // The column permutation vector
  std::vector<std::int64_t, AlignedAllocator<std::int64_t>> Q;

  // Boolean argument that determines how the scale factors have to be applied
  std::int64_t do_recip;

  // The scale factors
  std::vector<double, AlignedAllocator<double>> Rs;
  // Precomputed effective scaling (Rs or 1/Rs depending on do_recip)
  std::vector<double, AlignedAllocator<double>> Rseff;
  // Precomputed inverse of U diagonal
  std::vector<double, AlignedAllocator<double>> Udiag_inv;

  Dune::UMFPack<Mat> solver;
  const Mat& A;
  double A_norm{};
  int max_refinement_iter;
};
