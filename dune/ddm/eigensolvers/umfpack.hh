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
  explicit UMFPackMultivecSolver(const Mat& A) : solver(A) { init(); }

  template <class Multivec1, class Multivec2>
  void solve(Multivec1& xx, Multivec2& bb)
  {
    if (!xx.matches(bb)) DUNE_THROW(Dune::Exception, "Vectors are incompatible");

    constexpr auto block_size = Multivec1::blocksize();
    using VReal = std::experimental::fixed_size_simd<double, block_size>;

#define row_start(v, row) std::assume_aligned<64>(&((v).data_ptr[block * xx.rows_ * block_size + (row) * block_size]))

#pragma omp parallel for schedule(static)
#if 1 // Manually vectorised variant
    for (std::size_t block = 0; block < xx.blocks(); ++block) {
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
    for (std::size_t block = 0; block < xx.blocks(); ++block) {
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

    Lp.resize(n_row + 1);
    Lj.resize(lnz);
    Lx.resize(lnz);

    Up.resize(n_col + 1);
    Ui.resize(unz);
    Ux.resize(unz);

    P.resize(n_row);
    Q.resize(n_col);

    Rs.resize(n_row);

    status = umfpack_dl_get_numeric(Lp.data(), Lj.data(), Lx.data(), Up.data(), Ui.data(), Ux.data(), P.data(), Q.data(), nullptr, &do_recip, Rs.data(), Numeric);
    if (status != UMFPACK_OK) DUNE_THROW(Dune::Exception, "UMFPACK error in umfpack_dl_get_numeric (error code: " + std::to_string(status) + ")");

    // Precompute effective scaling factors to avoid divisions in the hot path
    Rseff.resize(n_row);
    if (do_recip > 0)
      for (std::int64_t i = 0; i < n_row; ++i) Rseff[i] = Rs[P[i]];
    else
      for (std::int64_t i = 0; i < n_row; ++i) Rseff[i] = 1.0 / Rs[P[i]];

    // Precompute inverse of U diagonal entries for the back solve
    Udiag_inv.resize(n_row);
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
};
