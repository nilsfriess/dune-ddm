#pragma once

#include "arnoldi.hh"
#include "concepts.hh"
#include "dune/ddm/eigensolvers/orthogonalisation.hh"
#include "dune/ddm/logger.hh"
#include "eigensolver_params.hh"
#include "lapacke.hh"

#include <dune/istl/bvector.hh>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

template <Eigenproblem EVP>
class BlockKrylovSchur {
  constexpr static std::size_t blocksize = EVP::blocksize;
  using Real = typename EVP::Real;
  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;
  using InnerProduct = typename EVP::InnerProduct;
  using Ortho = orthogonalisation::BlockOrthogonalisation<InnerProduct>;

public:
  BlockKrylovSchur(std::shared_ptr<EVP> evp, const EigensolverParams& params)
      : orth(std::make_shared<Ortho>(BetweenBlocks::ClassicalGramSchmidt, WithinBlocks::CholQR2, evp->get_inner_product()))
      , evp(std::move(evp))
      , lanczos(this->evp, orth, params)
      , nev(params.nev)
      , ncv(params.ncv)
      , tolerance(params.tolerance)
      , ritz_values(ncv, 0)
      , res_norms(nev, 0)
      , T(ncv * ncv, 0)
      , X(ncv * ncv, 0)

  {
    // Use CGS with a small number of reorthogonalisation passes (twice-is-enough)
    orth->set_reorthogonalization(true, 10);
    if (nev % blocksize != 0) throw std::invalid_argument("Number of requested eigenvalues must be multiple of blocksize");
    if (ncv % blocksize != 0) throw std::invalid_argument("Number of basis vectors must be multiple of blocksize");
    if (ncv <= nev) throw std::invalid_argument("Number of basis vectors must be larger than number of requested eigenvalues");
  }

  void apply()
  {
    auto& Tmat = lanczos.get_T_matrix();
    auto& Q = lanczos.get_basis();
    auto W = Q;  // work vector
    auto Z_ = W; // TODO: We only need one block
    auto Z = Z_.block_view(0);

    // We start by creating a Lanczos decomposition of the maximum allowed size
    lanczos.extend(0, ncv / blocksize);

    const std::size_t max_restarts = 100;
    while (its < max_restarts) {
      its++;

      // Transform the decomposition into Krylov-Schur form; in our case (real symmetric) this means diagonalising T
      solve_small();

      // Update number of converged blocks
      update_nconv();
      debug_print();

      // After the restart, we need to keep the residual block (it becomes the new residual block).
      // Therefore, we save it here, before computing the Ritz vectors.
      auto Qlast = Q.block_view(Q.blocks() - 1);
      Z = Qlast;

      // Compute the Ritz vectors
      // TODO: Here we multiply by the whole ncv x ncv matrix X. This is not necessary, we only keep
      //       the first nev vectors anyways, so we only need to compute Q * X[:, 1:nev].
      typename EVP::BlockMultiVec::BlockMatrix X_mat(Q.blocks() - 1);
      X_mat.from_flat_column_major(X);
      Q.mult(X_mat, W, nev / blocksize); // Compute Ritz vectors that we will keep
      std::swap(Q, W);

      if (nconv * blocksize >= nev) {
        logger::info_all("Converged target {} eigenpairs after {} restarts. Stopping.", nev, its);
        break;
      }

      // Not enough Ritz vectors have converged, so we prepare for restart.

      // Overwrite the first nev rows/columns of Tmat with a diagonal matrix containing the Ritz values
      for (std::size_t i = 0; i < nev / blocksize; ++i)
        for (std::size_t j = 0; j < nev / blocksize; ++j) Tmat.block_view(i, j).set_zero();

      for (std::size_t i = 0; i < nev / blocksize; ++i) {
        auto Tii = Tmat.block_view(i, i);
        for (std::size_t j = 0; j < blocksize; ++j) {
          const std::size_t idx = i * blocksize + j;
          Tii(j, j) = ritz_values[idx];
        }
      }

      // Just to be sure, clear the remaining blocks
      for (std::size_t i = nev / blocksize; i < Tmat.block_rows(); ++i) {
        Tmat.block_view(i, i).set_zero();
        if (i < Tmat.block_rows() - 1) {
          Tmat.block_view(i, i + 1).set_zero();
          Tmat.block_view(i + 1, i).set_zero();
        }
      }

      // Compute beta * E_m^T * X and put the leading blocksize x nev part into
      // the correct location of Tmat
      std::vector<Real> T_res_block(blocksize * ncv); // matrix of size blocksize x ncv stored in column-major
      auto beta = lanczos.get_beta();
      for (std::size_t i = 0; i < blocksize; ++i) {
        for (std::size_t j = 0; j < ncv; ++j) {
          Real sum = 0;
          for (std::size_t p = 0; p < blocksize; ++p) sum += beta(i, p) * X[(ncv - blocksize + p) + j * ncv];
          T_res_block[i + j * blocksize] = sum;
        }
      }

      for (std::size_t bcol = 0; bcol < nev / blocksize; ++bcol) {
        auto Tres = Tmat.block_view(nev / blocksize, bcol);
        for (std::size_t i = 0; i < blocksize; ++i) {
          for (std::size_t j = 0; j < blocksize; ++j) {
            const std::size_t col_idx = bcol * blocksize + j;
            Tres(i, j) = T_res_block[i + col_idx * blocksize];
          }
        }

        // Put the transposed block into (bcol, nconv) as well
        auto Tres_T = Tmat.block_view(bcol, nev / blocksize);
        Tres_T = Tres;
        Tres_T.transpose();
      }

      // Now the Lanczos relation does not hold anymore. We carry out one (quasi) Lanczos step to
      // create a proper Lanczos decomposition. After that, we can use the "normal" Lanczos method.
      auto k = nev / blocksize;
      auto Qi = Q.block_view(k);
      Qi = Z;                 // Restore the "residual" block that we saved earlier
      evp->apply_block(k, Q); // Compute Q_{k + 1} = A * Q_{k}

      // Compute the next diagonal block
      auto Qi1 = Q.block_view(k + 1);
      auto Tii = Tmat.block_view(k, k);
      evp->get_inner_product()->dot(Qi, Qi1, Tii);

      // Compute Q_{k + 1} -= Q_{k} * T(k, k)
      Qi.mult(Tii, Z);
      Qi1 -= Z;

      // TODO: This differs from the original paper, there they modify the vector differently.
      //       The original version is more efficient, so we should use that.
      // Orthogonalise Q_{k+1} against all previous vectors
      orth->orthonormalise_block_against_previous(Q, k + 1, nullptr, &beta);
      if (k + 1 < Tmat.block_rows()) {
        auto Ti1_i = Tmat.block_view(k + 1, k);
        Ti1_i = beta;

        auto Ti_i1 = Tmat.block_view(k, k + 1);
        Ti_i1 = beta;
        Ti_i1.transpose();
      }

      lanczos.extend(k + 1, ncv / blocksize);
    }
    if (its >= max_restarts) {
      if (nconv * blocksize >= nev) logger::info_all("Converged target {} eigenpairs after {} restarts. Stopping.", nev, its);
      else logger::warn_all("Maximum number of restarts reached ({}). Converged {} out of {} requested eigenpairs.", its, nconv * blocksize, nev);
    }
  }

  std::vector<Vec> extract_eigenvectors()
  {
    // Extract the first nev columns of Q (which contains the Ritz vectors after convergence)
    std::vector<Vec> eigenvectors(nev, Vec(evp->mat_size()));

    auto& Q = lanczos.get_basis();
    for (std::size_t k = 0; k < nev; ++k) {
      auto current_block = Q.block_view(k / blocksize);
      auto col = k % blocksize;
      for (std::size_t i = 0; i < evp->mat_size(); ++i) eigenvectors[k][i] = current_block(i, col);
    }

    // Log the computed eigenvalues
    std::vector<Real> transformed_eigenvalues(nev);
    for (std::size_t i = 0; i < nev; ++i) transformed_eigenvalues[i] = evp->transform_eigenvalue(ritz_values[i]);

    if (logger::get_level() <= logger::Level::trace)
      for (std::size_t i = 0; i < nev; ++i) logger::trace_all("Eigenvalue {}: {}", i, transformed_eigenvalues[i]);
    else logger::debug_all("Computed {} eigenvalues: smallest {}, largest {}", nev, transformed_eigenvalues[0], transformed_eigenvalues[nev - 1]);

    return eigenvectors;
  }

private:
  void update_nconv()
  {
    // Compute residuals and determine number of converged eigenpairs
    std::vector<Real> last_X(blocksize);
    std::vector<Real> tmp(blocksize);
    auto b = lanczos.get_beta();
    for (std::size_t i = nconv * blocksize; i < nev; ++i) { // Skip already converged Ritz values
      for (std::size_t j = 0; j < blocksize; ++j) last_X[j] = X[(ncv - blocksize + j) + i * ncv];

      b.mult(last_X, tmp);
      double sum_sq = std::accumulate(tmp.begin(), tmp.end(), 0.0, [](double acc, double x) { return acc + x * x; });
      auto res = std::sqrt(sum_sq);
      res_norms[i] = res;
    }

    const Real eps23 = pow(std::numeric_limits<Real>::epsilon(), Real(2) / 3);

    // Update nconv (in blocks)
    std::size_t new_nconv = 0;
    const std::size_t max_blocks = nev / blocksize;
    for (std::size_t bidx = nconv; bidx < max_blocks; ++bidx) {
      bool block_converged = true;
      for (std::size_t j = 0; j < blocksize; ++j) {
        const std::size_t idx = bidx * blocksize + j;

        // Compute specific threshold for this Ritz value
        auto thresh = std::max(tolerance * std::abs(ritz_values[idx]), eps23);

        if (res_norms[idx] > tolerance) {
          block_converged = false;
          break;
        }
      }
      if (block_converged) new_nconv++;
      else break;
    }
    logger::info("Computed residuals, converged blocks before {}, converged blocks now {}", nconv, nconv + new_nconv);
    nconv += new_nconv;
  }

  void solve_small()
  {
    T = lanczos.get_T_matrix().to_flat_column_major();

    // Diagonalise the T matrix (in general: transform it to Schur form, but in the symmetric case, this means diagonalising it)
    // TODO: We only need to consider the part of T that corresponds to unconverged Ritz values
    std::copy(T.begin(), T.end(), X.begin()); // Copy to X because syev solves in-place, and we want the result in X
    int info = lapacke::syevd(LAPACK_COL_MAJOR, 'V', 'U', (int)ncv, X.data(), (int)ncv, ritz_values.data());
    if (info != 0) throw std::runtime_error("FAILED: LAPACK syev failed with info = " + std::to_string(info));

    // Sort eigenvalues and eigenvectors
    std::vector<unsigned int> indices(ncv);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](const auto& i, const auto& j) { return evp->compare_untransformed(ritz_values[i], ritz_values[j]); });

    // Apply sorting permutation
    std::vector<Real> sorted_ritz_values(ritz_values.size());
    std::vector<Real> sorted_X(X.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
      sorted_ritz_values[i] = ritz_values[indices[i]];
      // Copy column indices[i] to column i (LAPACK_COL_MAJOR storage)
      for (std::size_t j = 0; j < ncv; ++j) sorted_X[j + i * ncv] = X[j + indices[i] * ncv];
    }
    ritz_values = std::move(sorted_ritz_values);
    X = std::move(sorted_X);
  }

  void debug_print()
  {
    // Nicely formatted table for Ritz values, residuals, thresholds, and convergence
    // Column widths and numeric formatting chosen to be compact and readable
    const int idx_w = 4;
    const int val_w = 14;
    const int res_w = 14;
    const int thr_w = 14;
    const int conv_w = 10;

    std::cout << std::left << std::setw(idx_w) << "#" << std::right << std::setw(val_w) << "Ritz value" << std::setw(res_w) << "Residual" << std::setw(thr_w) << "Threshold" << std::setw(conv_w)
              << "Converged" << std::endl;

    for (std::size_t i = 0; i < nev; i++) {
      // Choose notation: use fixed for moderately sized numbers, scientific otherwise
      const Real v = ritz_values[i];
      const Real r = res_norms[i];
      const Real t = tolerance;

      auto print_num = [&](Real x, int width) {
        // small/large magnitudes -> scientific with 6 sig figs, otherwise fixed with 6 decimals
        const Real ax = std::abs(x);
        if (ax != Real(0) && (ax < Real(1e-4) || ax >= Real(1e6))) std::cout << std::scientific << std::setprecision(6) << std::setw(width) << x;
        else std::cout << std::fixed << std::setprecision(6) << std::setw(width) << x;
        // restore default float format to avoid carrying formatting to next outputs
        std::cout << std::defaultfloat;
      };

      std::cout << std::left << std::setw(idx_w) << i;
      print_num(v, val_w);
      print_num(r, res_w);
      print_num(t, thr_w);
      std::cout << std::right << std::setw(conv_w) << ((r < t) ? "Yes" : "No") << std::endl;
    }
  }

  std::shared_ptr<Ortho> orth;
  std::shared_ptr<EVP> evp;
  BlockLanczos<EVP> lanczos;

  // Eigensolver parameters
  std::size_t nev;
  std::size_t ncv;
  double tolerance;

  std::size_t nconv{0}; // Number of converged blocks (NOT converged eigenpairs)
  std::size_t its{0};

  std::vector<Real> ritz_values;
  std::vector<Real> res_norms;

  std::vector<Real> T; // The flat T matrix
  std::vector<Real> X; // This holds the eigenvectors of the reduced problem
};

template <Eigenproblem EVP>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> block_krylov_schur(std::shared_ptr<EVP> evp, const EigensolverParams& params)
{
  BlockKrylovSchur solver(evp, params);
  solver.apply();
  return solver.extract_eigenvectors();
}
