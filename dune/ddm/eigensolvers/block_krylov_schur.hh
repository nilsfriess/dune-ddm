#pragma once

#include "arnoldi.hh"
#include "concepts.hh"
#include "dune/ddm/eigensolvers/orthogonalisation.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/logger.hh"
#include "eigensolver_params.hh"
#include "lapacke.hh"

#include <dune/istl/bvector.hh>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

template <Eigenproblem EVP, class Callback = NoCallback>
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
      , T(ncv * ncv, 0)
      , X(ncv * ncv, 0)

  {
    if (nev % blocksize != 0) throw std::invalid_argument("Number of requested eigenvalues must be multiple of blocksize");
    if (ncv % blocksize != 0) throw std::invalid_argument("Number of basis vectors must be multiple of blocksize");
    if (ncv <= nev) throw std::invalid_argument("Number of basis vectors must be larger than number of requested eigenvalues");
  }

  BlockKrylovSchur(std::shared_ptr<EVP> evp, const EigensolverParams& params, Callback&& callback)
      : orth(std::make_shared<Ortho>(BetweenBlocks::ClassicalGramSchmidt, WithinBlocks::CholQR2, evp->get_inner_product()))
      , evp(std::move(evp))
      , lanczos(this->evp, orth, params, std::move(callback))
      , nev(params.nev)
      , ncv(params.ncv)
      , tolerance(params.tolerance)
      , ritz_values(ncv, 0)
      , T(ncv * ncv, 0)
      , X(ncv * ncv, 0)

  {
    if (nev % blocksize != 0) throw std::invalid_argument("Number of requested eigenvalues must be multiple of blocksize");
    if (ncv % blocksize != 0) throw std::invalid_argument("Number of basis vectors must be multiple of blocksize");
    if (ncv <= nev) throw std::invalid_argument("Number of basis vectors must be larger than number of requested eigenvalues");
  }

  void apply()
  {
    auto& Tmat = lanczos.get_T_matrix();
    auto& Q = lanczos.get_basis();
    auto W = Q; // work vector

    while (true) {
      its++;

      // Extend the current Krylov decomposition block-by-block to the wanted size
      lanczos.extend(nconv + 1, ncv / blocksize);
      T = Tmat.to_flat_column_major();

      // Diagonalise the T matrix (in general: transform it to Schur form, but in the symmetric case, this means diagonalising it)
      std::copy(T.begin(), T.end(), X.begin()); // Copy to X because syev solves in-place, and we want the result in X
      int info = lapacke::syev(LAPACK_COL_MAJOR, 'V', 'U', (int)ncv, X.data(), (int)ncv, ritz_values.data());
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

      // if (logger::get_level() <= logger::Level::trace) {
      //   logger::info_all("Iteration {}, transformed Ritz values are", its);
      //   for (std::size_t i = 0; i < ncv; ++i) logger::info_all("  {}: {}", i, evp->transform_eigenvalue(ritz_values[i]));
      // }

      // Compute Ritz vectors. For that, first turn X into a SquareDenseBlockMatrix so that
      // we can use the optimised multiplication methods of the BlockMultiVector
      typename EVP::BlockMultiVec::BlockMatrix X_mat(Q.blocks() - 1);
      X_mat.from_flat_column_major(X);

      Q.mult(X_mat, W, Q.blocks() - 2); // Skip the last block of Q
      std::swap(Q, W);

      std::vector<Real> last_X(blocksize);
      std::vector<Real> tmp(blocksize);
      auto b = lanczos.get_beta();
      for (std::size_t i = 0; i < ncv; ++i) {
        for (std::size_t j = 0; j < blocksize; ++j) last_X[j] = X[(ncv - 1 + i * ncv) - blocksize + j + 1];

        b.mult(last_X, tmp);
        double sum_sq = std::accumulate(tmp.begin(), tmp.end(), 0.0, [](double acc, double x) { return acc + x * x; });
        auto res = std::sqrt(sum_sq);

        if (i < nev) std::cout << "Ritz value " << std::setw(2) << i << ": " << std::setprecision(10) << std::setw(12) << evp->transform_eigenvalue(ritz_values[i]) << ", Residual: " << res << "\n";
      }

      // std::cout << "Lanczos beta: ";
      // lanczos.get_beta().print();

      return;

      TODO("Lanczos");

      // // First we extend the current Krylov decomposition
      // std::vector<Real> final_beta_data(blocksize * blocksize);
      // typename EVP::BlockMultiVec::DenseBlockMatrixBlockView final_beta(final_beta_data.data());
      // lanczos_extend_decomposition(*evp, Q, nconv + 1, Q.blocks() - 1, orth, alphas, betas, &final_beta);

      // Next, update the Rayleigh quotient matrix by putting the newly computed alphas and betas into the matrix,
      // leaving the old alphas and betas, as well as the entries corresponding to the previous residual as-is.
      // build_tridiagonal_matrix(alphas, betas, T, nconv);

      // if (false) {
      //   // Debug: Print T matrix with block structure visualization
      //   std::cout << "############### T matrix #########################\n";

      //   // Print column headers to show block boundaries
      //   std::cout << "    ";
      //   for (std::size_t j = 0; j < ncv; ++j) {
      //     if (j % blocksize == 0 && j > 0) std::cout << "| ";
      //     std::cout << std::setw(10) << j << " ";
      //   }
      //   std::cout << "\n";

      //   // Print horizontal separator
      //   std::cout << "    ";
      //   for (std::size_t j = 0; j < ncv; ++j) {
      //     if (j % blocksize == 0 && j > 0) std::cout << "+-";
      //     std::cout << "-----------";
      //   }
      //   std::cout << "\n";

      //   for (std::size_t i = 0; i < ncv; ++i) {
      //     // Print row separators between blocks
      //     if (i % blocksize == 0 && i > 0) {
      //       std::cout << "    ";
      //       for (std::size_t j = 0; j < ncv; ++j) {
      //         if (j % blocksize == 0 && j > 0) std::cout << "+-";
      //         std::cout << "-----------";
      //       }
      //       std::cout << "\n";
      //     }

      //     // Print row index
      //     std::cout << std::setw(3) << i << " ";

      //     for (std::size_t j = 0; j < ncv; ++j) {
      //       // Add vertical separator between blocks
      //       if (j % blocksize == 0 && j > 0) std::cout << "| ";

      //       auto entry = T[j * ncv + i];
      //       if (entry == 0.0) std::cout << std::setw(10) << "*";
      //       else std::cout << std::setw(10) << std::fixed << std::setprecision(3) << entry;
      //       std::cout << " ";
      //     }
      //     std::cout << "\n";
      //   }
      //   std::cout << "##################################################\n";
      // }

      // // Compute residual norms for convergence checking
      // std::vector<Real> res_norms(nev);
      // for (std::size_t i = 0; i < nev; ++i) {
      //   // Extract the last blocksize components of the i-th eigenvector
      //   // LAPACK stores eigenvectors in columns (LAPACK_COL_MAJOR)
      //   // So the i-th eigenvector is column i: X[row + i * ncv] for row = 0...ncv-1
      //   std::vector<Real> X_last_components(blocksize);
      //   for (std::size_t j = 0; j < blocksize; ++j) {
      //     // Get the (ncv - blocksize + j)-th component of the i-th eigenvector
      //     std::size_t row = (ncv - blocksize) + j;
      //     X_last_components[j] = X[row + i * ncv];
      //   }

      //   // Compute residual: r = final_beta * X_last_components
      //   std::vector<Real> residual_vector(blocksize, 0.0);
      //   for (std::size_t row = 0; row < blocksize; ++row)
      //     for (std::size_t col = 0; col < blocksize; ++col) residual_vector[row] += final_beta.data()[row * blocksize + col] * X_last_components[col];

      //   // Compute 2-norm of the residual
      //   res_norms[i] = 0;
      //   for (const auto& val : residual_vector) res_norms[i] += val * val;
      //   res_norms[i] = std::sqrt(res_norms[i]);

      //   logger::info("Ritz value {}: {}, residual norm: {}", i, evp->transform_eigenvalue(ritz_values[i]), res_norms[i]);
      // }

      // // Compute Ritz vectors. For that, first turn X into a SquareDenseBlockMatrix so that
      // // we can use the optimised multiplication methods of the BlockMultiVector
      // typename EVP::BlockMultiVec::BlockMatrix X_mat(Q.blocks() - 1);
      // X_mat.from_flat_column_major(X);
      // Q.mult(X_mat, W, Q.blocks() - 2); // Skip the last block of Q
      // std::swap(Q, W);

      // // Check convergence and update nconv
      // std::size_t new_nconv = 0;
      // for (std::size_t i = 0; i < res_norms.size(); i += blocksize) {
      //   bool block_converged = true;
      //   for (std::size_t j = 0; j < blocksize && i + j < res_norms.size(); ++j) {
      //     if (res_norms[i + j] > tolerance) {
      //       block_converged = false;
      //       break;
      //     }
      //   }
      //   if (block_converged) new_nconv++;
      //   else break;
      // }

      // nconv = new_nconv;
      // if (true or nconv * blocksize >= nev) {
      //   logger::info("Converged! Found {} eigenvalues", nconv * blocksize);
      //   break;
      // }

      TODO("Krylov-Schur");
    }
  }

  std::vector<Vec> extract_eigenvectors()
  {
    // orth.orthonormalise(Q); // TODO: Is this necessary?

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
      for (std::size_t i = 0; i < nev; ++i) logger::trace("Eigenvalue {}: {}", i, transformed_eigenvalues[i]);
    else logger::debug("Computed {} eigenvalues: smallest {}, largest {}", nev, transformed_eigenvalues[0], transformed_eigenvalues[nev - 1]);

    return eigenvectors;
  }

private:
  std::shared_ptr<Ortho> orth;
  std::shared_ptr<EVP> evp;
  BlockLanczos<EVP, Callback> lanczos;

  // Eigensolver parameters
  std::size_t nev;
  std::size_t ncv;
  double tolerance;

  std::size_t nconv{0}; // Number of converged blocks (NOT converged eigenpairs)
  std::size_t its{0};

  std::vector<Real> ritz_values;

  // // TODO: Create a block triangular matrix type to hold these coefficients
  // std::vector<std::array<typename EVP::Real, EVP::blocksize * EVP::blocksize>> alpha_data;
  // std::vector<std::array<typename EVP::Real, EVP::blocksize * EVP::blocksize>> beta_data;
  // std::vector<typename EVP::BlockMultiVec::BlockMatrixBlockView> alphas;
  // std::vector<typename EVP::BlockMultiVec::BlockMatrixBlockView> betas;

  std::vector<Real> T; // The flat T matrix
  std::vector<Real> X; // This holds the eigenvectors of the reduced problem
};

template <Eigenproblem EVP, class Callback = NoCallback>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> block_krylov_schur(std::shared_ptr<EVP> evp, const EigensolverParams& params, Callback callback = Callback{})
{
  BlockKrylovSchur solver(evp, params, std::move(callback));
  solver.apply();
  return solver.extract_eigenvectors();
}
