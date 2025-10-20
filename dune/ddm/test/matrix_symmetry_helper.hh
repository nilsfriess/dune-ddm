#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace matrix_symmetry_helper {

/**
 * @brief Check if a matrix is symmetric within a given tolerance
 *
 * @param matrix The matrix to check for symmetry
 * @param name Optional name for reporting purposes
 * @param tolerance Tolerance for symmetry check (default: 1e-14)
 * @return true if matrix is symmetric within tolerance, false otherwise
 */
template <class Mat>
bool check_matrix_symmetry(const Mat& matrix, const std::string& name = "", double tolerance = 1e-14, bool verbose = false)
{
  if (verbose) std::cout << "=== Checking symmetry of matrix" << (name.empty() ? "" : " " + name) << " ===\n";

  if (matrix.N() != matrix.M()) {
    if (verbose) std::cout << "FAILED: Matrix" << (name.empty() ? "" : " " + name) << " is not square (" << matrix.N() << " x " << matrix.M() << ")\n";
    return false;
  }

  double max_asymmetry = 0.0;
  std::size_t asymmetric_entries = 0;
  std::size_t total_entries = 0;

  // Check symmetry by comparing A(i,j) with A(j,i)
  for (auto row_it = matrix.begin(); row_it != matrix.end(); ++row_it) {
    std::size_t i = row_it.index();

    for (auto col_it = row_it->begin(); col_it != row_it->end(); ++col_it) {
      std::size_t j = col_it.index();
      double aij = *col_it;
      total_entries++;

      // Find corresponding A(j,i) entry
      double aji = 0.0;
      bool found_aji = false;

      if (j < matrix.N()) {
        auto ji_row = matrix.begin() + j;
        for (auto ji_col = ji_row->begin(); ji_col != ji_row->end(); ++ji_col) {
          if (ji_col.index() == i) {
            aji = *ji_col;
            found_aji = true;
            break;
          }
        }
      }

      // If A(j,i) not found, it's implicitly zero in sparse matrix
      if (!found_aji && std::abs(aij) > tolerance) {
        asymmetric_entries++;
        max_asymmetry = std::max(max_asymmetry, std::abs(aij));
      }
      else if (found_aji) {
        double asymmetry = std::abs(aij - aji);
        if (asymmetry > tolerance) {
          asymmetric_entries++;
          max_asymmetry = std::max(max_asymmetry, asymmetry);
        }
      }
    }
  }

  if (verbose) std::cout << "Matrix" << (name.empty() ? "" : " " + name) << " statistics:\n";
  if (verbose) std::cout << "  Size: " << matrix.N() << " x " << matrix.M() << "\n";
  if (verbose) std::cout << "  Total non-zero entries: " << total_entries << "\n";
  if (verbose) std::cout << "  Asymmetric entries (|A(i,j) - A(j,i)| > " << tolerance << "): " << asymmetric_entries << "\n";
  if (verbose) std::cout << "  Maximum asymmetry: " << max_asymmetry << "\n";

  bool is_symmetric = (asymmetric_entries == 0);
  if (is_symmetric) {
    if (verbose) std::cout << "PASSED: Matrix" << (name.empty() ? "" : " " + name) << " is symmetric (within tolerance " << tolerance << ")\n";
  }
  else {
    if (verbose) std::cout << "FAILED: Matrix" << (name.empty() ? "" : " " + name) << " is NOT symmetric\n";
    if (total_entries > 0)
      if (verbose) std::cout << "  Percentage of asymmetric entries: " << (100.0 * static_cast<double>(asymmetric_entries) / static_cast<double>(total_entries)) << "%\n";
  }

  return is_symmetric;
}

/**
 * @brief Symmetrize a matrix using the formula (A + A^T)/2
 *
 * @param matrix The matrix to symmetrize
 * @param name Optional name for reporting purposes
 * @return Symmetrized matrix
 */
template <class Mat>
Mat symmetrize_matrix(const Mat& matrix, const std::string& name = "", bool verbose = false)
{
  if (verbose) std::cout << "=== Symmetrizing matrix" << (name.empty() ? "" : " " + name) << " using (A + A^T)/2 ===\n";

  if (matrix.N() != matrix.M()) {
    if (verbose) std::cout << "ERROR: Cannot symmetrize non-square matrix (" << matrix.N() << " x " << matrix.M() << ")\n";
    return matrix; // Return original matrix if not square
  }

  const std::size_t n = matrix.N();

  // Create result matrix with same sparsity pattern as union of A and A^T
  Mat result(n, n, Mat::BuildMode::row_wise);

  // First pass: determine sparsity pattern (union of A and A^T patterns)
  std::vector<std::set<std::size_t>> pattern(n);

  // Add entries from A
  for (auto row_it = matrix.begin(); row_it != matrix.end(); ++row_it) {
    std::size_t i = row_it.index();
    for (auto col_it = row_it->begin(); col_it != row_it->end(); ++col_it) {
      std::size_t j = col_it.index();
      pattern[i].insert(j);
      pattern[j].insert(i); // Also add transpose entry
    }
  }

  // Create matrix structure
  for (auto it = result.createbegin(); it != result.createend(); ++it) {
    std::size_t i = it.index();
    for (std::size_t j : pattern[i]) it.insert(j);
  }

  // Second pass: fill values with (A + A^T)/2
  for (auto row_it = result.begin(); row_it != result.end(); ++row_it) {
    std::size_t i = row_it.index();
    for (auto col_it = row_it->begin(); col_it != row_it->end(); ++col_it) {
      std::size_t j = col_it.index();

      // Get A(i,j)
      double aij = 0.0;
      auto orig_row = matrix.begin() + i;
      for (auto orig_col = orig_row->begin(); orig_col != orig_row->end(); ++orig_col) {
        if (orig_col.index() == j) {
          aij = *orig_col;
          break;
        }
      }

      // Get A(j,i)
      double aji = 0.0;
      if (j < matrix.N()) {
        auto orig_row_ji = matrix.begin() + j;
        for (auto orig_col_ji = orig_row_ji->begin(); orig_col_ji != orig_row_ji->end(); ++orig_col_ji) {
          if (orig_col_ji.index() == i) {
            aji = *orig_col_ji;
            break;
          }
        }
      }

      // Set result(i,j) = (A(i,j) + A(j,i))/2
      *col_it = (aij + aji) / 2.0;
    }
  }

  if (verbose) std::cout << "Matrix symmetrization completed. Original size: " << matrix.N() << "x" << matrix.M() << ", Result size: " << result.N() << "x" << result.M() << "\n";

  return result;
}

/**
 * @brief Check and optionally symmetrize matrices A and B
 *
 * @param A Matrix A (will be modified if not symmetric)
 * @param B Matrix B (will be modified if not symmetric)
 * @param tolerance Tolerance for symmetry check
 * @param force_symmetrize If true, always symmetrize even if matrices appear symmetric
 * @return true if both matrices are/were made symmetric successfully
 */
template <class Mat>
bool ensure_matrix_symmetry(std::shared_ptr<Mat>& A, std::shared_ptr<Mat>& B, double tolerance = 1e-12, bool force_symmetrize = false, bool verbose = false)
{
  if (verbose) std::cout << "\n=== Matrix Symmetry Analysis ===\n";

  // Check if the loaded matrices are symmetric
  bool A_symmetric = check_matrix_symmetry(*A, "A", tolerance, verbose);
  bool B_symmetric = check_matrix_symmetry(*B, "B", tolerance, verbose);

  bool success = true;

  if (!A_symmetric || force_symmetrize) {
    if (!A_symmetric) {
      if (verbose) std::cout << "WARNING: Matrix A is not symmetric! Symmetrizing it...\n";
    }
    else {
      if (verbose) std::cout << "INFO: Force symmetrizing matrix A...\n";
    }
    *A = symmetrize_matrix(*A, "A", verbose);
    // Verify the symmetrized matrix
    bool A_sym_check = check_matrix_symmetry(*A, "A (symmetrized)", 1e-14, verbose);
    if (!A_sym_check) {
      if (verbose) std::cout << "ERROR: Symmetrization of A failed!\n";
      success = false;
    }
  }

  if (!B_symmetric || force_symmetrize) {
    if (!B_symmetric) {
      if (verbose) std::cout << "WARNING: Matrix B is not symmetric! Symmetrizing it...\n";
    }
    else {
      if (verbose) std::cout << "INFO: Force symmetrizing matrix B...\n";
    }
    *B = symmetrize_matrix(*B, "B", verbose);
    // Verify the symmetrized matrix
    bool B_sym_check = check_matrix_symmetry(*B, "B (symmetrized)", 1e-14, verbose);
    if (!B_sym_check) {
      if (verbose) std::cout << "ERROR: Symmetrization of B failed!\n";
      success = false;
    }
  }

  if (success) {
    if (verbose) std::cout << "SUCCESS: Both matrices A and B are now symmetric.\n";
  }

  return success;
}

} // namespace matrix_symmetry_helper
