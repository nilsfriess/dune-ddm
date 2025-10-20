#include "dune/ddm/eigensolvers/eigensolver_params.hh"
#undef NDEBUG

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#include <dune/common/parallel/mpihelper.hh>
#pragma GCC diagnostic pop

#include "matrix_symmetry_helper.hh"
#include "problem.hh"

#include <dune/ddm/eigensolvers/arnoldi.hh>
#include <dune/ddm/eigensolvers/blockmatrix.hh>
#include <dune/ddm/eigensolvers/blockmultivector.hh>
#include <dune/ddm/eigensolvers/inner_products.hh>
#include <dune/ddm/eigensolvers/lapacke.hh>
#include <dune/ddm/eigensolvers/orthogonalisation.hh>
#include <dune/ddm/eigensolvers/shift_invert_eigenproblem.hh>
#include <dune/ddm/eigensolvers/standard_eigenproblem.hh>

// // A helper function to run the Lanczos decomposition and perform basic checks.
// // It is parameterized by a function that sets up the initial block Q_0.
// bool run_lanczos_test(const std::string& test_name, const std::function<void(BMV&, Ortho&)>& setup_initial_block, bool check_invariance)
// {
//   std::cout << "=== " << test_name << " ===\n";

//   constexpr std::size_t rows = 256;
//   const std::size_t m = 20;

//   // Create a simple diagonal matrix with more reasonable eigenvalues
//   auto A = std::make_shared<Mat>(rows, rows, Mat::BuildMode::row_wise);
//   for (auto it = A->createbegin(); it != A->createend(); ++it) it.insert(it.index());
//   for (std::size_t i = 0; i < rows; ++i) (*A)[i][i] = static_cast<Real>(rows - i); // Eigenvalues from 256 down to 1

//   // Create the eigenproblem
//   EVP evp(A);

//   // Create block multivector
//   BMV Q(rows, m * blocksize);

//   // Setup orthogonalization
//   auto ortho_ptr = std::make_shared<Ortho>(ModifiedGramSchmidt, ShiftedCholQR3, evp.get_inner_product());

//   // Initialize Q_0 using the provided function
//   setup_initial_block(Q, *ortho_ptr);

//   // Build the Lanczos decomposition
//   std::vector<std::vector<Real>> alpha_data(m - 1);
//   std::vector<std::vector<Real>> beta_data(m - 1);
//   std::vector<DenseMatView> alpha_coeffs;
//   std::vector<DenseMatView> beta_coeffs;
//   for (std::size_t i = 0; i < m - 1; ++i) {
//     alpha_data[i].resize(blocksize * blocksize);
//     beta_data[i].resize(blocksize * blocksize);
//     alpha_coeffs.emplace_back(alpha_data[i].data());
//     beta_coeffs.emplace_back(beta_data[i].data());
//   }

//   std::vector<Real> final_beta_data(blocksize * blocksize);
//   DenseMatView final_beta(final_beta_data.data());
//   if (!lanczos_extend_decomposition(evp, Q, 0, Q.blocks() - 1, *ortho_ptr, alpha_coeffs, beta_coeffs, &final_beta)) {
//     // Breakdown is not a failure if we expect an invariant subspace.
//     if (!check_invariance) {
//       std::cout << "FAILED: Lanczos decomposition build failed unexpectedly.\n";
//       return false;
//     }
//   }

//   bool passed = true;

//   std::vector<Real> T_data((m - 1) * blocksize * (m - 1) * blocksize, 0); // Dense column-major matrix
//   build_tridiagonal_matrix(alpha_coeffs, beta_coeffs, T_data);

//   const std::size_t T_size = (m - 1) * blocksize;
//   const int T_size_int = static_cast<int>(T_size);

//   // Compute eigenvalues of T using LAPACK
//   std::vector<Real> eigenvalues(T_size);
//   std::vector<std::pair<Real, std::size_t>> eigen_pairs; // Declare outside the if block

//   int info = lapacke::syev(LAPACK_COL_MAJOR, 'V', 'U', T_size_int, T_data.data(), T_size_int, eigenvalues.data());

//   if (info != 0) {
//     std::cout << "FAILED: LAPACK syev failed with info = " << info << "\n";
//     passed = false;
//   }
//   else {
//     // Create pairs of (eigenvalue, eigenvector_index) for sorting
//     eigen_pairs.reserve(T_size);
//     for (std::size_t i = 0; i < T_size; ++i) eigen_pairs.emplace_back(eigenvalues[i], i);

//     // Sort by eigenvalue in descending order
//     std::sort(eigen_pairs.begin(), eigen_pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

//     // Extract sorted eigenvalues
//     for (std::size_t i = 0; i < T_size; ++i) eigenvalues[i] = eigen_pairs[i].first;

//     // Compare the largest eigenvalues to analytical values
//     std::cout << "Comparing computed Ritz values to analytical eigenvalues:\n";
//     std::cout << std::setw(20) << "Ritz" << std::setw(20) << "Analytical" << std::setw(20) << "Abs. Error" << std::setw(20) << "Rel. Error\n";
//     std::cout << "----------------------------------------------------------------------------------------------------\n";

//     for (std::size_t i = 0; i < std::min(T_size, std::size_t(10)); ++i) {
//       Real analytical_eigenvalue = static_cast<Real>(rows - i); // Largest eigenvalues: 256, 255, 254, ...
//       double abs_error = std::abs(eigenvalues[i] - analytical_eigenvalue);
//       double rel_error = abs_error / analytical_eigenvalue;
//       std::cout << std::setw(20) << eigenvalues[i] << std::setw(20) << analytical_eigenvalue << std::setw(20) << abs_error << std::setw(20) << rel_error << "\n";
//     }

//     std::cout << "Test with random vector completed.\n";
//   }

//   // Compute residual norm using properly ordered eigenvectors
//   if (!eigen_pairs.empty()) {
//     std::vector<Real> last_block_eigenvecs(blocksize * T_size);
//     for (std::size_t i = 0; i < T_size; ++i) {      // For each eigenvector (in sorted order)
//       std::size_t orig_idx = eigen_pairs[i].second; // Original LAPACK index
//       for (std::size_t j = 0; j < blocksize; ++j) { // Last blocksize components
//         std::size_t row_idx = T_size - blocksize + j;
//         last_block_eigenvecs[j * T_size + i] = T_data[orig_idx * T_size + row_idx];
//       }
//     }

//     // Multiply last block eigenvectors by final beta matrix
//     std::vector<Real> residual_vecs(blocksize * T_size, 0.0);
//     for (std::size_t i = 0; i < T_size; ++i) { // For each eigenvector
//       for (std::size_t row = 0; row < blocksize; ++row)
//         for (std::size_t col = 0; col < blocksize; ++col) residual_vecs[row * T_size + i] += final_beta_data[row * blocksize + col] * last_block_eigenvecs[col * T_size + i];
//     }

//     // Compute individual residual norms
//     std::cout << "Individual Ritz pair residual norms:\n";
//     for (std::size_t i = 0; i < std::min(T_size, std::size_t(10)); ++i) {
//       Real residual_norm = 0.0;
//       for (std::size_t j = 0; j < blocksize; ++j) {
//         Real val = residual_vecs[j * T_size + i];
//         residual_norm += val * val;
//       }
//       residual_norm = std::sqrt(residual_norm);
//       std::cout << "  Ritz pair " << i << ": ||residual|| = " << residual_norm << ", eigenvalue = " << eigenvalues[i] << "\n";
//     }

//     double total_residual_norm = 0;
//     for (const auto& val : residual_vecs) total_residual_norm += val * val;
//     total_residual_norm = std::sqrt(total_residual_norm);
//     std::cout << "Total residual norm (Frobenius norm): " << total_residual_norm << "\n";
//   }

//   if (passed) std::cout << "PASSED: " << test_name << "\n";
//   return passed;
// }

// /** @brief Test the Lanczos algorithm with a random initial subspace. */
// bool test_random_subspace()
// {
//   auto setup = [](BMV& Q, Ortho& ortho) {
//     // Initialize Q_0 with random data and orthonormalize it
//     std::mt19937 rng(42); // NOLINT: Deterministic seed for reproducible tests
//     std::normal_distribution<double> dist(0.0, 1.0);
//     auto Q0 = Q.block_view(0);
//     for (std::size_t r = 0; r < Q0.rows(); ++r)
//       for (std::size_t c = 0; c < Q0.cols(); ++c) Q0(r, c) = dist(rng);
//     ortho.orthonormalise_block_against_previous(Q, 0);
//   };
//   return run_lanczos_test("Testing Lanczos with a random subspace", setup, false);
// }

template <class EVP>
bool test_lanczos_decomposition(EVP& evp, std::size_t m)
{
  std::cout << "=== " << " Test if a Lanczos decomposition is formed correctly" << " ===\n";

  using InnerProduct = typename EVP::InnerProduct;
  using Ortho = orthogonalisation::BlockOrthogonalisation<InnerProduct>;
  using BMV = typename EVP::BlockMultiVec;
  using DenseMatView = typename BMV::DenseBlockMatrixBlockView;
  using Real = typename EVP::Real;
  constexpr std::size_t blocksize = EVP::blocksize;

  // Create the eigenproblem
  Ortho ortho(BetweenBlocks::ModifiedGramSchmidt, WithinBlocks::ModifiedGramSchmidt, evp.get_inner_product());
  BMV Q(evp.mat_size(), m * blocksize);
  Q.set_random();
  evp.apply(Q, Q); // Apply A to random Q to get a better starting guess
  ortho.orthonormalise(Q);

  // Build the Lanczos decomposition
  // TODO: Create a matrix type for this
  std::vector<std::vector<Real>> alpha_data(m - 1);
  std::vector<std::vector<Real>> beta_data(m - 1);
  std::vector<DenseMatView> alpha_coeffs;
  std::vector<DenseMatView> beta_coeffs;
  for (std::size_t i = 0; i < m - 1; ++i) {
    alpha_data[i].resize(blocksize * blocksize);
    beta_data[i].resize(blocksize * blocksize);
    alpha_coeffs.emplace_back(alpha_data[i].data());
    beta_coeffs.emplace_back(beta_data[i].data());
  }

  std::vector<Real> final_beta_data(blocksize * blocksize);
  DenseMatView final_beta(final_beta_data.data());
  if (!lanczos_extend_decomposition(evp, Q, 0, Q.blocks() - 1, ortho, alpha_coeffs, beta_coeffs, &final_beta)) {
    std::cout << "ERROR: Lanczos decomposition failed\n";
    return false;
  }

  // Build a "flat" block tridiagonal matrix
  std::vector<Real> T_data((m - 1) * blocksize * (m - 1) * blocksize, 0); // Dense column-major matrix
  build_tridiagonal_matrix(alpha_coeffs, beta_coeffs, T_data);

  using BlockMatrix = typename EVP::BlockMatrix;
  BlockMatrix T(Q.blocks() - 1);
  T.from_flat_column_major(T_data);

  BMV V(Q.rows(), Q.cols() - blocksize);

  // Copy all columns except the last into V
  for (std::size_t b = 0; b < Q.blocks() - 1; ++b) {
    auto Qb = Q.block_view(b);
    auto Vb = V.block_view(b);
    Vb = Qb;
  }
  auto W = V;
  auto Z = V;

  // Compute W = AV (=A Q[:, 0 : m-1])
  evp.apply(V, W);

  // Compute Z = VT
  V.mult(T, Z);

  // Compute W = W - V
  W -= Z;

  // Subtract the residual term
  auto Qlast = Q.block_view(Q.blocks() - 1);
  auto Vlast = V.block_view(0); // Just reuse some block of V
  auto Wlast = W.block_view(W.blocks() - 1);
  Qlast.mult(final_beta, Vlast);
  Wlast -= Vlast;

  bool passed = true;
  if (W.two_norm() < 1e-8) { std::cout << "PASSED: Lanczos decomposition is formed correctly, error is " << W.two_norm() << "\n"; }
  else {
    std::cout << "FAILED: Lanczos decomposition is not formed correctly, error is " << W.two_norm() << "\n";
    passed = false;
  }
  return passed;
}

namespace {

} // anonymous namespace

int main2(int argc, char* argv[])
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);

  using Real = double;
  constexpr std::size_t blocksize = 1;
  using Mat = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;

  {
    using EVP = StandardEigenproblem<Mat, blocksize>;

    Mat A(100, 100, Mat::BuildMode::row_wise);
    for (auto it = A.createbegin(); it != A.createend(); ++it) it.insert(it.index());
    for (std::size_t i = 0; i < 100; ++i) A[i][i] = static_cast<Real>(i + 1); // Eigenvalues from 1 to 100

    // Check if the diagonal matrix is symmetric (it should be)
    bool A_symmetric = matrix_symmetry_helper::check_matrix_symmetry(A, "A (diagonal)", 1e-14);
    if (!A_symmetric) {
      std::cout << "ERROR: Diagonal matrix is not symmetric! This should not happen.\n";
      return 1;
    }

    EVP evp(std::make_shared<Mat>(A));
    test_lanczos_decomposition(evp, 8);
  }

  {
    std::string Afile = "A_cd.mtx";
    std::string Bfile = "B_cd.mtx";

    auto A = std::make_shared<Mat>();
    auto B = std::make_shared<Mat>();
    Dune::loadMatrixMarket(*A, Afile);
    Dune::loadMatrixMarket(*B, Bfile);

    // Check and symmetrize matrices if needed
    matrix_symmetry_helper::ensure_matrix_symmetry(A, B, 1e-12, false);

    using EVP = ShiftInvertEigenproblem<Mat, blocksize>;
    ShiftInvertEigenproblem<Mat, blocksize> evp(*A, B, 1e-4);
    test_lanczos_decomposition(evp, 3);
  }

  return 0;
}

int main(int argc, char* argv[])
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);

  using Real = double;
  constexpr std::size_t blocksize = 4;
  using Mat = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;

  {
    std::string Afile = "A_geneo_1.mtx";
    std::string Bfile = "B_geneo_1.mtx";

    auto A = std::make_shared<Mat>();
    auto B = std::make_shared<Mat>();
    Dune::loadMatrixMarket(*A, Afile);
    Dune::loadMatrixMarket(*B, Bfile);

    // Check and symmetrize matrices if needed
    matrix_symmetry_helper::ensure_matrix_symmetry(A, B, 1e-12, false);

    EigensolverParams params;
    params.nev = 8;
    params.ncv = 16;

    using EVP = ShiftInvertEigenproblem<Mat, blocksize>;
    auto evp = std::make_shared<ShiftInvertEigenproblem<Mat, blocksize>>(*A, B, params.shift);
    auto orth = std::make_shared<orthogonalisation::BlockOrthogonalisation<typename EVP::InnerProduct>>(BetweenBlocks::ClassicalGramSchmidt, WithinBlocks::CholQR2, evp->get_inner_product());

    BlockLanczos lanczos(evp, orth, params, [](auto& x) {
      static std::size_t it = 0;
      std::cout << x.block_view(it++).frobenius_norm() << "\n";
    });
    lanczos.extend(1, params.ncv / blocksize - 1);
  }

  return 0;
}
