#pragma once

#include "eigensolver_params.hh"
#include "eigensolver_result.hh"
#include "trl/eigensolvers/lanczos.hh"
#include "trl/eigensolvers/params.hh"
#include "trl/impl/openmp/evp_base.hh"
#include "umfpack.hh"

#include <cstdlib>
#include <dune/ddm/logger.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/umfpack.hh>
#include <memory>
#include <random>
#include <sstream>

namespace detail {
constexpr std::size_t align_up(std::size_t value, std::size_t alignment) { return (value + (alignment - 1)) & ~(alignment - 1); }
} // namespace detail

template <class Mat, unsigned int blocksize>
class TRLGEVP : public trl::openmp::EVPBase<typename Mat::field_type, blocksize> {
public:
  using Base = trl::openmp::EVPBase<typename Mat::field_type, blocksize>;
  using BlockView = typename Base::BlockView;

  TRLGEVP(const Mat& A, const Mat& B, double shift)
      : Base(A.N())
      , B(B)
      , tmp((typename Mat::field_type*)std::aligned_alloc(64, sizeof(typename Mat::field_type) * detail::align_up(A.N() * blocksize, 64)))
      , tmp_view(tmp, A.N())
  {
    auto A_minus_sigma_B = A;
    A_minus_sigma_B.axpy(-shift, B);

    solver = std::make_unique<UMFPackMultivecSolver>(A_minus_sigma_B);
  }

  TRLGEVP(const Mat& A, const Mat& B, Dune::UMFPack<Mat>& constraint_solver_, const std::vector<bool>* subdomain_boundary, double shift)
      : Base(A.N())
      , B(B)
      , tmp((typename Mat::field_type*)std::aligned_alloc(64, sizeof(typename Mat::field_type) * detail::align_up(A.N() * blocksize, 64)))
      , tmp_view(tmp, A.N())
      , constraint_solver(std::make_unique<UMFPackMultivecSolver>(constraint_solver_))
      , subdomain_boundary(subdomain_boundary)
  {
    auto A_minus_sigma_B = A;
    A_minus_sigma_B.axpy(-shift, B);

    solver = std::make_unique<UMFPackMultivecSolver>(A_minus_sigma_B);
  }

  void apply(BlockView Xin, BlockView Yout) override
  {
    if (constraint_solver) {
      applyB(Xin, Yout);
      solver->solve(tmp_view, Yout);

      for (std::size_t i = 0; i < subdomain_boundary->size(); ++i) {
        if ((*subdomain_boundary)[i] == 0)
          for (unsigned int j = 0; j < blocksize; ++j) tmp_view.data_[i * blocksize + j] = 0;
      }
      constraint_solver->solve(Yout, tmp_view);
    }
    else {
      applyB(Xin, tmp_view);
      solver->solve(Yout, tmp_view);
    }
  }

  void dot(typename Base::BlockView V, typename Base::BlockView W, typename Base::BlockMatrixBlockView R) override
  {
    applyB(W, tmp_view);
    V.dot(tmp_view, R);
  }

  // private:
  void applyB(BlockView Xin, BlockView Yout)
  {
    auto* xdata = Xin.data_;
    auto* ydata = Yout.data_;
#pragma omp parallel for schedule(static) default(none) shared(B, xdata, ydata)
    for (auto ri = B.begin(); ri < B.end(); ++ri) {
      const std::size_t row_idx = ri.index();

// Initialize output row to zero
#pragma omp simd
      for (std::size_t j = 0; j < blocksize; ++j) ydata[row_idx * blocksize + j] = 0;

      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        auto coeff = *ci;
        auto col_idx = ci.index();

#pragma omp simd
        for (std::size_t j = 0; j < blocksize; ++j) ydata[row_idx * blocksize + j] += coeff * xdata[col_idx * blocksize + j];
      }
    }
  }

  const Mat& B;

  std::unique_ptr<UMFPackMultivecSolver> solver;

  typename Mat::field_type* tmp;
  BlockView tmp_view;

  std::unique_ptr<UMFPackMultivecSolver> constraint_solver{nullptr};
  const std::vector<bool>* subdomain_boundary = nullptr;
  int apply_constraint_every = 1;
  int count = 0;
};

template <class Mat, class Scalar = typename Mat::field_type>
[[nodiscard]] ddm::GevpSolution<Scalar> trl_gevp(const Mat& A, const Mat& B, const EigensolverParams& params)
{
  constexpr unsigned int blocksize = 1;
  using EVP = TRLGEVP<Mat, blocksize>;
  auto evp = std::make_shared<EVP>(A, B, params.shift);

  trl::EigensolverParams lanczos_params{.nev = params.nev, .ncv = params.ncv, .max_restarts = 1000, .tolerance = params.tolerance};
  trl::BlockLanczos<EVP> lanczos(evp, lanczos_params);

  std::mt19937 rng(params.seed);
  std::normal_distribution<double> dist;
  auto V0 = lanczos.initial_block();
  auto temp_storage = evp->create_multivector(A.N(), blocksize);
  auto T0 = temp_storage.block_view(0);
  std::generate_n(T0.data_, A.N() * blocksize, [&]() { return dist(rng); });
  evp->apply(T0, V0);

  lanczos.solve();

  auto& V = lanczos.get_basis();
  const auto& Y = evp->get_current_eigenvectors();
  // Debug output
  const auto& eigenvalues = evp->get_current_eigenvalues();

  if (eigenvalues.empty()) {
    logger::warn_all("TRL eigensolver produced no eigenvalues. Returning empty basis.");
    return {};
  }

  // Transform the Ritz values back to eigenvalues of the original problem. Because of the
  // shift-invert transformation the Ritz value mu corresponds to lambda = 1 / mu + shift.
  std::vector<Scalar> transformed_eigenvalues;
  transformed_eigenvalues.reserve(params.nev);
  for (std::size_t i = 0; i < params.nev; ++i) transformed_eigenvalues.push_back(static_cast<Scalar>(1 / eigenvalues[i] + params.shift));

  {
    std::ostringstream eval_stream;
    for (std::size_t i = 0; i < transformed_eigenvalues.size(); ++i) {
      if (i > 0) eval_stream << ", ";
      eval_stream << transformed_eigenvalues[i];
    }
    logger::info_all("Computed {} eigenvalues: {}", eigenvalues.size(), eval_stream.str());
  }

  // Compute Ritz vectors: z_j = V * y_j where y_j is the j-th column of Y
  // V has (A.N() rows) x (params.ncv / blocksize blocks)
  // Y has (params.ncv / blocksize) x (params.ncv / blocksize) blocks
  std::vector<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>> ritz_vectors;

  // For each converged eigenvalue
  for (std::size_t j = 0; j < params.nev; ++j) {
    Dune::BlockVector<Dune::FieldVector<Scalar, 1>> z(A.N());

    // Get a temporary block view for accumulation
    auto z_block = temp_storage.block_view(0);
    z_block.set_zero();

    // Accumulate: z += V_i * Y_{i,j}
    // This uses the same method as the Lanczos restart code
    for (std::size_t i = 0; i < params.ncv / blocksize; ++i) {
      auto Vi = V.block_view(i);
      auto Yij = Y.block_view(i, j / blocksize);

      // For blocksize=1: z += Vi * Y[i,j] where Vi is a column vector and Y[i,j] is a scalar
      // For blocksize>1: extracts the (j % blocksize)-th column from the block multiplication
      Vi.mult_add(Yij, z_block);
    }

    // Copy result from BlockView to Dune::BlockVector
    // For blocksize=1, this is straightforward; for blocksize>1, extract the appropriate column
    const std::size_t col_in_block = j % blocksize;
    for (std::size_t row = 0; row < A.N(); ++row) z[row] = z_block.data_[row * blocksize + col_in_block];

    ritz_vectors.push_back(z);
  }

  // TRL does not currently surface iteration/operator-application counts, so we only report
  // convergence here.
  return {.eigenvectors = std::move(ritz_vectors), .eigenvalues = std::move(transformed_eigenvalues), .info = {.converged = true}};
}

template <class Mat, class Scalar = typename Mat::field_type>
[[nodiscard]] ddm::GevpSolution<Scalar>
trl_gevp(const Mat& A, const Mat& B, Dune::InverseOperator<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>, Dune::BlockVector<Dune::FieldVector<Scalar, 1>>>* constraint_solver,
         const std::vector<bool>& subdomain_boundary, const EigensolverParams& params)
{
  constexpr unsigned int blocksize = 1;
  using EVP = TRLGEVP<Mat, blocksize>;

  if (auto umfpack_constraint_solver = dynamic_cast<Dune::UMFPack<Mat>*>(constraint_solver)) {
    auto evp = std::make_shared<EVP>(A, B, *umfpack_constraint_solver, &subdomain_boundary, params.shift);

    trl::EigensolverParams lanczos_params{.nev = params.nev, .ncv = params.ncv, .max_restarts = params.maxit, .tolerance = params.tolerance};
    trl::BlockLanczos<EVP> lanczos(evp, lanczos_params);

    std::mt19937 rng(params.seed);
    std::normal_distribution<double> dist;
    auto V0 = lanczos.initial_block();
    auto temp_storage = evp->create_multivector(A.N(), blocksize);
    auto T0 = temp_storage.block_view(0);
    std::generate_n(T0.data_, A.N() * blocksize, [&]() { return dist(rng); });
    evp->apply(T0, V0);

    lanczos.solve();

    auto& V = lanczos.get_basis();
    // for (std::size_t i = 0; i < params.nev / blocksize; ++i) {
    //   auto Vi = V.block_view(i);
    //   T0.copy_from(Vi);

    //   for (std::size_t i = 0; i < subdomain_boundary.size(); ++i) {
    //     if (subdomain_boundary[i] == 0)
    //       for (unsigned int j = 0; j < blocksize; ++j) T0.data_[i * blocksize + j] = 0;
    //   }

    //   evp->constraint_solver->solve(Vi, T0);
    // }

    const auto& Y = evp->get_current_eigenvectors();
    // Debug output
    const auto& eigenvalues = evp->get_current_eigenvalues();

    if (eigenvalues.empty()) {
      logger::warn_all("TRL eigensolver (constrained) produced no eigenvalues. Returning empty basis.");
      return {};
    }

    // Transform the Ritz values back to eigenvalues of the original problem. Because of the
    // shift-invert transformation the Ritz value mu corresponds to lambda = 1 / mu + shift.
    std::vector<Scalar> transformed_eigenvalues;
    transformed_eigenvalues.reserve(params.nev);
    for (std::size_t i = 0; i < params.nev; ++i) transformed_eigenvalues.push_back(static_cast<Scalar>(1 / eigenvalues[i] + params.shift));

    {
      std::ostringstream eval_stream;
      for (std::size_t i = 0; i < transformed_eigenvalues.size(); ++i) {
        if (i > 0) eval_stream << ", ";
        eval_stream << transformed_eigenvalues[i];
      }
      logger::info_all("Computed {} eigenvalues: {}", eigenvalues.size(), eval_stream.str());
    }

    // Compute Ritz vectors: z_j = V * y_j where y_j is the j-th column of Y
    // V has (A.N() rows) x (params.ncv / blocksize blocks)
    // Y has (params.ncv / blocksize) x (params.ncv / blocksize) blocks
    std::vector<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>> ritz_vectors;

    // For each converged eigenvalue
    for (std::size_t j = 0; j < params.nev; ++j) {
      Dune::BlockVector<Dune::FieldVector<Scalar, 1>> z(A.N());

      // Get a temporary block view for accumulation
      auto z_block = temp_storage.block_view(0);
      z_block.set_zero();

      // Accumulate: z += V_i * Y_{i,j}
      // This uses the same method as the Lanczos restart code
      for (std::size_t i = 0; i < params.ncv / blocksize; ++i) {
        auto Vi = V.block_view(i);
        auto Yij = Y.block_view(i, j / blocksize);

        // For blocksize=1: z += Vi * Y[i,j] where Vi is a column vector and Y[i,j] is a scalar
        // For blocksize>1: extracts the (j % blocksize)-th column from the block multiplication
        Vi.mult_add(Yij, z_block);
      }

      // Copy result from BlockView to Dune::BlockVector
      // For blocksize=1, this is straightforward; for blocksize>1, extract the appropriate column
      const std::size_t col_in_block = j % blocksize;
      for (std::size_t row = 0; row < A.N(); ++row) z[row] = z_block.data_[row * blocksize + col_in_block];

      ritz_vectors.push_back(z);
    }

    // TRL does not currently surface iteration/operator-application counts, so we only report
    // convergence here.
    return {.eigenvectors = std::move(ritz_vectors), .eigenvalues = std::move(transformed_eigenvalues), .info = {.converged = true}};
  }
  else {
    throw std::invalid_argument("The constraint solver must be of type Dune::UMFPack");
  }

  return {};
}
