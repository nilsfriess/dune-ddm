#pragma once

#include "../logger.hh"

#include <cstddef>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>
#include <memory>
#include <unordered_map>
#include <vector>

#if DUNE_DDM_HAVE_UMFPACK_SIMD
#include <experimental/simd>
#endif

#include "../strumpack.hh"

/**
 * @brief Energy-minimizing extension from boundary to interior of a subdomain region.
 *
 * This class implements energy-minimizing extension that takes values defined on the
 * boundary of a region within a subdomain and extends them harmonically to the interior
 * of that region. The extension minimizes the energy in the sense that it solves:
 *
 *   A_interior * u_interior = -A_interior_boundary * u_boundary
 *
 * where A is the discretized operator (e.g., stiffness matrix), u_boundary contains
 * the known values on the boundary, and u_interior are the unknown interior values
 * to be computed.
 *
 * @tparam Mat Matrix type (typically a DUNE matrix)
 * @tparam Vec Vector type (typically a DUNE vector)
 */
template <class Mat, class Vec>
class EnergyMinimalExtension {
public:
  /**
   * @brief Constructor for energy-minimizing extension operator.
   *
   * @param A The full subdomain matrix (boundary + interior)
   * @param interior_indices Mapping from interior DOF index to subdomain DOF index
   * @param boundary_indices Mapping from boundary DOF index to subdomain DOF index
   */
  EnergyMinimalExtension(const Mat& A, const std::vector<std::size_t>& interior_indices, const std::vector<std::size_t>& boundary_indices)
      : A(A)
      , interior_indices(interior_indices)
      , boundary_indices(boundary_indices)
  {
    // Create mapping from subdomain DOF index to interior DOF index for efficient lookup
    std::unordered_map<std::size_t, std::size_t> subdomain_to_interior;
    subdomain_to_interior.reserve(interior_indices.size());
    for (std::size_t i = 0; i < interior_indices.size(); ++i) subdomain_to_interior[interior_indices[i]] = i;

    // Extract the interior-interior block of the matrix A_ii
    const auto N = interior_indices.size();
    interior_matrix = std::make_shared<Mat>();

    auto avg = A.nonzeroes() / A.N() + 2;
    interior_matrix->setBuildMode(Mat::implicit);
    interior_matrix->setImplicitBuildModeParameters(avg, 0.2);
    interior_matrix->setSize(N, N);
    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci)
        if (subdomain_to_interior.count(ri.index()) and subdomain_to_interior.count(ci.index())) interior_matrix->entry(subdomain_to_interior[ri.index()], subdomain_to_interior[ci.index()]) = *ci;
    }
    interior_matrix->compress();

    // Initialize the direct solver for the interior system
    initialize_solver();
  }

private:
  /**
   * @brief Initialize the direct solver
   */
  void initialize_solver()
  {
#ifndef DUNE_DDM_HAVE_STRUMPACK
    solver = std::make_unique<Dune::UMFPack<Mat>>();
    solver->setOption(UMFPACK_ORDERING, UMFPACK_ORDERING_METIS);
    solver->setOption(UMFPACK_IRSTEP, 0); // Disable iterative refinement for performance
    solver->setMatrix(*interior_matrix);
#else
    solver = std::make_unique<Dune::STRUMPACK<Mat>>(*interior_matrix);
#endif
  }

public:
  EnergyMinimalExtension(const EnergyMinimalExtension&) = delete;
  EnergyMinimalExtension(const EnergyMinimalExtension&&) = delete;
  EnergyMinimalExtension& operator=(const EnergyMinimalExtension&) = delete;
  EnergyMinimalExtension& operator=(const EnergyMinimalExtension&&) = delete;
  ~EnergyMinimalExtension() = default;

  /**
   * @brief Perform energy-minimizing extension from boundary to interior.
   *
   * Given values on the boundary of the region, this function computes the
   * energy-minimizing (harmonic) extension to the interior by solving:
   *   A_ii * u_i = -A_ib * u_b
   * where u_b are the given boundary values and u_i are the computed interior values.
   *
   * @param boundary_values Vector containing values on the boundary DOFs
   * @return Vector containing the computed interior values
   */
  Vec extend(const Vec& boundary_values)
  {
#if DUNE_DDM_HAVE_UMFPACK_SIMD
    // Use helper function to reduce code duplication
    Vec rhs_interior = compute_interior_rhs(boundary_values);
#else
    // Copy the boundary values to a vector that lives on the whole subdomain
    Vec v_full(A.N());
    v_full = 0;
    for (std::size_t i = 0; i < boundary_values.N(); ++i) v_full[boundary_indices[i]] = boundary_values[i];

    // Multiply by the whole subdomain matrix: A * [0; u_boundary]
    Vec A_vfull(A.N());
    A.mv(v_full, A_vfull);

    // Extract the values corresponding to interior DOFs (this gives A_ib * u_b)
    Vec rhs_interior(interior_indices.size());
    for (std::size_t i = 0; i < interior_indices.size(); ++i) rhs_interior[i] = A_vfull[interior_indices[i]];
#endif

    // Solve A_ii * u_interior = -A_ib * u_boundary
    Vec interior_solution(interior_indices.size());
    interior_solution = 0;
    Dune::InverseOperatorResult res;
    solver->apply(interior_solution, rhs_interior, res);
    interior_solution *= -1.;

    return interior_solution;
  }

#if DUNE_DDM_HAVE_UMFPACK_SIMD
private:
  /**
   * @brief Helper function to compute A * [0; u_boundary] for a single boundary vector.
   *
   * This is extracted as a helper to reduce code duplication between single and SIMD versions.
   *
   * @param boundary_values Input boundary values
   * @return Result of matrix-vector multiplication restricted to interior DOFs
   */
  Vec compute_interior_rhs(const Vec& boundary_values) const
  {
    // Copy the boundary values to a vector that lives on the whole subdomain
    Vec v_full(A.N());
    v_full = 0;
    for (std::size_t i = 0; i < boundary_values.N(); ++i) v_full[boundary_indices[i]] = boundary_values[i];

    // Multiply by the whole subdomain matrix: A * [0; u_boundary]
    Vec A_vfull(A.N());
    A.mv(v_full, A_vfull);

    // Extract the values corresponding to interior DOFs (this gives A_ib * u_b)
    Vec rhs_interior(interior_indices.size());
    for (std::size_t i = 0; i < interior_indices.size(); ++i) rhs_interior[i] = A_vfull[interior_indices[i]];

    return rhs_interior;
  }

public:
  /**
   * @brief SIMD version for extending multiple boundary value vectors simultaneously.
   *
   * This function performs energy-minimizing extension for multiple boundary value
   * vectors in parallel using SIMD instructions for improved performance.
   *
   * @param boundary_vectors Vector of boundary value vectors to extend
   * @return Vector of interior solution vectors
   * @note The number of input vectors must be divisible by the SIMD width
   */
  std::vector<Vec> extend(const std::vector<Vec>& boundary_vectors)
  {
    namespace stdx = std::experimental;

    using Scalar = double;
    using ScalarV = stdx::native_simd<Scalar>;

    if (boundary_vectors.size() % ScalarV::size() != 0) DUNE_THROW(Dune::Exception, "Number of vectors must be divisible by SIMD width " + std::to_string(ScalarV::size()) + "\n");

    Vec zero(interior_indices.size());
    zero = 0;
    std::vector<Vec> interior_solutions(boundary_vectors.size(), zero);

    auto numeric_data = solver->get_numeric_data();

    std::vector<ScalarV> rhs_interior_simd(interior_indices.size());
    for (std::size_t k = 0; k < boundary_vectors.size() / ScalarV::size(); ++k) {
      auto block_start = k * ScalarV::size();

      // Process SIMD block of vectors
      for (std::size_t i = block_start; i < block_start + ScalarV::size(); ++i) {
        Vec rhs_interior = compute_interior_rhs(boundary_vectors[i]);

        // Pack into SIMD vector
        for (std::size_t j = 0; j < interior_indices.size(); ++j) rhs_interior_simd[j][i % ScalarV::size()] = rhs_interior[j];
      }

      // Apply inverse of interior matrix to all vectors of the current block
      std::vector<ScalarV> interior_solutions_simd(interior_indices.size());
      solver->apply_simd(interior_solutions_simd, rhs_interior_simd, numeric_data);

      // Extract results and apply sign correction
      for (std::size_t i = block_start; i < block_start + ScalarV::size(); ++i)
        for (std::size_t j = 0; j < interior_indices.size(); ++j) interior_solutions[i][j] = -interior_solutions_simd[j][i % ScalarV::size()];
    }

    return interior_solutions;
  }
#endif

private:
  const Mat& A;                                     ///< Reference to the full subdomain matrix
  const std::vector<std::size_t>& interior_indices; ///< Mapping from interior DOF index to subdomain DOF index
  const std::vector<std::size_t>& boundary_indices; ///< Mapping from boundary DOF index to subdomain DOF index

#ifndef DUNE_DDM_HAVE_STRUMPACK
  std::unique_ptr<Dune::UMFPack<Mat>> solver; ///< Direct solver for interior system (UMFPACK)
#else
  std::unique_ptr<Dune::STRUMPACK<Mat>> solver; ///< Direct solver for interior system (STRUMPACK)
#endif

  std::shared_ptr<Mat> interior_matrix; ///< Interior-interior block matrix A_ii
};
