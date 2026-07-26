#pragma once

/** @file coarse_spaces.hh
    @brief Free functions to build coarse space bases such as GenEO and MsGFEM spectral coarse spaces.

    This file provides named free functions for building various types of coarse spaces
    used in domain decomposition methods. The coarse spaces are organised along two independent
    axes that can be configured from an ini file:

      [coarsespace]
      domain = full              # full | ring
      constraint = none          # none | harmonic
      constraint_evp = lagrange_multiplier  # lagrange_multiplier | alternating (only for constraint = harmonic)
      ring_extension = from_ring            # from_ring | from_boundary (only for domain = ring)

    The four resulting combinations are implemented by:
      - build_geneo_coarse_space()           (domain = full,  constraint = none)
      - build_geneo_ring_coarse_space()      (domain = ring,  constraint = none)
      - build_msgfem_coarse_space()          (domain = full,  constraint = harmonic)
      - build_msgfem_ring_coarse_space()     (domain = ring,  constraint = harmonic)
 */

#include "../eigensolvers/eigensolvers.hh"
#include "../logger.hh"
#include "../pou.hh"
#include "energy_minimal_extension.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/timer.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>
#include <mpi.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace detail {
/**
 * @brief Apply partition of unity scaling and normalise eigenvectors.
 *
 * Applies final processing to eigenvectors: scales each component by the partition
 * of unity and normalises to unit length. This is the common final step in all
 * coarse space construction methods.
 *
 * @param eigenvectors Vector of eigenvectors to process (modified in-place).
 * @param pou Partition of unity.
 */
template <class Vec>
inline void finalize_eigenvectors(std::vector<Vec>& eigenvectors, const PartitionOfUnity& pou)
{
  for (auto& vec : eigenvectors) {
    // Apply partition of unity scaling
    for (std::size_t i = 0; i < vec.size(); ++i) vec[i] *= pou[i];
    // Normalize to unit length
    vec *= 1. / vec.two_norm();
  }
}

/**
 * @brief Scale matrix entries with partition of unity weights, i.e. compute C <- D*C*D
 *
 * Modifies matrix C in-place by scaling each entry C[i][j] with pou[i] * pou[j].
 * This creates the weighted matrix commonly used in GenEO-type eigenproblems.
 *
 * @param C Matrix to scale (modified in-place).
 * @param pou Partition of unity vector for scaling.
 * @param index_mapping Optional mapping from matrix indices to pou indices.
 *                      If empty, direct indexing is used.
 */
template <class Mat, class Vec>
void scale_matrix_with_pou(Mat& C, const Vec& pou, const std::vector<std::size_t>& index_mapping = {})
{
  for (auto ri = C.begin(); ri != C.end(); ++ri) {
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      std::size_t i_idx = index_mapping.empty() ? ri.index() : index_mapping[ri.index()];
      std::size_t j_idx = index_mapping.empty() ? ci.index() : index_mapping[ci.index()];
      *ci *= pou[i_idx] * pou[j_idx];
    }
  }
}

enum class DOFType : std::uint8_t { Interior, Boundary, Dirichlet };

struct DOFPartition {
  std::vector<DOFType> partition;

  std::size_t n_interior;
  std::size_t n_boundary;
  std::size_t n_dirichlet;
};

template <class Mat>
DOFPartition partition_dofs(const Mat& A, const std::vector<bool>& subdomain_boundary_mask, const std::vector<bool>& dirichlet_mask)
{
  std::vector<DOFType> dof_partitioning(A.N());

  std::size_t num_interior = 0;
  std::size_t num_boundary = 0;
  std::size_t num_dirichlet = 0;
  for (std::size_t i = 0; i < A.N(); ++i) {
    if (dirichlet_mask[i] > 0) {
      dof_partitioning[i] = DOFType::Dirichlet;
      num_dirichlet++;
    }
    else if (subdomain_boundary_mask[i]) {
      dof_partitioning[i] = DOFType::Boundary;
      num_boundary++;
    }
    else {
      dof_partitioning[i] = DOFType::Interior;
      num_interior++;
    }
  }
  logger::trace_all("Partitioned dofs, have {} in interior, {} on subdomain boundary, {} on Dirichlet boundary", num_interior, num_boundary, num_dirichlet);

  return DOFPartition{
      .partition = std::move(dof_partitioning),
      .n_interior = num_interior,
      .n_boundary = num_boundary,
      .n_dirichlet = num_dirichlet,
  };
}

/** @brief Wrapper struct for the data returned by get_ring_info()
 *
 *  @note Both sets defined in this struct use subdomain-local indices, not ring-local indices.
 */
struct RingInfo {
  std::unordered_map<std::size_t, std::size_t> subdomain_to_ring; ///< Maps subdomain indices (that exist in the ring) to ring-local indices
  std::unordered_set<std::size_t> inner_boundary;                 ///< The DOFs on the boundary between ring and interior
  std::unordered_set<std::size_t> behind_inner_boundary;          ///< DOFs one layer into the ring (from the inner boundary)
};

/** @brief Analyse the ring region and find DOFs on the inner boundary and DOFs one layer into the ring.
 */
template <class Mat>
RingInfo get_ring_info(const Mat& A_sub, const std::vector<std::size_t>& ring_region_to_subdomain)
{
  RingInfo ring_info;

  // First we create a subdomain -> ring mapping
  ring_info.subdomain_to_ring.reserve(ring_region_to_subdomain.size());
  for (std::size_t i = 0; i < ring_region_to_subdomain.size(); ++i) ring_info.subdomain_to_ring[ring_region_to_subdomain[i]] = i;

  // Next identify the inner boundary: These are the DOFs in the ring that have neighbours (in terms of the matrix graph
  // of A_sub) that are not in the ring.
  for (auto ring_dof : ring_region_to_subdomain) {
    for (auto ci = A_sub[ring_dof].begin(); ci != A_sub[ring_dof].end(); ++ci)
      if ((ci.index() != ring_dof) and not ring_info.subdomain_to_ring.contains(ci.index())) {
        ring_info.inner_boundary.insert(ring_dof);
        break;
      }
  }

  // Finally, identify the DOFs one layer into the ring. These are DOFs that are:
  // - inside the ring
  // - not on the boundary of the ring
  // - connected to a boundary DOF
  for (auto ring_dof : ring_region_to_subdomain) {
    if (ring_info.inner_boundary.contains(ring_dof)) continue;
    for (auto ci = A_sub[ring_dof].begin(); ci != A_sub[ring_dof].end(); ++ci)
      if ((ci.index() != ring_dof) and ring_info.inner_boundary.contains(ci.index())) {
        ring_info.behind_inner_boundary.insert(ring_dof);
        break;
      }
  }

  logger::trace_all("Created ring info: have {} DOFs on inner ring boundary, {} DOFs one layer into the ring", ring_info.inner_boundary.size(), ring_info.behind_inner_boundary.size());

  return ring_info;
}

/** @brief Create a modified partition of unity for the ring coarse spaces
 *
 * Takes the given partition of unity and creates a std::vector<Scalar> that represents a
 * modified partition of unity function that only lives on the ring and vanishes on the
 * inner boundary of the ring, as required by the ring-type coarse spaces.
 */
template <class Scalar>
std::vector<Scalar> make_ring_pou(const PartitionOfUnity& pou, const std::vector<std::size_t>& ring_region_to_subdomain, const RingInfo& ring_info)
{
  std::vector<Scalar> ring_pou(ring_region_to_subdomain.size());
  for (std::size_t i = 0; i < ring_pou.size(); ++i) {
    auto index_in_subdomain = ring_region_to_subdomain[i];
    if (ring_info.inner_boundary.contains(index_in_subdomain)) ring_pou[i] = 0;
    else ring_pou[i] = pou[index_in_subdomain];
  }
  return ring_pou;
}

/**
 * @brief Extract the ring-local submatrix from a full subdomain matrix.
 *
 * Given a matrix A defined on the full subdomain and a mapping from ring indices
 * to subdomain indices, extracts the submatrix corresponding to ring-ring couplings
 * using ring-local indexing.
 *
 * @param A Full subdomain matrix.
 * @param ring_region_to_subdomain Mapping from ring-local indices to subdomain indices.
 * @param ring_info Ring information containing the subdomain-to-ring map.
 * @return Ring-local submatrix.
 */
template <class Scalar>
Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>> extract_ring_submatrix(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A, const std::vector<std::size_t>& ring_region_to_subdomain,
                                                                         const RingInfo& ring_info)
{
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;
  const auto n_ring = ring_region_to_subdomain.size();
  const auto avg = A.nonzeroes() / A.N() + 1;

  Matrix A_ring(n_ring, n_ring, avg, 0.2, Matrix::implicit);
  for (std::size_t i = 0; i < n_ring; ++i) {
    auto sub_i = ring_region_to_subdomain[i];
    for (auto cit = A[sub_i].begin(); cit != A[sub_i].end(); ++cit) {
      auto it = ring_info.subdomain_to_ring.find(cit.index());
      if (it != ring_info.subdomain_to_ring.end()) A_ring.entry(i, it->second) = *cit;
    }
  }
  A_ring.compress();
  return A_ring;
}

/**
 * @brief Solve an MsGFEM-type saddle-point eigenvalue problem.
 *
 * Assembles and solves the constrained eigenvalue problem with Lagrange multipliers
 * that enforces A-harmonicity in the interior DOFs. This is the core computation
 * shared between MsGFEM and MsGFEM-ring coarse spaces.
 *
 * @param A_neu Matrix for the eigenproblem (LHS main block and RHS base).
 * @param A_constraint Matrix for the harmonicity constraint (interior rows).
 * @param partition DOF partition into Interior/Boundary/Dirichlet.
 * @param pou POU values indexed by original matrix indices (anything with operator[]).
 * @param eig_ptree Eigensolver parameters.
 * @return Eigenvectors in original matrix indexing (Dirichlet DOFs set to 0).
 */
template <class Scalar, class POU>
[[nodiscard]] ddm::GevpSolution<Scalar> solve_msgfem_saddle_point_evp(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A_neu,
                                                                      const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A_constraint, const DOFPartition& partition, const POU& pou,
                                                                      const Dune::ParameterTree& eig_ptree)
{
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;

  // Create a reordered index set: first interior dofs, then boundary dofs, then Dirichlet dofs
  std::vector<std::size_t> reordering(A_neu.N());
  std::size_t cnt_interior = 0;
  std::size_t cnt_boundary = partition.n_interior;
  std::size_t cnt_dirichlet = partition.n_interior + partition.n_boundary;
  for (std::size_t i = 0; i < reordering.size(); ++i)
    if (partition.partition[i] == DOFType::Interior) reordering[i] = cnt_interior++;
    else if (partition.partition[i] == DOFType::Boundary) reordering[i] = cnt_boundary++;
    else reordering[i] = cnt_dirichlet++;

  // Assemble the left-hand side of the eigenproblem
  Matrix A_lhs;
  const auto n_big = partition.n_interior + partition.n_boundary + partition.n_interior;
  const auto avg = 2 * (A_constraint.nonzeroes() / A_constraint.N());
  A_lhs.setBuildMode(Matrix::implicit);
  A_lhs.setImplicitBuildModeParameters(avg, 0.2);
  A_lhs.setSize(n_big, n_big);

  // Assemble the part corresponding to the a-harmonic constraint
  for (auto rit = A_constraint.begin(); rit != A_constraint.end(); ++rit) {
    auto ii = rit.index();
    auto ri = reordering[ii];
    if (partition.partition[ii] != DOFType::Interior) continue;

    for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
      auto jj = cit.index();
      auto rj = reordering[jj];

      if (partition.partition[jj] != DOFType::Dirichlet) {
        A_lhs.entry(rj, partition.n_interior + partition.n_boundary + ri) = *cit;
        A_lhs.entry(partition.n_interior + partition.n_boundary + ri, rj) = *cit;
      }
    }
  }

  // Assemble the remaining part of the matrix
  for (auto rit = A_neu.begin(); rit != A_neu.end(); ++rit) {
    auto ii = rit.index();
    auto ri = reordering[ii];
    if (partition.partition[ii] == DOFType::Dirichlet) continue;

    for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
      auto jj = cit.index();
      auto rj = reordering[jj];

      if (partition.partition[jj] != DOFType::Dirichlet) A_lhs.entry(ri, rj) = *cit;
    }
  }
  A_lhs.compress();

  // Next, assemble the right-hand side of the eigenproblem
  Matrix B;
  B.setBuildMode(Matrix::implicit);
  B.setImplicitBuildModeParameters(avg, 0.2);
  B.setSize(n_big, n_big);

  for (auto rit = A_neu.begin(); rit != A_neu.end(); ++rit) {
    auto ii = rit.index();
    auto ri = reordering[ii];
    if (partition.partition[ii] != DOFType::Interior) continue;

    for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
      auto jj = cit.index();
      auto rj = reordering[jj];

      if (partition.partition[jj] == DOFType::Interior) B.entry(ri, rj) = pou[ii] * pou[jj] * (*cit);
    }
  }
  B.compress();

  // Solve the eigenproblem
  auto solution = ddm::solve_gevp(A_lhs, B, eig_ptree);
  const auto& eigvecs = solution.eigenvectors;

  // Extract the actual eigenvectors (map from reordered to original indexing). The eigenvalues
  // of the saddle-point problem are exactly the MsGFEM eigenvalues, so we pass them through.
  Vector zero(A_neu.N());
  zero = 0;
  std::vector<Vector> result(eigvecs.size(), zero);
  for (std::size_t k = 0; k < result.size(); ++k) {
    for (std::size_t i = 0; i < A_neu.N(); ++i)
      if (partition.partition[i] != DOFType::Dirichlet) result[k][i] = eigvecs[k][reordering[i]];
  }

  return {.eigenvectors = std::move(result), .eigenvalues = std::move(solution.eigenvalues), .info = solution.info};
}

/**
 * @brief Extend ring eigenvectors to the full subdomain using energy-minimal extension.
 *
 * Takes eigenvectors defined on the ring region and extends them to the full subdomain
 * by operator-harmonically extending from one layer into the ring (behind_inner_boundary)
 * to the subdomain interior and inner ring boundary.
 *
 * @param ring_eigenvectors Eigenvectors defined on the ring region.
 * @param A_sub Full subdomain matrix (for the extension operator).
 * @param ring_region_to_subdomain Mapping from ring indices to subdomain indices.
 * @param ring_info Ring region information (inner boundary, behind inner boundary).
 * @return Eigenvectors defined on the full subdomain.
 */
template <class Scalar>
std::vector<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>> extend_ring_eigenvectors(const std::vector<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>>& ring_eigenvectors,
                                                                                      const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A_sub,
                                                                                      const std::vector<std::size_t>& ring_region_to_subdomain, const RingInfo& ring_info)
{
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;

  // We create two index sets:
  // 1. the interior part of the subdomain combined with the inner boundary of the ring
  // 2. the DOFs one layer into the ring
  // The eigenvectors are then extended by taking their values on the index set 2. and extending them to the index set 1.
  // On the remaining part of the ring, their original values are taken
  std::vector<std::size_t> extension_boundary(ring_info.behind_inner_boundary.begin(), ring_info.behind_inner_boundary.end());
  std::vector<std::size_t> extension_interior(A_sub.N() - ring_region_to_subdomain.size() + ring_info.inner_boundary.size());
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < A_sub.N(); ++i)
    if (ring_info.inner_boundary.contains(i) or not ring_info.subdomain_to_ring.contains(i)) extension_interior[cnt++] = i;
  assert(cnt == extension_interior.size());

  EnergyMinimalExtension<Matrix, Vector> extension(A_sub, extension_interior, extension_boundary);

  Vector zero(A_sub.N());
  zero = 0;
  std::vector<Vector> combined_vectors(ring_eigenvectors.size(), zero);

  Vector dirichlet_data(ring_info.behind_inner_boundary.size());
  for (std::size_t k = 0; k < ring_eigenvectors.size(); ++k) {
    const auto& evec = ring_eigenvectors[k];

    std::size_t cnt = 0;
    for (auto idx : ring_info.behind_inner_boundary) dirichlet_data[cnt++] = evec[ring_info.subdomain_to_ring.at(idx)];

    auto interior_vec = extension.extend(dirichlet_data);

    // First set the values in the ring
    for (std::size_t i = 0; i < evec.N(); ++i) combined_vectors[k][ring_region_to_subdomain[i]] = evec[i];

    // Next fill the interior values (note that the interior and ring now overlap, so this overrides some values)
    for (std::size_t i = 0; i < interior_vec.N(); ++i) combined_vectors[k][extension_interior[i]] = interior_vec[i];
  }

  return combined_vectors;
}

} // namespace detail

/**
 * @brief The result of building a coarse space: the basis plus its spectral provenance.
 *
 * `eigenvalues[i]` is the eigenvalue associated with `basis[i]`. The eigenvalue is invariant
 * under the post-processing applied to the vectors (ring extension, POU scaling, normalisation),
 * so the pairing stays meaningful even though `basis[i]` is no longer the raw eigenvector.
 *
 * For non-spectral coarse spaces (e.g. the POU/Nicolaides space) `eigenvalues` is empty and
 * `info.converged` is true.
 */
template <class Scalar = double>
struct CoarseSpaceResult {
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;

  std::vector<Vector> basis;       ///< POU-scaled, normalised coarse basis vectors.
  std::vector<Scalar> eigenvalues; ///< Eigenvalue for each basis vector (parallel to `basis`); empty for non-spectral spaces.
  ddm::EigensolverResult info{};   ///< Eigensolver run metadata (converged / iterations / operator applications).
};

// ============================================================================
//  Free functions to build spectral coarse spaces
// ============================================================================

/**
 * @brief Build a GenEO coarse space (domain = full, constraint = none).
 *
 * Computes a few eigenvectors corresponding to the smallest eigenvalues of
 * $Ax = \lambda DBDx$, where $A$ and $B$ are the provided matrices and $D$ is a diagonal
 * partition of unity matrix. The computed eigenvectors are scaled by the partition of unity
 * and normalised.
 *
 * @param A Neumann matrix on the overlapping subdomain (left-hand side of eigenproblem).
 * @param B Matrix defined in the overlap region (used to construct right-hand side).
 * @param pou Partition of unity vector (diagonal of D matrix).
 * @param ptree ParameterTree containing solver and selection parameters.
 * @param ptree_prefix Prefix for parameter subtree (default "coarse_space").
 * @return CoarseSpaceResult holding the basis, the associated eigenvalues, and eigensolver metadata.
 */
template <class Scalar = double>
[[nodiscard]] CoarseSpaceResult<Scalar> build_geneo_coarse_space(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A, const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& B,
                                                                 const PartitionOfUnity& pou, const Dune::ParameterTree& ptree, const std::string& ptree_prefix = "coarse_space")
{
  Logger::ScopedLog setup_sl{Logger::get().registerOrGetEvent("Coarse space", "setup")};

  const auto& subtree = ptree_prefix.empty() ? ptree : ptree.sub(ptree_prefix);
  const auto& eig_ptree = subtree.sub("eigensolver");

  auto C = B;
  detail::scale_matrix_with_pou(C, pou);

  auto solution = ddm::solve_gevp(A, C, eig_ptree);

  if (solution.eigenvectors.empty())
    logger::warn_all("build_geneo_coarse_space: Eigensolver returned no basis vectors. "
                     "The coarse space will be empty. Check eigensolver parameters (shift, threshold, nev).");

  detail::finalize_eigenvectors(solution.eigenvectors, pou);
  return {.basis = std::move(solution.eigenvectors), .eigenvalues = std::move(solution.eigenvalues), .info = solution.info};
}

/**
 * @brief Build a GenEO-Ring coarse space (domain = ring, constraint = none). See arXiv:2510.26548.
 *
 * Like the standard GenEO coarse space, the basis vectors are eigenvectors of $Ax = \lambda DBDx$,
 * but $A$ and $B$ are only defined on a ring region around the subdomain boundary, leading to a
 * cheaper eigenproblem. The partition of unity @p pou should be defined on the full subdomain;
 * the ring-local POU is extracted via @p ring_region_to_subdomain.
 *
 * After solving the ring eigenproblem, the eigenvectors must be extended to the full subdomain.
 * The extension mode is read from `ptree.sub(ptree_prefix).get("ring_extension", "from_ring")`:
 * - `"from_ring"`: Energy-minimal extension from one layer into the ring to the subdomain interior,
 *   as described in arXiv:2510.26548. Requires @p A_sub.
 * - `"from_boundary"`: Extends from the overlapping subdomain boundary using $A_\text{sub}^{-1}$.
 *   The resulting basis vectors are operator-harmonic on the whole subdomain by construction.
 *   This variant has not been theoretically analysed but has practical advantages (the inverse
 *   is usually already available from the fine level). Requires @p A_sub_inv and @p subdomain_boundary_mask.
 *
 * @param A Neumann matrix on the ring region (left-hand side of eigenproblem).
 * @param B Matrix defined on the ring region (used to construct right-hand side).
 * @param ring_region_to_subdomain Mapping from ring-local indices to subdomain indices.
 * @param pou Partition of unity vector on the full subdomain.
 * @param A_sub Full subdomain matrix (needed for ring analysis and energy-minimal extension).
 * @param ptree ParameterTree containing solver and selection parameters.
 * @param ptree_prefix Prefix for parameter subtree (default "coarse_space").
 * @param A_sub_inv Inverse of the subdomain matrix (needed only for ring_extension = from_boundary).
 * @param subdomain_boundary_mask Boolean mask for subdomain boundary DOFs (needed only for ring_extension = from_boundary).
 * @return CoarseSpaceResult holding the basis, the associated eigenvalues, and eigensolver metadata.
 */
template <class Scalar = double>
[[nodiscard]] CoarseSpaceResult<Scalar> build_geneo_ring_coarse_space(
    const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A, const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& B, const std::vector<std::size_t>& ring_region_to_subdomain,
    const PartitionOfUnity& pou, const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A_sub, const Dune::ParameterTree& ptree, const std::string& ptree_prefix = "coarse_space",
    Dune::InverseOperator<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>, Dune::BlockVector<Dune::FieldVector<Scalar, 1>>>* A_sub_inv = nullptr, const std::vector<bool>& subdomain_boundary_mask = {})
{
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;

  Logger::ScopedLog setup_sl{Logger::get().registerOrGetEvent("Coarse space", "setup")};

  const auto& subtree = ptree_prefix.empty() ? ptree : ptree.sub(ptree_prefix);
  const auto& eig_ptree = subtree.sub("eigensolver");
  auto ring_extension = subtree.get("ring_extension", "from_ring");

  auto* ring_eigensolver_event = Logger::get().registerOrGetEvent("Ring coarse space", "solve evp");
  auto* ring_extend_event = Logger::get().registerOrGetEvent("Ring coarse space", "extend");

  // Identify important regions within the ring (e.g. the inner boundary)
  auto ring_info = detail::get_ring_info(A_sub, ring_region_to_subdomain);

  // Modified partition of unity that vanishes at the inner boundary of the ring region
  auto ring_pou = detail::make_ring_pou<Scalar>(pou, ring_region_to_subdomain, ring_info);

  // Solve the ring eigenproblem. The eigenvalues characterise the ring spectrum and are invariant
  // under the extension below, so we keep them alongside the (soon to be extended) eigenvectors.
  std::vector<Vector> basis;
  std::vector<Scalar> eigenvalues;
  ddm::EigensolverResult info;
  {
    Logger::ScopedLog sl{ring_eigensolver_event};

    auto C = B;
    detail::scale_matrix_with_pou(C, ring_pou);
    auto solution = ddm::solve_gevp(A, C, eig_ptree);
    basis = std::move(solution.eigenvectors);
    eigenvalues = std::move(solution.eigenvalues);
    info = solution.info;
  }

  // Extend the eigenvectors to the whole subdomain
  {
    Logger::ScopedLog sl{ring_extend_event};

    if (ring_extension == "from_ring") { basis = detail::extend_ring_eigenvectors<Scalar>(basis, A_sub, ring_region_to_subdomain, ring_info); }
    else if (ring_extension == "from_boundary") {
      if (!A_sub_inv) DUNE_THROW(Dune::Exception, "ring_extension = from_boundary requires A_sub_inv to be provided");
      if (subdomain_boundary_mask.empty()) DUNE_THROW(Dune::Exception, "ring_extension = from_boundary requires subdomain_boundary_mask to be provided");

      Vector rhs(A_sub.N());
      Vector zero(A_sub.N());
      zero = 0;
      std::vector<Vector> extended_vectors(basis.size(), zero);

      for (std::size_t i = 0; i < extended_vectors.size(); ++i) {
        rhs = 0;
        for (std::size_t j = 0; j < ring_region_to_subdomain.size(); ++j)
          if (subdomain_boundary_mask[ring_region_to_subdomain[j]]) rhs[ring_region_to_subdomain[j]] = basis[i][j];

        Dune::InverseOperatorResult res;
        A_sub_inv->apply(extended_vectors[i], rhs, res);
      }

      std::swap(basis, extended_vectors);
    }
    else DUNE_THROW(Dune::NotImplemented, "Unknown ring_extension mode: " + ring_extension + " (expected 'from_ring' or 'from_boundary')");
  }

  detail::finalize_eigenvectors(basis, pou);
  return {.basis = std::move(basis), .eigenvalues = std::move(eigenvalues), .info = info};
}

/**
 * @brief Build an MsGFEM coarse space (domain = full, constraint = harmonic).
 *
 * Computes eigenvectors of $Ax = \lambda DADx$ restricted to the $A$-harmonic subspace.
 * For details see:
 * - https://doi.org/10.1007/s10208-025-09711-z
 * - https://doi.org/10.1016/j.jcp.2024.113013
 *
 * The matrices @p A_neu (eigenproblem) and @p A_sub (harmonic constraint) can differ.
 * For symmetric problems they are usually the same; for non-symmetric problems, @p A_neu
 * usually corresponds to a symmetric part of the operator.
 *
 * The variant for solving the constrained eigenproblem is read from
 * `ptree.sub(ptree_prefix).get("constraint_evp", "lagrange_multiplier")`:
 * - `"lagrange_multiplier"`: Enforce the constraint weakly using Lagrange multipliers,
 *   leading to a saddle-point eigenproblem of twice the original size. Requires @p A_sub
 *   and @p dirichlet_mask.
 * - `"alternating"`: Apply $P^T A P$ on-the-fly using $A_\text{sub}^{-1}$ to project
 *   into the harmonic subspace in each eigensolver iteration. Usually faster if the
 *   factorised $A_\text{sub}^{-1}$ is already available. Requires @p A_sub_inv.
 *
 * @param A_neu Neumann matrix on the overlapping subdomain.
 * @param pou Partition of unity vector (diagonal of D matrix).
 * @param subdomain_boundary_mask Boolean mask for subdomain boundary DOFs.
 * @param ptree ParameterTree containing solver and selection parameters.
 * @param ptree_prefix Prefix for parameter subtree (default "coarse_space").
 * @param A_sub Subdomain matrix (needed for constraint_evp = lagrange_multiplier).
 * @param dirichlet_mask Boolean mask for global Dirichlet boundary DOFs (needed for constraint_evp = lagrange_multiplier).
 * @param A_sub_inv Inverse of subdomain matrix (needed for constraint_evp = alternating).
 * @return CoarseSpaceResult holding the basis, the associated eigenvalues, and eigensolver metadata.
 */
template <class Scalar = double>
[[nodiscard]] CoarseSpaceResult<Scalar>
build_msgfem_coarse_space(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A_neu, const PartitionOfUnity& pou, const std::vector<bool>& subdomain_boundary_mask,
                          const Dune::ParameterTree& ptree, const std::string& ptree_prefix = "coarse_space", const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>* A_sub = nullptr,
                          const std::vector<bool>& dirichlet_mask = {},
                          Dune::InverseOperator<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>, Dune::BlockVector<Dune::FieldVector<Scalar, 1>>>* A_sub_inv = nullptr)
{
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;

  Logger::ScopedLog setup_sl{Logger::get().registerOrGetEvent("Coarse space", "setup")};

  const auto& subtree = ptree_prefix.empty() ? ptree : ptree.sub(ptree_prefix);
  const auto& eig_ptree = subtree.sub("eigensolver");
  auto constraint_evp = subtree.get("constraint_evp", "lagrange_multiplier");

  std::vector<Vector> basis;
  std::vector<Scalar> eigenvalues;
  ddm::EigensolverResult info;

  if (constraint_evp == "lagrange_multiplier") {
    if (!A_sub) DUNE_THROW(Dune::Exception, "constraint_evp = lagrange_multiplier requires A_sub to be provided");
    if (A_sub->N() != A_neu.N()) DUNE_THROW(Dune::Exception, "The two matrices must have the same size");
    if (dirichlet_mask.size() != A_sub->N()) DUNE_THROW(Dune::Exception, "The matrix and the Dirichlet mask must have the same size");
    if (pou.size() != A_sub->N()) DUNE_THROW(Dune::Exception, "The matrix and the partition of unity must have the same size");

    auto partition = detail::partition_dofs(*A_sub, subdomain_boundary_mask, dirichlet_mask);
    auto solution = detail::solve_msgfem_saddle_point_evp<Scalar>(A_neu, *A_sub, partition, pou, eig_ptree);
    basis = std::move(solution.eigenvectors);
    eigenvalues = std::move(solution.eigenvalues);
    info = solution.info;
  }
  else if (constraint_evp == "alternating") {
    if (!A_sub_inv) DUNE_THROW(Dune::Exception, "constraint_evp = alternating requires A_sub_inv to be provided");
    if (pou.size() != A_neu.N()) DUNE_THROW(Dune::Exception, "The matrix and the partition of unity must have the same size");

    Matrix C(A_neu);
    detail::scale_matrix_with_pou(C, pou);
    auto solution = ddm::solve_gevp(A_neu, C, A_sub_inv, subdomain_boundary_mask, eig_ptree);
    basis = std::move(solution.eigenvectors);
    eigenvalues = std::move(solution.eigenvalues);
    info = solution.info;
  }
  else DUNE_THROW(Dune::NotImplemented, "Unknown constraint_evp mode: " + constraint_evp + " (expected 'lagrange_multiplier' or 'alternating')");

  if (basis.empty())
    logger::warn_all("build_msgfem_coarse_space: Eigensolver returned no basis vectors. "
                     "The coarse space will be empty and no Galerkin preconditioner will be set up. "
                     "Check eigensolver parameters (shift, threshold, nev).");

  detail::finalize_eigenvectors(basis, pou);
  return {.basis = std::move(basis), .eigenvalues = std::move(eigenvalues), .info = info};
}

/**
 * @brief Build an MsGFEM-Ring coarse space (domain = ring, constraint = harmonic).
 *
 * Combines the MsGFEM approach (constrained eigenproblem enforcing $A$-harmonicity)
 * with the ring approach (restricting the eigenproblem to a ring region around the subdomain
 * boundary). The inner boundary of the ring is treated as a subdomain boundary of the ring
 * sub-problem, while the actual subdomain boundary forms the other boundary.
 *
 * The DOFs within the ring are partitioned as follows:
 * - Interior: ring DOFs not on the inner ring boundary, not on the subdomain boundary, and not
 *   on the Dirichlet boundary. The $A$-harmonicity constraint is enforced for these DOFs.
 * - Boundary: ring DOFs on the inner ring boundary or on the subdomain boundary (but not Dirichlet).
 *   These are free (unconstrained) DOFs in the eigenproblem.
 * - Dirichlet: ring DOFs on the global Dirichlet boundary. These are excluded from the system.
 *
 * The variant for solving the constrained eigenproblem is read from
 * `ptree.sub(ptree_prefix).get("constraint_evp", "lagrange_multiplier")`:
 * - `"lagrange_multiplier"`: The $A$-harmonicity constraint is assembled using the ring-local
 *   restriction of the full subdomain matrix @p A_sub, leading to a saddle-point eigenproblem.
 *   This is important for non-symmetric problems where @p A_neu and @p A_sub correspond to
 *   different PDEs. Requires @p dirichlet_mask.
 * - `"alternating"`: Apply $P^T A P$ on-the-fly using an internally constructed UMFPack solver
 *   on the ring-local matrix to project into the harmonic subspace during each eigensolver
 *   iteration. Does not require an externally provided solver.
 *
 * After solving the constrained eigenproblem on the ring, the eigenvectors are extended to the
 * full subdomain. The extension mode is read from
 * `ptree.sub(ptree_prefix).get("ring_extension", "from_ring")`:
 * - `"from_ring"`: Energy-minimal extension from one layer into the ring to the subdomain interior,
 *   as in the GenEO-Ring coarse space.
 * - `"from_boundary"`: Extends from the overlapping subdomain boundary using $A_\text{sub}^{-1}$.
 *   Requires @p A_sub_inv.
 *
 * @param A_neu Neumann matrix on the ring region (for the eigenproblem LHS main block and RHS).
 * @param ring_region_to_subdomain Mapping from ring-local indices to subdomain indices.
 * @param pou Partition of unity vector on the full subdomain.
 * @param A_sub Full subdomain matrix (for the harmonicity constraint, ring info, and extension).
 * @param dirichlet_mask Boolean mask (full subdomain) for global Dirichlet boundary DOFs (needed for constraint_evp = lagrange_multiplier).
 * @param subdomain_boundary_mask Boolean mask (full subdomain) for subdomain boundary DOFs.
 * @param ptree ParameterTree containing solver and selection parameters.
 * @param ptree_prefix Prefix for parameter subtree (default "coarse_space").
 * @param A_sub_inv Inverse of the subdomain matrix (needed only for ring_extension = from_boundary).
 * @return CoarseSpaceResult holding the basis, the associated eigenvalues, and eigensolver metadata.
 */
template <class Scalar = double>
[[nodiscard]] CoarseSpaceResult<Scalar>
build_msgfem_ring_coarse_space(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A_neu, const std::vector<std::size_t>& ring_region_to_subdomain, const PartitionOfUnity& pou,
                               const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A_sub, const std::vector<bool>& dirichlet_mask, const std::vector<bool>& subdomain_boundary_mask,
                               const Dune::ParameterTree& ptree, const std::string& ptree_prefix = "coarse_space",
                               Dune::InverseOperator<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>, Dune::BlockVector<Dune::FieldVector<Scalar, 1>>>* A_sub_inv = nullptr)
{
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;

  Logger::ScopedLog setup_sl{Logger::get().registerOrGetEvent("Coarse space", "setup")};

  const auto& subtree = ptree_prefix.empty() ? ptree : ptree.sub(ptree_prefix);
  const auto& eig_ptree = subtree.sub("eigensolver");
  auto ring_extension = subtree.get("ring_extension", "from_ring");
  auto constraint_evp = subtree.get("constraint_evp", "lagrange_multiplier");

  auto* ring_eigensolver_event = Logger::get().registerOrGetEvent("MsGFEM-Ring coarse space", "solve evp");
  auto* ring_extend_event = Logger::get().registerOrGetEvent("MsGFEM-Ring coarse space", "extend");

  // Identify important regions within the ring (inner boundary, behind inner boundary)
  auto ring_info = detail::get_ring_info(A_sub, ring_region_to_subdomain);

  // Create ring-local boundary mask. The inner boundary of the ring is treated as subdomain boundary.
  const auto n_ring = ring_region_to_subdomain.size();
  std::vector<bool> ring_boundary_mask(n_ring);
  for (std::size_t i = 0; i < n_ring; ++i) {
    auto sub_idx = ring_region_to_subdomain[i];
    ring_boundary_mask[i] = ring_info.inner_boundary.contains(sub_idx) || subdomain_boundary_mask[sub_idx];
  }

  // Create ring POU: project full POU to ring indices
  std::vector<Scalar> ring_pou(n_ring);
  for (std::size_t i = 0; i < n_ring; ++i) ring_pou[i] = pou[ring_region_to_subdomain[i]];

  // Extract the ring-local restriction of A_sub for the harmonicity constraint
  auto A_sub_ring = detail::extract_ring_submatrix<Scalar>(A_sub, ring_region_to_subdomain, ring_info);

  // Solve the constrained eigenproblem on the ring. The eigenvalues are invariant under the
  // extension below, so we keep them alongside the (soon to be extended) eigenvectors.
  std::vector<Vector> basis;
  std::vector<Scalar> eigenvalues;
  ddm::EigensolverResult info;
  {
    Logger::ScopedLog sl{ring_eigensolver_event};

    if (constraint_evp == "lagrange_multiplier") {
      // Create ring-local Dirichlet mask (only needed for the saddle-point formulation)
      std::vector<bool> ring_dirichlet_mask(n_ring);
      for (std::size_t i = 0; i < n_ring; ++i) ring_dirichlet_mask[i] = dirichlet_mask[ring_region_to_subdomain[i]];

      auto partition = detail::partition_dofs(A_neu, ring_boundary_mask, ring_dirichlet_mask);
      auto solution = detail::solve_msgfem_saddle_point_evp<Scalar>(A_neu, A_sub_ring, partition, ring_pou, eig_ptree);
      basis = std::move(solution.eigenvectors);
      eigenvalues = std::move(solution.eigenvalues);
      info = solution.info;
    }
    else if (constraint_evp == "alternating") {
      // Construct a UMFPack solver on the ring-local matrix. We need to modify the matrix by
      // eliminating the ring boundary DOFs, which are treated as Dirichlet DOFs in the projection.
      // This is done by setting the corresponding rows and columns to zero and putting ones
      // on the diagonal. We can modify the matrix in-place since it's not used for anything else.
      for (std::size_t i = 0; i < n_ring; ++i) {
        if (ring_boundary_mask[i]) {
          for (auto it = A_sub_ring[i].begin(); it != A_sub_ring[i].end(); ++it) *it = 0;
          A_sub_ring.entry(i, i) = 1;
        }
      }

      using RingMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;
      auto ring_solver = std::make_shared<Dune::UMFPack<RingMatrix>>();
      ring_solver->setOption(UMFPACK_ORDERING, UMFPACK_ORDERING_METIS);
      ring_solver->setOption(UMFPACK_IRSTEP, 0);
      ring_solver->setMatrix(A_sub_ring);

      RingMatrix C(A_neu);
      detail::scale_matrix_with_pou(C, ring_pou);
      auto solution = ddm::solve_gevp(A_neu, C, ring_solver.get(), ring_boundary_mask, eig_ptree);
      basis = std::move(solution.eigenvectors);
      eigenvalues = std::move(solution.eigenvalues);
      info = solution.info;
    }
    else DUNE_THROW(Dune::NotImplemented, "Unknown constraint_evp mode: " + constraint_evp + " (expected 'lagrange_multiplier' or 'alternating')");
  }

  // Extend the eigenvectors to the whole subdomain
  {
    Logger::ScopedLog sl{ring_extend_event};

    if (ring_extension == "from_ring") { basis = detail::extend_ring_eigenvectors<Scalar>(basis, A_sub, ring_region_to_subdomain, ring_info); }
    else if (ring_extension == "from_boundary") {
      if (!A_sub_inv) DUNE_THROW(Dune::Exception, "ring_extension = from_boundary requires A_sub_inv to be provided");

      Vector rhs(A_sub.N());
      Vector zero(A_sub.N());
      zero = 0;
      std::vector<Vector> extended_vectors(basis.size(), zero);

      for (std::size_t i = 0; i < extended_vectors.size(); ++i) {
        rhs = 0;
        for (std::size_t j = 0; j < ring_region_to_subdomain.size(); ++j)
          if (subdomain_boundary_mask[ring_region_to_subdomain[j]]) rhs[ring_region_to_subdomain[j]] = basis[i][j];

        Dune::InverseOperatorResult res;
        A_sub_inv->apply(extended_vectors[i], rhs, res);
      }

      std::swap(basis, extended_vectors);
    }
    else DUNE_THROW(Dune::NotImplemented, "Unknown ring_extension mode: " + ring_extension + " (expected 'from_ring' or 'from_boundary')");
  }

  detail::finalize_eigenvectors(basis, pou);
  return {.basis = std::move(basis), .eigenvalues = std::move(eigenvalues), .info = info};
}

// ============================================================================
//  POU (Nicolaides) coarse space — non-spectral
// ============================================================================

/**
 * @brief Build a POU coarse space from given template vectors.
 *
 * The provided template vectors are used directly as the coarse basis: each is
 * scaled by the partition of unity and normalised (see finalize_eigenvectors).
 *
 * @param template_vecs Template vectors used as the (unscaled) coarse basis.
 * @param pou Partition of unity used for scaling/normalisation.
 * @return CoarseSpaceResult holding the basis, the associated eigenvalues, and eigensolver metadata.
 */
template <class Vec>
[[nodiscard]] CoarseSpaceResult<typename Vec::field_type> build_pou_coarse_space(const std::vector<Vec>& template_vecs, const PartitionOfUnity& pou)
{
  logger::info("Setting up POU coarse space");

  std::vector<Vec> basis = template_vecs;
  detail::finalize_eigenvectors(basis, pou);
  // Non-spectral space: no eigenvalues, but the space is always successfully constructed.
  return {.basis = std::move(basis), .eigenvalues = {}, .info = {.converged = true}};
}

/**
 * @brief Build a POU coarse space spanned by the partition of unity itself.
 *
 * Produces a single basis vector that is constant one, scaled by the partition
 * of unity and normalised. This is the classic Nicolaides coarse space.
 *
 * @param pou Partition of unity used for scaling/normalisation.
 * @return CoarseSpaceResult with a single basis vector; eigenvalues is empty (non-spectral).
 */
template <class Scalar = double>
[[nodiscard]] CoarseSpaceResult<Scalar> build_pou_coarse_space(const PartitionOfUnity& pou)
{
  using Vec = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;

  logger::info("Setting up POU coarse space");

  std::vector<Vec> basis(1);
  basis[0].resize(pou.size());
  for (std::size_t i = 0; i < pou.size(); ++i) basis[0][i] = 1.0;

  detail::finalize_eigenvectors(basis, pou);
  // Non-spectral space: no eigenvalues, but the space is always successfully constructed.
  return {.basis = std::move(basis), .eigenvalues = {}, .info = {.converged = true}};
}
