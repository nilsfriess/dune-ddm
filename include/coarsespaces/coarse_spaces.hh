#pragma once

/** @file coarse_spaces.hh
    @brief Helper functions to create coarse space bases such as GenEO and MsGFEM spectral coarse spaces.
*/

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/bvector.hh>

#include <spdlog/spdlog.h>

#include "coarsespaces/energy_minimal_extension.hh"
#include "eigensolvers.hh"
#include "pou.hh"

namespace detail {

/**
 * @brief Parse eigensolver parameters from ParameterTree.
 *
 * Converts coarse space selection parameters into eigensolver-specific parameters.
 * Supports both fixed and adaptive eigenvector selection modes.
 *
 * @param subtree ParameterTree subtree containing eigensolver configuration.
 * @return ParameterTree configured for the eigensolver.
 */
inline Dune::ParameterTree parse_eigensolver_params(const Dune::ParameterTree &subtree)
{
  Dune::ParameterTree eig_ptree;
  const auto &mode_string = subtree.get("mode", "fixed");

  if (mode_string == "fixed") {
    eig_ptree["eigensolver_nev"] = std::to_string(subtree.get("n", 10));
  }
  else if (mode_string == "adaptive") {
    eig_ptree["eigensolver_nev_target"] = std::to_string(subtree.get("n_target", 10));
    eig_ptree["eigensolver_nev_max"] = std::to_string(subtree.get("n_max", 100));
    eig_ptree["eigensolver_threshold"] = std::to_string(subtree.get("threshold", 0.5));
    eig_ptree["eigensolver_keep_strict"] = "true";
  }
  else {
    DUNE_THROW(Dune::NotImplemented, "Unknown coarse space mode '" + mode_string + "', use either 'fixed' or 'adaptive'");
  }

  return eig_ptree;
}

/**
 * @brief Apply partition of unity scaling and normalize eigenvectors.
 *
 * Applies final processing to eigenvectors: scales each component by the partition
 * of unity and normalizes to unit length. This is a common final step in all
 * coarse space construction methods.
 *
 * @param eigenvectors Vector of eigenvectors to process (modified in-place).
 * @param pou Partition of unity.
 */
template <class Vec>
inline void finalize_eigenvectors(std::vector<Vec> &eigenvectors, const PartitionOfUnity &pou)
{
  for (auto &vec : eigenvectors) {
    // Apply partition of unity scaling
    for (std::size_t i = 0; i < vec.size(); ++i) {
      vec[i] *= pou[i];
    }
    // Normalize to unit length
    vec *= 1. / vec.two_norm();
  }
}

/**
 * @brief Scale matrix entries with partition of unity weights.
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
void scale_matrix_with_pou(Mat &C, const Vec &pou, const std::vector<std::size_t> &index_mapping = {})
{
  for (auto ri = C.begin(); ri != C.end(); ++ri) {
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      std::size_t i_idx = index_mapping.empty() ? ri.index() : index_mapping[ri.index()];
      std::size_t j_idx = index_mapping.empty() ? ci.index() : index_mapping[ci.index()];
      *ci *= pou[i_idx] * pou[j_idx];
    }
  }
}

template <class Mat>
inline void scale_matrix_with_pou(Mat &C, const PartitionOfUnity &pou, const std::vector<std::size_t> &index_mapping = {})
{
  for (auto ri = C.begin(); ri != C.end(); ++ri) {
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      std::size_t i_idx = index_mapping.empty() ? ri.index() : index_mapping[ri.index()];
      std::size_t j_idx = index_mapping.empty() ? ci.index() : index_mapping[ci.index()];
      *ci *= pou[i_idx] * pou[j_idx];
    }
  }
}

} // namespace detail

/**
 * @brief Builds the classical GenEO coarse space basis.
 *
 * Constructs the GenEO (Generalized Eigenproblems in the Overlaps) coarse space by solving
 * the generalized eigenproblem \f$ Ax = \lambda DBDx \f$, where \p A and \p B are matrices
 * and \p D is a diagonal matrix representing a partition of unity (passed as \p pou).
 *
 * The GenEO method selects eigenvectors corresponding to eigenvalues below a threshold,
 * which represent modes that are poorly handled by the domain decomposition preconditioner.
 *
 * The selection of eigenfunctions is controlled via parameters in the subtree
 * of \p ptree named \p ptree_prefix ("geneo" by default).
 *
 * **Parameter tree structure:**
 * - `mode`: Selection mode ("fixed" or "adaptive") (default: "fixed")
 * - For fixed mode: `n`: Number of eigenvectors to compute (default: 10)
 * - For adaptive mode:
 *   - `n_target`: Target number of eigenvectors (default: 10)
 *   - `n_max`: Maximum number of eigenvectors (default: 100)
 *   - `threshold`: Eigenvalue threshold for selection (default: 0.5)
 *
 * @param A Neumann matrix on the overlapping subdomain (left-hand side of eigenproblem).
 * @param B Neumann matrix defined in the overlap region (used to construct right-hand side).
 * @param pou Partition of unity vector (diagonal of D matrix).
 * @param ptree ParameterTree containing solver and selection parameters.
 * @param ptree_prefix Prefix for parameter subtree (default: "geneo").
 * @return Vector of normalized GenEO coarse space basis vectors.
 *
 * @throws Dune::Exception if matrix and partition of unity sizes do not match.
 * @throws Dune::NotImplemented for unknown mode in ptree.
 */
template <class Mat>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> build_geneo_coarse_space(const Mat &A, const Mat &B, const PartitionOfUnity &pou, const Dune::ParameterTree &ptree,
                                                                                      const std::string &ptree_prefix = "geneo")
{
  spdlog::info("Setting up GenEO coarse space");

  if (pou.size() != A.N()) {
    DUNE_THROW(Dune::Exception, "The matrix and the partition of unity must have the same size");
  }

  const auto &subtree = ptree.sub(ptree_prefix);

  Mat C = B; // The rhs of the eigenproblem
  detail::scale_matrix_with_pou(C, pou);

  Dune::ParameterTree eig_ptree = detail::parse_eigensolver_params(subtree);
  auto eigenvectors = solveGEVP(A, C, Eigensolver::Spectra, eig_ptree);

  detail::finalize_eigenvectors(eigenvectors, pou);
  return eigenvectors;
}

/**
 * @brief Builds the GenEO ring coarse space basis.
 *
 * Constructs a GenEO coarse space by solving the generalized eigenproblem on a ring
 * (overlap region), then extending the eigenvectors energy-minimally to the interior.
 * This is computationally cheaper than the classical GenEO method since both the
 * eigenproblem and extension are more efficient than solving on the full domain.
 *
 * The interior-to-subdomain mapping and inner ring boundary are automatically computed
 * based on the connectivity information in the Dirichlet matrix A_dir. DOFs not in the
 * ring are identified as interior DOFs, and the inner ring boundary is determined by
 * finding ring DOFs that have neighbors outside the ring.
 *
 * The selection of eigenfunctions is controlled via parameters in the subtree
 * of \p ptree named \p ptree_prefix ("geneo_ring" by default).
 *
 * **Parameter tree structure:**
 * - `mode`: Selection mode ("fixed" or "adaptive") (default: "fixed")
 * - For fixed mode: `n`: Number of eigenvectors to compute (default: 10)
 * - For adaptive mode:
 *   - `n_target`: Target number of eigenvectors (default: 10)
 *   - `n_max`: Maximum number of eigenvectors (default: 100)
 *   - `threshold`: Eigenvalue threshold for selection (default: 0.5)
 *
 * @param A_dir Dirichlet matrix for energy-minimal extension and connectivity analysis.
 * @param A Matrix for eigenproblem (typically Neumann matrix on the ring).
 * @param pou Partition of unity vector.
 * @param ring_to_subdomain Mapping from ring dofs to subdomain indices.
 * @param ptree ParameterTree containing solver and selection parameters.
 * @param ptree_prefix Prefix for parameter subtree (default: "geneo_ring").
 * @return Vector of normalized GenEO ring coarse space basis vectors.
 *
 * @throws Dune::NotImplemented for unknown mode in ptree.
 */
template <class Mat>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> build_geneo_ring_coarse_space(const Mat &A_dir, const Mat &A, const PartitionOfUnity &pou,
                                                                                           const std::vector<std::size_t> &ring_to_subdomain, const Dune::ParameterTree &ptree,
                                                                                           const std::string &ptree_prefix = "geneo_ring")
{
  spdlog::info("Setting up GenEO ring coarse space");

  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;

  const auto &subtree = ptree.sub(ptree_prefix);

  // We first create a modified partition of unity that vanishes in the interior (i.e. the region outside the "ring")
  // and on the inner boundary of the ring. We also create a interior-to-subdomain mapping.

  std::unordered_map<std::size_t, std::size_t> subdomain_to_ring;
  for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
    subdomain_to_ring[ring_to_subdomain[i]] = i;
  }

  std::vector<std::size_t> interior_to_subdomain(A_dir.N() - ring_to_subdomain.size(), 0);
  std::vector<std::size_t> inner_ring_boundary_to_subdomain;
  std::unordered_set<std::size_t> inner_ring_boundary_dofs; // For fast lookup
  inner_ring_boundary_to_subdomain.reserve(ring_to_subdomain.size());

  auto mod_pou = pou;
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < mod_pou.size(); ++i) {
    if (not subdomain_to_ring.contains(i)) { // Zero in the interior
      interior_to_subdomain[cnt++] = i;
      mod_pou[i] = 0;
    }
    else {
      for (auto ci = A_dir[i].begin(); ci != A_dir[i].end(); ++ci) {
        if (not subdomain_to_ring.contains(ci.index())) {
          // A neighbouring dof of dof i is outside the ring => dof i is on the ring boundary
          inner_ring_boundary_dofs.insert(i);
          inner_ring_boundary_to_subdomain.push_back(i);
          mod_pou[i] = 0;
          break;
        }
      }
    }
  }
  assert(cnt == interior_to_subdomain.size());

  Mat C = A; // The rhs of the eigenproblem
  detail::scale_matrix_with_pou(C, mod_pou, ring_to_subdomain);

  // Now we can solve the eigenproblem
  Dune::ParameterTree eig_ptree = detail::parse_eigensolver_params(subtree);
  auto eigenvectors = solveGEVP(A, C, Eigensolver::Spectra, eig_ptree);

  // Now we have computed a set of eigenvectors on the ring. To obtain basis vectors on the full
  // subdomain, we extend those eigenvectors energy-minimally to the interior. However, we don't
  // extend from the inner ring boundary but from one layer within the ring, as required by the
  // theory.
  // TODO: Allow to extend from the inner ring boundary to compare the effect in the numerical results.
  std::vector<std::size_t> inside_ring_boundary_to_subdomain;
  inside_ring_boundary_to_subdomain.reserve(ring_to_subdomain.size());
  for (auto i : ring_to_subdomain) {
    for (auto ci = A_dir[i].begin(); ci != A_dir[i].end(); ++ci) {
      // Check if a neighbouring dof of dof i lies on the inner ring boundary but i itself does not
      if (inner_ring_boundary_dofs.contains(ci.index()) and not inner_ring_boundary_dofs.contains(i)) {
        inside_ring_boundary_to_subdomain.push_back(i);
      }
    }
  }

  // Of course we then also have to extend the "interior" to also include the inner ring boundary
  std::vector<std::size_t> extended_interior_to_subdomain(interior_to_subdomain.size() + inner_ring_boundary_to_subdomain.size());
  cnt = 0;
  for (auto i : interior_to_subdomain) {
    extended_interior_to_subdomain[cnt++] = i;
  }
  for (auto i : inner_ring_boundary_to_subdomain) {
    extended_interior_to_subdomain[cnt++] = i;
  }

  // Next, we extend the eigenvectors energy-minimally to the rest of the domain
  EnergyMinimalExtension<Mat, Vec> ext(A_dir, extended_interior_to_subdomain, inside_ring_boundary_to_subdomain);

  // Here we create another map from 'inside ring boundary' to 'ring' to avoid too many hash map lookups below
  std::vector<std::size_t> inside_boundary_to_ring(inside_ring_boundary_to_subdomain.size());
  for (std::size_t i = 0; i < inside_ring_boundary_to_subdomain.size(); ++i) {
    inside_boundary_to_ring[i] = subdomain_to_ring[inside_ring_boundary_to_subdomain[i]];
  }

  Vec zero(A_dir.N());
  zero = 0;
  std::vector<Vec> combined_vectors(eigenvectors.size(), zero);

  Vec dirichlet_data(inside_ring_boundary_to_subdomain.size()); // Will be set each iteration
  for (std::size_t k = 0; k < eigenvectors.size(); ++k) {
    const auto &evec = eigenvectors[k];

    for (std::size_t i = 0; i < inside_boundary_to_ring.size(); ++i) {
      dirichlet_data[i] = evec[inside_boundary_to_ring[i]];
    }

    auto interior_vec = ext.extend(dirichlet_data);

    // First set the values in the ring
    for (std::size_t i = 0; i < evec.N(); ++i) {
      combined_vectors[k][ring_to_subdomain[i]] = evec[i];
    }

    // Next fill the interior values (note that the interior and ring now overlap, so this overrides some values)
    for (std::size_t i = 0; i < interior_vec.N(); ++i) {
      combined_vectors[k][extended_interior_to_subdomain[i]] = interior_vec[i];
    }
  }

  detail::finalize_eigenvectors(combined_vectors, pou);
  return combined_vectors;
}

/**
 * @brief Builds a coarse space based on Multiscale Generalized Finite Element Method (MsGFEM).
 *
 * Constructs the MsGFEM coarse space by solving a constrained generalized eigenproblem where
 * eigenvectors satisfy an A-harmonicity constraint. This is achieved by formulating a saddle
 * point system with Lagrange multipliers that enforce \f$ Au = 0 \f$ in the interior.
 *
 * The method is particularly effective for problems with high contrast coefficients where
 * traditional coarse spaces may not capture the essential multiscale behavior.
 *
 * **Mathematical formulation:**
 * The constrained eigenproblem has the form:
 * \f[
 * \begin{pmatrix} A & A^T \\ A & 0 \end{pmatrix}
 * \begin{pmatrix} x \\ \lambda \end{pmatrix} =
 * \mu \begin{pmatrix} D & 0 \\ 0 & 0 \end{pmatrix}
 * \begin{pmatrix} x \\ \lambda \end{pmatrix}
 * \f]
 * where the constraint \f$ Ax = 0 \f$ enforces A-harmonicity in the interior.
 *
 * The selection of eigenfunctions is controlled via parameters in the subtree
 * of \p ptree named \p ptree_prefix ("msgfem" by default).
 *
 * **Parameter tree structure:**
 * - `mode`: Selection mode ("fixed" or "adaptive") (default: "fixed")
 * - For fixed mode: `n`: Number of eigenvectors to compute (default: 10)
 * - For adaptive mode:
 *   - `n_target`: Target number of eigenvectors (default: 10)
 *   - `n_max`: Maximum number of eigenvectors (default: 100)
 *   - `threshold`: Eigenvalue threshold for selection (default: 0.5)
 *
 * @param A Neumann matrix on the overlapping subdomain.
 * @param pou Partition of unity vector (diagonal of D matrix).
 * @param dirichlet_mask Mask vector indicating Dirichlet boundary DOFs (>0 means Dirichlet).
 * @param subdomain_boundary_mask Mask vector indicating subdomain boundary DOFs (>0 means boundary).
 * @param ptree ParameterTree containing solver and selection parameters.
 * @param ptree_prefix Prefix for parameter subtree (default: "msgfem").
 * @return Vector of normalized MsGFEM coarse space basis vectors.
 *
 * @throws Dune::Exception if matrix and mask sizes do not match.
 * @throws Dune::NotImplemented for unknown mode in ptree.
 */
template <class Mat, class MaskVec1, class MaskVec2>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> build_msgfem_coarse_space(const Mat &A, const PartitionOfUnity &pou, const MaskVec1 &dirichlet_mask,
                                                                                       const MaskVec2 &subdomain_boundary_mask, const Dune::ParameterTree &ptree,
                                                                                       const std::string &ptree_prefix = "msgfem")
{
  if (dirichlet_mask.N() != A.N()) {
    DUNE_THROW(Dune::Exception, "The matrix and the Dirichlet mask must have the same size");
  }

  if (pou.size() != A.N()) {
    DUNE_THROW(Dune::Exception, "The matrix and the partition of unity must have the same size");
  }

  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;

  const auto &subtree = ptree.sub(ptree_prefix);

  // Partition the degrees of freedom
  enum class DOFType : std::uint8_t { Interior, Boundary, Dirichlet };
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
  spdlog::get("all_ranks")->debug("Partitioned dofs, have {} in interior, {} on subdomain boundary, {} on Dirichlet boundary", num_interior, num_boundary, num_dirichlet);

  // Create a reordered index set: first interior dofs, then boundary dofs, then Dirichlet dofs
  std::vector<std::size_t> reordering(A.N());
  std::size_t cnt_interior = 0;
  std::size_t cnt_boundary = num_interior;
  std::size_t cnt_dirichlet = num_interior + num_boundary;
  for (std::size_t i = 0; i < reordering.size(); ++i) {
    if (dof_partitioning[i] == DOFType::Interior) {
      reordering[i] = cnt_interior++;
    }
    else if (dof_partitioning[i] == DOFType::Boundary) {
      reordering[i] = cnt_boundary++;
    }
    else {
      reordering[i] = cnt_dirichlet++;
    }
  }

  // Assemble the left-hand side of the eigenproblem
  Mat A_lhs;
  const auto n_big = num_interior + num_boundary + num_interior; // size of the big eigenproblem, including the harmonicity constraint
  const auto avg = 2 * (A.nonzeroes() / A.N());
  A_lhs.setBuildMode(Mat::implicit);
  A_lhs.setImplicitBuildModeParameters(avg, 0.2);
  A_lhs.setSize(n_big, n_big);

  // Assemble the part corresponding to the a-harmonic constraint
  for (auto rit = A.begin(); rit != A.end(); ++rit) {
    auto ii = rit.index();
    auto ri = reordering[ii];
    if (dof_partitioning[ii] != DOFType::Interior) {
      continue;
    }

    for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
      auto jj = cit.index();
      auto rj = reordering[jj];

      if (dof_partitioning[jj] != DOFType::Dirichlet) {
        A_lhs.entry(rj, num_interior + num_boundary + ri) = *cit;
        A_lhs.entry(num_interior + num_boundary + ri, rj) = *cit;
      }
    }
  }

  // Assemble the remaining part of the matrix
  for (auto rit = A.begin(); rit != A.end(); ++rit) {
    auto ii = rit.index();
    auto ri = reordering[ii];
    if (dof_partitioning[ii] == DOFType::Dirichlet) { // Skip Dirchlet dofs
      continue;
    }

    for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
      auto jj = cit.index();
      auto rj = reordering[jj];

      if (dof_partitioning[jj] != DOFType::Dirichlet) {
        A_lhs.entry(ri, rj) = *cit;
      }
    }
  }
  A_lhs.compress();

  // Next, assemble the right-hand side of the eigenproblem
  Mat B;
  B.setBuildMode(Mat::implicit);
  B.setImplicitBuildModeParameters(avg, 0.2);
  B.setSize(n_big, n_big);

  for (auto rit = A.begin(); rit != A.end(); ++rit) {
    auto ii = rit.index();
    auto ri = reordering[ii];
    if (dof_partitioning[ii] != DOFType::Interior) {
      continue;
    }

    for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
      auto jj = cit.index();
      auto rj = reordering[jj];

      if (dof_partitioning[jj] == DOFType::Interior) {
        B.entry(ri, rj) = pou[ii] * pou[jj] * (*cit);
      }
    }
  }
  B.compress();

  // Now we can solve the eigenproblem
  Dune::ParameterTree eig_ptree = detail::parse_eigensolver_params(subtree);
  auto eigenvectors = solveGEVP(A_lhs, B, Eigensolver::Spectra, eig_ptree);

  // Finally, extract the actual eigenvectors
  Vec v(A.N());
  v = 0;
  std::vector<Vec> eigenvectors_actual(eigenvectors.size(), v);
  for (std::size_t k = 0; k < eigenvectors.size(); ++k) {
    for (std::size_t i = 0; i < A.N(); ++i) {
      if (dof_partitioning[i] != DOFType::Dirichlet) {
        eigenvectors_actual[k][i] = eigenvectors[k][reordering[i]];
      }
    }
  }

  detail::finalize_eigenvectors(eigenvectors_actual, pou);
  return eigenvectors_actual;
}

/**
 * @brief Builds the MsGFEM ring coarse space basis.
 *
 * Constructs a MsGFEM coarse space by solving the constrained generalized eigenproblem on a ring
 * (overlap region), then extending the eigenvectors energy-minimally to the interior.
 * Combines the A-harmonic constraint from MsGFEM with the computational efficiency of the ring approach.
 *
 * This method enforces A-harmonicity only within the ring region, making it more computationally
 * efficient than full MsGFEM while maintaining the multiscale properties needed for problems
 * with high contrast coefficients.
 *
 * The selection of eigenfunctions is controlled via parameters in the subtree
 * of \p ptree named \p ptree_prefix ("msgfem_ring" by default).
 *
 * **Parameter tree structure:**
 * - `mode`: Selection mode ("fixed" or "adaptive") (default: "fixed")
 * - For fixed mode: `n`: Number of eigenvectors to compute (default: 10)
 * - For adaptive mode:
 *   - `n_target`: Target number of eigenvectors (default: 10)
 *   - `n_max`: Maximum number of eigenvectors (default: 100)
 *   - `threshold`: Eigenvalue threshold for selection (default: 0.5)
 *
 * @param A Matrix for the eigenproblem (typically the Neumann matrix on the extended overlap region).
 * @param pou Partition of unity vector.
 * @param dirichlet_mask Mask vector indicating Dirichlet boundary DOFs (>0 means Dirichlet).
 * @param subdomain_boundary_mask Mask vector indicating subdomain boundary DOFs (>0 means boundary).
 * @param ring_to_subdomain Mapping from ring dofs to subdomain indices.
 * @param ptree ParameterTree containing solver and selection parameters.
 * @param ptree_prefix Prefix for parameter subtree (default: "msgfem_ring").
 * @return Vector of normalized MsGFEM ring coarse space basis vectors.
 *
 * @throws Dune::Exception if matrix and mask sizes do not match.
 * @throws Dune::NotImplemented for unknown mode in ptree.
 */
template <class Mat, class MaskVec1, class MaskVec2>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> build_msgfem_ring_coarse_space(const Mat &A_dir, const Mat &A, int overlap, const PartitionOfUnity &pou, const MaskVec1 &dirichlet_mask,
                                                                                            const MaskVec2 &subdomain_boundary_mask, const std::vector<std::size_t> &ring_to_subdomain,
                                                                                            const Dune::ParameterTree &ptree, const std::string &ptree_prefix = "msgfem_ring")
{
  using Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>;

  spdlog::info("Setting up MsGFEM ring coarse space");

  // Handle edge case: empty ring
  if (ring_to_subdomain.empty()) {
    DUNE_THROW(Dune::Exception, "The ring to subdomain mapping is empty, cannot build MsGFEM ring coarse space");
  }

  const auto &subtree = ptree.sub(ptree_prefix);

  // Similar as in the GenEO coarse space, we start by creating a modification of the
  // partition of unity function. Here we identify the different classes of dofs via
  // their distance to the overlapping subdomain boundary, so let's compute that
  // distance first (for sufficiently many layers of dofs).
  // TODO: It would probably make sense to factor this out and use the same construction
  //       in the GenEO coarse space.
  std::vector<int> boundary_distance(A_dir.N(), std::numeric_limits<int>::max() - 1);
  for (std::size_t i = 0; i < boundary_distance.size(); ++i) {
    if (subdomain_boundary_mask[i] > 0) {
      boundary_distance[i] = 0;
    }
  }

  for (int round = 0; round < 2 * overlap + 2; ++round) {
    for (std::size_t i = 0; i < boundary_distance.size(); ++i) {
      for (auto cit = A_dir[i].begin(); cit != A_dir[i].end(); ++cit) {
        auto nb_dist_plus_one = boundary_distance[cit.index()] + 1;
        if (nb_dist_plus_one < boundary_distance[i]) {
          boundary_distance[i] = nb_dist_plus_one;
        }
      }
    }
  }

  auto ring_width = 2 * overlap - 2 * pou.get_shrink();
  auto mod_pou = pou;
  for (std::size_t i = 0; i < mod_pou.size(); ++i) {
    if (boundary_distance[i] >= pou.get_shrink() + ring_width) {
      mod_pou[i] = 0;
    }
  }

  // // Create ring-to-subdomain inverse mapping
  // std::unordered_map<std::size_t, std::size_t> subdomain_to_ring;
  // subdomain_to_ring.reserve(ring_to_subdomain.size());
  // for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
  //   subdomain_to_ring[ring_to_subdomain[i]] = i;
  // }

  // // Create sets for boundary identification
  // std::unordered_set<std::size_t> ring_dofs;
  // for (const auto &i : ring_to_subdomain) {
  //   ring_dofs.insert(i);
  // }

  // std::unordered_set<std::size_t> interior_dofs;
  // for (const auto &i : interior_to_subdomain) {
  //   interior_dofs.insert(i);
  // }

  // std::vector<std::size_t> inside_ring_boundary_to_subdomain;
  // inside_ring_boundary_to_subdomain.reserve(ring_to_subdomain.size());
  // for (const auto &idx : interior_to_subdomain) {
  //   for (auto ci = A_dir[idx].begin(); ci != A_dir[idx].end(); ++ci) {
  //     if (idx == ci.index() or interior_dofs.contains(ci.index())) {
  //       continue;
  //     }

  //     if (ring_dofs.contains(ci.index())) {
  //       inside_ring_boundary_to_subdomain.push_back(idx);
  //       continue;
  //     }
  //   }
  // }

  std::unordered_set<std::size_t> inside_ring_boundary_dofs;
  for (const auto &i : ring_to_subdomain) {
    if (boundary_distance[i] == 2 * overlap) {
      inside_ring_boundary_dofs.insert(i);
    }
  }

  // Partition DOFs in ring: Interior (ring interior), Boundary (ring boundary + inside ring boundary), Dirichlet
  enum class DOFType : std::uint8_t { Interior, Boundary, Dirichlet };
  std::vector<DOFType> dof_partitioning(ring_to_subdomain.size());
  std::size_t num_interior = 0;
  std::size_t num_boundary = 0;
  std::size_t num_dirichlet = 0;

  for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
    auto subdomain_idx = ring_to_subdomain[i];

    if (dirichlet_mask[subdomain_idx] > 0) {
      dof_partitioning[i] = DOFType::Dirichlet;
      num_dirichlet++;
    }
    else if (subdomain_boundary_mask[subdomain_idx] || inside_ring_boundary_dofs.contains(subdomain_idx)) {
      dof_partitioning[i] = DOFType::Boundary;
      num_boundary++;
    }
    else {
      dof_partitioning[i] = DOFType::Interior;
      num_interior++;
    }
  }

  spdlog::get("all_ranks")->debug("Partitioned ring dofs, have {} in interior, {} on boundary, {} on Dirichlet boundary", num_interior, num_boundary, num_dirichlet);

  // Create reordered index set: interior, then boundary, then Dirichlet
  std::vector<std::size_t> reordering(ring_to_subdomain.size());
  std::size_t cnt_interior = 0;
  std::size_t cnt_boundary = num_interior;
  std::size_t cnt_dirichlet = num_interior + num_boundary;

  for (std::size_t i = 0; i < reordering.size(); ++i) {
    if (dof_partitioning[i] == DOFType::Interior) {
      reordering[i] = cnt_interior++;
    }
    else if (dof_partitioning[i] == DOFType::Boundary) {
      reordering[i] = cnt_boundary++;
    }
    else {
      reordering[i] = cnt_dirichlet++;
    }
  }

  // Assemble left-hand side matrix (constrained system with A-harmonic constraint)
  const auto n_big = num_interior + num_boundary + num_interior; // Include Lagrange multipliers
  const auto avg = 2 * (A.nonzeroes() / A.N());
  Mat A_lhs;
  A_lhs.setBuildMode(Mat::implicit);
  A_lhs.setImplicitBuildModeParameters(avg, 0.2);
  A_lhs.setSize(n_big, n_big);

  // Assemble A-harmonic constraint: A*u = 0 for interior DOFs
  for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
    if (dof_partitioning[i] != DOFType::Interior) {
      continue;
    }

    auto ri = reordering[i];

    for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
      auto j = cit.index(); // j is also a ring index

      if (dof_partitioning[j] != DOFType::Dirichlet) {
        auto rj = reordering[j];
        // Add constraint entries: A^T on top, A on bottom
        A_lhs.entry(rj, num_interior + num_boundary + ri) = *cit;
        A_lhs.entry(num_interior + num_boundary + ri, rj) = *cit;
      }
    }
  }

  // Assemble main matrix block
  for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
    if (dof_partitioning[i] == DOFType::Dirichlet) {
      continue;
    }

    auto ri = reordering[i];

    for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
      auto j = cit.index(); // j is also a ring index

      if (dof_partitioning[j] != DOFType::Dirichlet) {
        auto rj = reordering[j];
        A_lhs.entry(ri, rj) = *cit;
      }
    }
  }
  A_lhs.compress();

  // Assemble right-hand side matrix (weighted with partition of unity)
  Mat B;
  B.setBuildMode(Mat::implicit);
  B.setImplicitBuildModeParameters(avg, 0.2);
  B.setSize(n_big, n_big);

  for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
    if (dof_partitioning[i] == DOFType::Dirichlet) {
      continue;
    }

    auto subdomain_ii = ring_to_subdomain[i];
    auto ri = reordering[i];

    for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
      auto j = cit.index(); // j is also a ring index
      auto subdomain_jj = ring_to_subdomain[j];

      if (dof_partitioning[j] != DOFType::Dirichlet) {
        auto rj = reordering[j];
        B.entry(ri, rj) = mod_pou[subdomain_ii] * mod_pou[subdomain_jj] * (*cit);
      }
    }
  }
  B.compress();

  // Solve constrained eigenproblem
  Dune::ParameterTree eig_ptree = detail::parse_eigensolver_params(subtree);
  auto eigenvectors_constrained = solveGEVP(A_lhs, B, Eigensolver::Spectra, eig_ptree);

  // Extract actual eigenvectors (first part of constrained solution)
  Vec v_ring(ring_to_subdomain.size());
  v_ring = 0;
  std::vector<Vec> eigenvectors_ring(eigenvectors_constrained.size(), v_ring);

  for (std::size_t k = 0; k < eigenvectors_constrained.size(); ++k) {
    for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
      if (dof_partitioning[i] != DOFType::Dirichlet) {
        eigenvectors_ring[k][i] = eigenvectors_constrained[k][reordering[i]];
      }
    }
  }

  // Next, we identify the region where we compute the harmonic extension
  std::vector<std::size_t> extension_interior_to_subdomain;
  std::vector<std::size_t> extension_boundary_to_subdomain;
  extension_interior_to_subdomain.reserve(A_dir.N());
  extension_boundary_to_subdomain.reserve(ring_to_subdomain.size());
  for (std::size_t i = 0; i < A_dir.N(); ++i) {
    if (boundary_distance[i] > pou.get_shrink() + ring_width - 1) {
      extension_interior_to_subdomain.push_back(i);
    }
    else if (boundary_distance[i] == pou.get_shrink() + ring_width - 1) {
      extension_boundary_to_subdomain.push_back(i);
    }
  }

  // Set up energy-minimal extension
  EnergyMinimalExtension<Mat, Vec> ext(A_dir, extension_interior_to_subdomain, extension_boundary_to_subdomain);

  // Here we create another map from 'inside ring boundary' to 'ring' to avoid too many hash map lookups below
  std::unordered_map<std::size_t, std::size_t> subdomain_to_ring;
  for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
    subdomain_to_ring[ring_to_subdomain[i]] = i;
  }

  std::vector<std::size_t> inside_boundary_to_ring(extension_boundary_to_subdomain.size());
  for (std::size_t i = 0; i < extension_boundary_to_subdomain.size(); ++i) {
    inside_boundary_to_ring[i] = subdomain_to_ring[extension_boundary_to_subdomain[i]];
  }

  Vec zero(A_dir.N());
  zero = 0;
  std::vector<Vec> combined_vectors(eigenvectors_ring.size(), zero);

  Vec dirichlet_data(extension_boundary_to_subdomain.size()); // Will be set each iteration
  for (std::size_t k = 0; k < eigenvectors_ring.size(); ++k) {
    const auto &evec = eigenvectors_ring[k];

    for (std::size_t i = 0; i < inside_boundary_to_ring.size(); ++i) {
      dirichlet_data[i] = evec[inside_boundary_to_ring[i]];
    }

    auto interior_vec = ext.extend(dirichlet_data);

    // First set the values in the ring
    for (std::size_t i = 0; i < evec.N(); ++i) {
      combined_vectors[k][ring_to_subdomain[i]] = evec[i];
    }

    // Next fill the interior values (note that the interior and ring now overlap, so this overrides some values)
    for (std::size_t i = 0; i < interior_vec.N(); ++i) {
      combined_vectors[k][extension_interior_to_subdomain[i]] = interior_vec[i];
    }
  }

  detail::finalize_eigenvectors(combined_vectors, pou);
  return combined_vectors;
}

/**
 * @brief Builds a partition of unity (POU) coarse space basis.
 *
 * Constructs a simple coarse space consisting of a single basis vector that is
 * constant 1 on each subdomain, scaled by the partition of unity and normalized.
 * This provides a basic coarse space that captures the constant mode on each subdomain.
 *
 * The POU coarse space is computationally very cheap as it requires no eigenvalue
 * computation, but may be less effective than spectral methods for problems with
 * complex coefficient variations.
 *
 * @param pou Partition of unity vector for scaling.
 * @return Vector containing a single normalized POU coarse space basis vector.
 */
inline std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> build_pou_coarse_space(const PartitionOfUnity &pou)
{
  spdlog::info("Setting up POU coarse space");

  // Create a single basis vector that is constant 1, scaled by partition of unity
  std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> basis(1);
  basis[0].resize(pou.size());

  // Initialize with constant 1
  for (std::size_t i = 0; i < pou.size(); ++i) {
    basis[0][i] = 1.0;
  }

  // Apply partition of unity scaling and normalization
  detail::finalize_eigenvectors(basis, pou);

  return basis;
}
