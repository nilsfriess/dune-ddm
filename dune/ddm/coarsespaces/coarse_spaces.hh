#pragma once

/** @file coarse_spaces.hh
    @brief Classes to create coarse space bases such as GenEO and MsGFEM spectral coarse spaces.

    This file provides a class-based interface for building various types of coarse spaces
    used in domain decomposition methods. Each coarse space type is implemented as a separate
    class that computes the basis vectors in its constructor and provides access via get_basis().
*/

#include "../logger.hh"
#include "dune/ddm/eigensolvers/spectra.hh"

#include <Eigen/Dense>
#include <dune/common/exceptions.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/timer.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/solver.hh>
#include <fstream>
#include <mpi.h>
#include <ostream>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if DUNE_DDM_HAVE_TASKFLOW
#include <taskflow/taskflow.hpp>
#endif

#include "../eigensolvers/eigensolvers.hh"
#include "../eigensolvers/umfpack.hh"
#include "../pou.hh"
#include "energy_minimal_extension.hh"

namespace detail {
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

template <class Mat>
inline void scale_matrix_with_pou(Mat& C, const PartitionOfUnity& pou, const std::vector<std::size_t>& index_mapping = {})
{
  for (auto ri = C.begin(); ri != C.end(); ++ri) {
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      std::size_t i_idx = index_mapping.empty() ? ri.index() : index_mapping[ri.index()];
      std::size_t j_idx = index_mapping.empty() ? ci.index() : index_mapping[ci.index()];
      *ci *= pou[i_idx] * pou[j_idx];
    }
  }
}

template <class Mat, class ExtendedRemoteIndices, class Vec>
std::shared_ptr<Mat> build_algebraic_neumann(const Mat& A_novlp, const Mat& A, const ExtendedRemoteIndices& extids, const Vec& dirichlet_mask)
{
  /** We use the approach of Al Daas, Jolivet, Rees (doi 10.1137/22M1469833).
      We proceed as follows:
      1. Identify our own subdomain boundary
      2. Tell our neighbours what or subdomain boundary is
      3. Prepare the corrections that we need to send to our neighbours
      4. Apply the corrections
  */
  IdentifyBoundaryDataHandle ibdh(A, extids.get_parallel_index_set());
  auto& varcomm = extids.get_overlapping_communicator();
  varcomm.forward(ibdh);

  // Create a vector that is 1 in the interior of our subdomain and 2 on the boundary.
  std::vector<int> boundary_indicator(extids.size(), 1);
  for (std::size_t i = 0; i < extids.size(); ++i)
    if (ibdh.get_boundary_mask()[i]) boundary_indicator[i] = 2;

  int rank{};
  MPI_Comm_rank(extids.get_remote_indices().communicator(), &rank);
  CopyVectorDataHandleWithRank cvdh(boundary_indicator, rank);
  varcomm.forward(cvdh);

  // Now the map cvdh.copied_vecs contains copies of our neighbours boundary masks (for each neighbour).
  // We will now use them to compute the corrections for the correct matrix entries of our neighbours.
  std::map<int, Vec> corrections_for_rank;
  for (const auto& [remoterank, remote_boundary_indicator] : cvdh.copied_vecs) {
    corrections_for_rank.insert({remoterank, Vec(extids.size())});
    corrections_for_rank[remoterank] = 0;

    for (auto ri = A_novlp.begin(); ri != A_novlp.end(); ++ri) {
      if (remote_boundary_indicator[ri.index()] == 2) {
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if (remote_boundary_indicator[ci.index()] != 0) continue;
          corrections_for_rank[remoterank][ri.index()] += std::abs(*ci);
        }
      }
    }
  }

  // Now we have to send the corrections to the corresponding ranks. First we count how much data we need to send to each rank.
  std::map<int, std::size_t> count_for_rank;
  for (const auto& [remoterank, corrections] : corrections_for_rank) {
    count_for_rank[remoterank] = 0;

    for (auto c : corrections)
      if (c != 0) count_for_rank[remoterank]++; // We only need to send non-zero data
  }

  // Let's send this info to our neighbours while we compute the actual data to send
  std::vector<MPI_Request> requests;
  requests.reserve(3 * count_for_rank.size());
  for (const auto& [remoterank, count] : count_for_rank) MPI_Isend(&count, 1, MPI_UNSIGNED_LONG, remoterank, 0, extids.get_remote_indices().communicator(), &requests.emplace_back());

  std::map<int, std::vector<std::size_t>> indices_for_rank;
  std::map<int, std::vector<double>> values_for_rank;
  Dune::GlobalLookupIndexSet glis(extids.get_parallel_index_set());
  for (const auto& [remoterank, corrections] : corrections_for_rank) {
    indices_for_rank[remoterank].resize(count_for_rank[remoterank]);
    values_for_rank[remoterank].resize(count_for_rank[remoterank]);
    int count = 0;

    for (std::size_t i = 0; i < corrections.size(); ++i) {
      const auto c = corrections[i];

      if (c != 0) {
        indices_for_rank[remoterank][count] = glis.pair(i)->global();
        values_for_rank[remoterank][count] = c;
        count++;
      }
    }

    MPI_Isend(indices_for_rank[remoterank].data(), count, MPI_UNSIGNED_LONG, remoterank, 1, extids.get_remote_indices().communicator(), &requests.emplace_back());
    MPI_Isend(values_for_rank[remoterank].data(), count, MPI_DOUBLE, remoterank, 2, extids.get_remote_indices().communicator(), &requests.emplace_back());
  }

  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

  // Now reserve enough data for the data we will receive (we'll reuse the data structures from above for that)
  for (auto& [remoterank, count] : count_for_rank) {
    MPI_Recv(&count, 1, MPI_UNSIGNED_LONG, remoterank, 0, extids.get_remote_indices().communicator(), MPI_STATUS_IGNORE);

    indices_for_rank[remoterank].resize(count);
    values_for_rank[remoterank].resize(count);
    MPI_Recv(indices_for_rank[remoterank].data(), count, MPI_UNSIGNED_LONG, remoterank, 1, extids.get_remote_indices().communicator(), MPI_STATUS_IGNORE);
    MPI_Recv(values_for_rank[remoterank].data(), count, MPI_DOUBLE, remoterank, 2, extids.get_remote_indices().communicator(), MPI_STATUS_IGNORE);
  }

  // Now we can apply all corrections
  auto Aneu = std::make_shared<Mat>(A);
  const auto& paridxs = extids.get_parallel_index_set();
  for (auto& [remoterank, count] : count_for_rank) {
    const auto& indices = indices_for_rank[remoterank];
    const auto& values = values_for_rank[remoterank];

    for (std::size_t i = 0; i < indices.size(); ++i)
      if (paridxs.exists(indices[i])) {
        auto li = paridxs[indices[i]].local();
        if (dirichlet_mask[li] == 0) // Only apply correction if not on Dirichlet boundary
          (*Aneu)[li][li] -= values[i];
      }
      else {
        DUNE_THROW(Dune::Exception, "Got an index that is not in our subdomain");
      }
  }

  return Aneu;
}

} // namespace detail

/**
 * @brief Abstract base class for coarse space builders.
 *
 * This class provides a common interface for all coarse space construction methods.
 * Derived classes implement specific algorithms (GenEO, MsGFEM, etc.) and compute
 * the basis vectors in their constructors.
 *
 * @tparam Vec Vector type for basis vectors (default: Dune::BlockVector<Dune::FieldVector<double, 1>>)
 */
template <class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class CoarseSpaceBuilder {
public:
  virtual ~CoarseSpaceBuilder() = default;

  // Disable copy/move operations for base class
  CoarseSpaceBuilder(const CoarseSpaceBuilder&) = delete;
  CoarseSpaceBuilder& operator=(const CoarseSpaceBuilder&) = delete;
  CoarseSpaceBuilder(CoarseSpaceBuilder&&) = delete;
  CoarseSpaceBuilder& operator=(CoarseSpaceBuilder&&) = delete;

  /**
   * @brief Get the computed coarse space basis vectors.
   * @return Const reference to vector of basis vectors.
   */
  virtual const std::vector<Vec>& get_basis() const { return basis_; }

  /**
   * @brief Get the number of basis vectors.
   * @return Size of the coarse space.
   */
  virtual std::size_t size() const { return basis_.size(); }

#if DUNE_DDM_HAVE_TASKFLOW
  virtual tf::Task& get_setup_task() { return setup_task; }
#endif

protected:
  CoarseSpaceBuilder() = default;

  /// Storage for computed basis vectors
  std::vector<Vec> basis_;

#if DUNE_DDM_HAVE_TASKFLOW
  /// Setup taskflow task
  tf::Task setup_task;
#endif
};

/**
 * @brief GenEO (Generalized Eigenproblems in the Overlaps) coarse space builder.
 *
 * Constructs the classical GenEO coarse space by solving the generalized eigenproblem
 * \f$ Ax = \lambda DBDx \f$, where A and B are matrices and D is a diagonal matrix
 * representing a partition of unity.
 *
 * @tparam Mat Matrix type
 * @tparam Vec Vector type for basis vectors
 */
template <class Mat, class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class GenEOCoarseSpace : public CoarseSpaceBuilder<Vec> {
  template <class M, class V>
  friend class AlgebraicGenEOCoarseSpace;

public:
#if DUNE_DDM_HAVE_TASKFLOW
  /**
   * @brief Construct GenEO coarse space.
   *
   * @param A Neumann matrix on the overlapping subdomain (left-hand side of eigenproblem).
   * @param B Neumann matrix defined in the overlap region (used to construct right-hand side).
   * @param pou Partition of unity vector (diagonal of D matrix).
   * @param ptree ParameterTree containing solver and selection parameters.
   * @param taskflow Taskflow instance for parallel execution.
   * @param ptree_prefix Prefix for parameter subtree (default: "geneo").
   */
  // Note: We intentionally pass shared_ptrs by value to capture them safely in the taskflow lambda
  GenEOCoarseSpace(std::shared_ptr<const Mat> A, std::shared_ptr<const Mat> B, std::shared_ptr<const PartitionOfUnity> pou, const Dune::ParameterTree& ptree, tf::Taskflow& taskflow,
                   const std::string& ptree_prefix = "geneo")
  {
    const auto& subtree = ptree.sub(ptree_prefix);
    Dune::ParameterTree eig_ptree = subtree.sub("eigensolver");

    this->setup_task = taskflow.emplace([A, B, pou, eig_ptree, this] { setup_geneo_impl(A, B, pou, eig_ptree); }).name("GenEO coarse space setup");
  }

  /**
   * @brief Alternative constructor that defers task creation to the caller.
   *
   * This allows external code to create the task and call setup_geneo_impl() directly,
   * enabling composition with other tasks (e.g., building Neumann matrices first).
   */
  GenEOCoarseSpace() = default;

  /**
   * @brief Create the setup task manually. Use this when you need to add dependencies.
   *
   * @return A lambda that can be used with taskflow.emplace() or subflow.emplace()
   */
  template <class TaskflowOrSubflow>
  auto create_setup_task(TaskflowOrSubflow& tf, std::shared_ptr<const Mat> A, std::shared_ptr<const Mat> B, std::shared_ptr<const PartitionOfUnity> pou,
                         const Dune::ParameterTree& eig_ptree) -> tf::Task
  {
    return tf.emplace([A, B, pou, eig_ptree, this] { setup_geneo_impl(A, B, pou, eig_ptree); }).name("GenEO coarse space setup");
  }

protected:
  /**
   * @brief Core implementation of GenEO setup - can be called from any task context.
   */
  void setup_geneo_impl(std::shared_ptr<const Mat> A, std::shared_ptr<const Mat> B, std::shared_ptr<const PartitionOfUnity> pou, const Dune::ParameterTree& eig_ptree)
  {
    logger::info("Setting up GenEO coarse space");

    if (pou->size() != A->N()) DUNE_THROW(Dune::Exception, "The matrix and the partition of unity must have the same size");

    auto C = std::make_shared<Mat>(*B); // The rhs of the eigenproblem
    detail::scale_matrix_with_pou(*C, *pou);

    this->basis_ = solve_gevp(A, C, eig_ptree);

    detail::finalize_eigenvectors(this->basis_, *pou);
  }
#endif
};

#if 0
/**
 * @brief GenEO (Generalized Eigenproblems in the Overlaps) coarse space builder.
 *
 * Constructs the classical GenEO coarse space by solving the generalized eigenproblem
 * \f$ Ax = \lambda DBDx \f$, where A and B are matrices and D is a diagonal matrix
 * representing a partition of unity.
 *
 * @tparam Mat Matrix type
 * @tparam Vec Vector type for basis vectors
 */
template <class Mat, class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class AlgebraicGenEOCoarseSpace : public CoarseSpaceBuilder<Vec> {
public:
#if DUNE_DDM_HAVE_TASKFLOW
  /**
   * @brief Construct GenEO coarse space.
   *
   * @param A Dirichlet matrix on the overlapping subdomain (left-hand side of eigenproblem).
   * @param pou Partition of unity vector (diagonal of D matrix).
   * @param dirichlet_mask Mask vector indicating Dirichlet boundary DOFs (>0 means Dirichlet).
   * @param ptree ParameterTree containing solver and selection parameters.
   * @param taskflow Taskflow instance for parallel execution.
   * @param ptree_prefix Prefix for parameter subtree (default: "geneo").
   */
  // Note: We intentionally pass shared_ptrs by value to capture them safely in the taskflow lambda
  template <class ExtendedRemoteIndices, class DirichletMask>
  AlgebraicGenEOCoarseSpace(std::shared_ptr<const Mat> A_novlp, std::shared_ptr<const Mat> A, std::shared_ptr<const PartitionOfUnity> pou, const DirichletMask& dirichlet_mask,
                            const ExtendedRemoteIndices& extids, const Dune::ParameterTree& ptree, tf::Taskflow& taskflow, const std::string& ptree_prefix = "algebraic_geneo")
  {
    const auto& subtree = ptree.sub(ptree_prefix);
    Dune::ParameterTree eig_ptree = subtree.sub("eigensolver");

    // Use a subflow to handle the sequential dependencies
    this->setup_task = taskflow
                           .emplace([A_novlp, A, pou, &extids, &dirichlet_mask, eig_ptree, this](tf::Subflow& subflow) {
                             // Task 1: Build the Neumann matrix
                             auto build_neumann_task =
                                 subflow.emplace([A_novlp, A, &extids, &dirichlet_mask, this] { Aneu_matrix = detail::build_algebraic_neumann(*A_novlp, *A, extids, dirichlet_mask); })
                                     .name("Build algebraic Neumann matrix");

                             // Task 2: Run GenEO setup with the Neumann matrix
                             auto geneo_setup_task = subflow.emplace([A, pou, eig_ptree, this] { geneo.setup_geneo_impl(Aneu_matrix, A, pou, eig_ptree); }).name("GenEO eigenproblem solve");
                             geneo_setup_task.succeed(build_neumann_task);

                             // Task 3: Copy basis vectors
                             auto copy_basis_task = subflow.emplace([this] { this->basis_ = std::move(geneo.basis_); }).name("Copy GenEO basis");
                             copy_basis_task.succeed(geneo_setup_task);
                           })
                           .name("Algebraic GenEO coarse space setup");
  }
#endif

private:
  std::shared_ptr<Mat> Aneu_matrix;
  GenEOCoarseSpace<Mat, Vec> geneo;
};
#endif

template <class Mat, class MaskVec, class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class ConstraintGenEOCoarseSpace : public CoarseSpaceBuilder<Vec> {
public:
#if DUNE_DDM_HAVE_TASKFLOW
  /**
   * @brief Construct GenEO coarse space.
   *
   * @param A Neumann matrix on the overlapping subdomain (left-hand side of eigenproblem).
   * @param B Neumann matrix defined in the overlap region (used to construct right-hand side).
   * @param pou Partition of unity vector (diagonal of D matrix).
   * @param ptree ParameterTree containing solver and selection parameters.
   * @param taskflow Taskflow instance for parallel execution.
   * @param ptree_prefix Prefix for parameter subtree (default: "geneo").
   */
  // Note: We intentionally pass shared_ptrs by value to capture them safely in the taskflow lambda
  // TODO: Pass the references as shared_ptrs as well.
  ConstraintGenEOCoarseSpace(std::shared_ptr<const Mat> A_dir, std::shared_ptr<const Mat> A, std::shared_ptr<const Mat> B, std::shared_ptr<const PartitionOfUnity> pou,
                             const MaskVec& subdomain_boundary, const Dune::ParameterTree& ptree, tf::Taskflow& taskflow, const std::string& ptree_prefix = "constraint_geneo")
  {
    const auto& subtree = ptree.sub(ptree_prefix);
    Dune::ParameterTree eig_ptree = subtree.sub("eigensolver");

    this->setup_task = taskflow
                           .emplace([A_dir, A, B, pou, eig_ptree, &subdomain_boundary, this] {
                             logger::info("Setting up GenEO coarse space with manual constraint");
                             if (pou->size() != A->N()) DUNE_THROW(Dune::Exception, "The matrix and the partition of unity must have the same size");

                             auto C = std::make_shared<Mat>(*B); // The rhs of the eigenproblem
                             detail::scale_matrix_with_pou(*C, *pou);

                             // Assemble the matrix A_ii corresponding to interior dofs
                             std::unordered_map<std::size_t, std::size_t> subdomain_to_interior;
                             subdomain_to_interior.reserve(subdomain_boundary.size());
                             std::size_t cnt = 0;
                             for (std::size_t i = 0; i < subdomain_boundary.size(); ++i)
                               if (subdomain_boundary[i] == 0) subdomain_to_interior[i] = cnt++;

                             // Extract the interior-interior block of the matrix A_ii
                             const auto N = subdomain_to_interior.size();
                             auto interior_matrix = std::make_shared<Mat>();

                             auto avg = A_dir->nonzeroes() / A_dir->N() + 2;
                             interior_matrix->setBuildMode(Mat::implicit);
                             interior_matrix->setImplicitBuildModeParameters(avg, 0.2);
                             interior_matrix->setSize(N, N);
                             for (auto ri = A_dir->begin(); ri != A_dir->end(); ++ri) {
                               for (auto ci = ri->begin(); ci != ri->end(); ++ci)
                                 if (subdomain_to_interior.count(ri.index()) > 0 and subdomain_to_interior.count(ci.index()) > 0)
                                   interior_matrix->entry(subdomain_to_interior[ri.index()], subdomain_to_interior[ci.index()]) = *ci;
                             }
                             interior_matrix->compress();

                             UMFPackMultivecSolver interior_solver(*interior_matrix, 0); // 0 means no iterative refinement

                             const auto solve_constraint = [&](auto& X, std::size_t block) {
                               using BMV = std::remove_cvref_t<decltype(X)>;

                               // Compute the interior right-hand side
                               BMV W(X.rows(), BMV::blocksize);
                               BMV AW(X.rows(), BMV::blocksize);

                               // Copy X to W at the boundary indices
                               W.set_zero();
                               auto Wb = W.block_view(0);
                               auto Xb = X.block_view(block);
                               for (std::size_t i = 0; i < subdomain_boundary.size(); ++i)
                                 if (subdomain_boundary[i] != 0)
                                   for (std::size_t j = 0; j < BMV::blocksize; ++j) Wb(i, j) = Xb(i, j);

                               // Compute AW = A*W
                               auto AWb = AW.block_view(0);
                               Wb.apply_to_mat(*A_dir, AWb);

                               // Extract the interior values
                               BMV Xi(subdomain_to_interior.size(), BMV::blocksize);
                               auto Xib = Xi.block_view(0);
                               for (std::size_t i = 0; i < subdomain_boundary.size(); ++i)
                                 if (subdomain_boundary[i] == 0)
                                   for (std::size_t j = 0; j < BMV::blocksize; ++j) Xib(subdomain_to_interior[i], j) = AWb(i, j);

                               // Solve in the interior
                               interior_solver(Xi, 0);

                               // Copy the results back into X
                               for (std::size_t i = 0; i < subdomain_boundary.size(); ++i)
                                 if (subdomain_boundary[i] == 0)
                                   for (std::size_t j = 0; j < BMV::blocksize; ++j) Xb(i, j) = -Xib(subdomain_to_interior[i], j);
                             };

                             this->basis_ = solve_gevp(A, C, solve_constraint, eig_ptree);

                             detail::finalize_eigenvectors(this->basis_, *pou);
                           })
                           .name("GenEO coarse space setup");
  }
#endif
};

/**
 * @brief GenEO ring coarse space builder.
 *
 * Constructs a GenEO coarse space by solving the generalized eigenproblem on a ring
 * (overlap region), then extending the eigenvectors energy-minimally to the interior.
 * This is computationally cheaper than the classical GenEO method.
 *
 * @tparam Mat Matrix type
 * @tparam Vec Vector type for basis vectors
 */
template <class Mat, class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class GenEORingCoarseSpace : public CoarseSpaceBuilder<Vec> {
public:
#if DUNE_DDM_HAVE_TASKFLOW
  /**
   * @brief Construct GenEO ring coarse space.
   *
   * @param A_dir Dirichlet matrix for energy-minimal extension and connectivity analysis.
   * @param A Matrix for eigenproblem (typically Neumann matrix on the ring).
   * @param pou Partition of unity vector.
   * @param ring_to_subdomain Mapping from ring dofs to subdomain indices.
   * @param ptree ParameterTree containing solver and selection parameters.
   * @param ptree_prefix Prefix for parameter subtree (default: "geneo_ring").
   */
  GenEORingCoarseSpace(std::shared_ptr<const Mat> A_dir, std::shared_ptr<const Mat> A, std::shared_ptr<const PartitionOfUnity> pou, const std::vector<std::size_t>& ring_to_subdomain,
                       const Dune::ParameterTree& ptree, tf::Taskflow& taskflow, const std::string& ptree_prefix = "geneo_ring")
  {
    const auto& subtree = ptree.sub(ptree_prefix);
    Dune::ParameterTree eig_ptree = subtree.sub("eigensolver");

    this->setup_task = taskflow
                           .emplace([A_dir, A, pou, ring_to_subdomain, eig_ptree, this](tf::Subflow& subflow) {
                             logger::info("Setting up GenEO ring coarse space");

                             auto setup_ring_data_task = subflow
                                                             .emplace([&]() {
                                                               // We first create a modified partition of unity that vanishes in the interior (i.e. the region outside the "ring")
                                                               // and on the inner boundary of the ring. We also create a interior-to-subdomain mapping.

                                                               subdomain_to_ring.clear();
                                                               for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) subdomain_to_ring[ring_to_subdomain[i]] = i;

                                                               interior_to_subdomain.resize(A_dir->N() - ring_to_subdomain.size());
                                                               inner_ring_boundary_to_subdomain.clear();
                                                               inner_ring_boundary_dofs.clear(); // For fast lookup
                                                               inner_ring_boundary_to_subdomain.reserve(ring_to_subdomain.size());

                                                               mod_pou = std::make_unique<PartitionOfUnity>(*pou);
                                                               std::size_t cnt = 0;
                                                               for (std::size_t i = 0; i < mod_pou->size(); ++i) {
                                                                 if (not subdomain_to_ring.contains(i)) { // Zero in the interior
                                                                   interior_to_subdomain[cnt++] = i;
                                                                   (*mod_pou)[i] = 0;
                                                                 }
                                                                 else {
                                                                   for (auto ci = (*A_dir)[i].begin(); ci != (*A_dir)[i].end(); ++ci) {
                                                                     if (not subdomain_to_ring.contains(ci.index())) {
                                                                       // A neighbouring dof of dof i is outside the ring => dof i is on the ring boundary
                                                                       inner_ring_boundary_dofs.insert(i);
                                                                       inner_ring_boundary_to_subdomain.push_back(i);
                                                                       (*mod_pou)[i] = 0;
                                                                       break;
                                                                     }
                                                                   }
                                                                 }
                                                               }
                                                               assert(cnt == interior_to_subdomain.size());
                                                             })
                                                             .name("Setup ring data structures");

                             auto solve_eigenproblem_task = subflow
                                                                .emplace([&]() {
                                                                  auto C = std::make_shared<Mat>(*A); // The rhs of the eigenproblem
                                                                  detail::scale_matrix_with_pou(*C, *mod_pou, ring_to_subdomain);

                                                                  // Now we can solve the eigenproblem
                                                                  eigenvectors_ring = solve_gevp(A, C, eig_ptree);
                                                                })
                                                                .name("Solve ring eigenproblem");

                             auto setup_harmonic_extension_task =
                                 subflow
                                     .emplace([&]() {
                                       // Now we have computed a set of eigenvectors on the ring. To obtain basis vectors on the full
                                       // subdomain, we extend those eigenvectors energy-minimally to the interior. However, we don't
                                       // extend from the inner ring boundary but from one layer within the ring, as required by the
                                       // theory.
                                       // TODO: Allow to extend from the inner ring boundary to compare the effect in the numerical results.
                                       inside_ring_boundary_to_subdomain.clear();
                                       inside_ring_boundary_to_subdomain.reserve(ring_to_subdomain.size());
                                       for (auto i : ring_to_subdomain) {
                                         for (auto ci = (*A_dir)[i].begin(); ci != (*A_dir)[i].end(); ++ci) {
                                           // Check if a neighbouring dof of dof i lies on the inner ring boundary but i itself does not
                                           if (inner_ring_boundary_dofs.contains(ci.index()) and not inner_ring_boundary_dofs.contains(i)) inside_ring_boundary_to_subdomain.push_back(i);
                                         }
                                       }

                                       // Of course we then also have to extend the "interior" to also include the inner ring boundary
                                       extended_interior_to_subdomain.resize(interior_to_subdomain.size() + inner_ring_boundary_to_subdomain.size());
                                       std::size_t cnt = 0;
                                       for (auto i : interior_to_subdomain) extended_interior_to_subdomain[cnt++] = i;
                                       for (auto i : inner_ring_boundary_to_subdomain) extended_interior_to_subdomain[cnt++] = i;

                                       // Set up energy-minimal extension
                                       ext = std::make_unique<EnergyMinimalExtension<Mat, Vec>>(*A_dir, extended_interior_to_subdomain, inside_ring_boundary_to_subdomain);
                                     })
                                     .name("Setup harmonic extension");

                             auto compute_harmonic_extension_task = //
                                 subflow
                                     .emplace([&]() {
                                       // Here we create another map from 'inside ring boundary' to 'ring' to avoid too many hash map lookups below
                                       std::vector<std::size_t> inside_boundary_to_ring(inside_ring_boundary_to_subdomain.size());
                                       for (std::size_t i = 0; i < inside_ring_boundary_to_subdomain.size(); ++i) inside_boundary_to_ring[i] = subdomain_to_ring[inside_ring_boundary_to_subdomain[i]];

                                       Vec zero(A_dir->N());
                                       zero = 0;
                                       std::vector<Vec> combined_vectors(eigenvectors_ring.size(), zero);

                                       Vec dirichlet_data(inside_ring_boundary_to_subdomain.size()); // Will be set each iteration
                                       for (std::size_t k = 0; k < eigenvectors_ring.size(); ++k) {
                                         const auto& evec = eigenvectors_ring[k];

                                         for (std::size_t i = 0; i < inside_boundary_to_ring.size(); ++i) dirichlet_data[i] = evec[inside_boundary_to_ring[i]];

                                         auto interior_vec = ext->extend(dirichlet_data);

                                         // First set the values in the ring
                                         for (std::size_t i = 0; i < evec.N(); ++i) combined_vectors[k][ring_to_subdomain[i]] = evec[i];

                                         // Next fill the interior values (note that the interior and ring now overlap, so this overrides some values)
                                         for (std::size_t i = 0; i < interior_vec.N(); ++i) combined_vectors[k][extended_interior_to_subdomain[i]] = interior_vec[i];
                                       }

                                       this->basis_ = std::move(combined_vectors);
                                       detail::finalize_eigenvectors(this->basis_, *pou);
                                     })
                                     .name("Compute harmonic extension");

                             setup_ring_data_task.precede(solve_eigenproblem_task, setup_harmonic_extension_task);
                             compute_harmonic_extension_task.succeed(solve_eigenproblem_task, setup_harmonic_extension_task);
                           })
                           .name("GenEO ring coarse space setup");
  }
#endif

private:
  std::unordered_map<std::size_t, std::size_t> subdomain_to_ring;
  std::vector<std::size_t> interior_to_subdomain;
  std::vector<std::size_t> inner_ring_boundary_to_subdomain;
  std::unordered_set<std::size_t> inner_ring_boundary_dofs;
  std::unique_ptr<PartitionOfUnity> mod_pou;
  std::vector<std::size_t> inside_ring_boundary_to_subdomain;
  std::vector<std::size_t> extended_interior_to_subdomain;
  std::vector<Vec> eigenvectors_ring;
  std::unique_ptr<EnergyMinimalExtension<Mat, Vec>> ext;
};

#if DUNE_DDM_HAVE_TASKFLOW
/**
 * @brief MsGFEM (Multiscale Generalized Finite Element Method) coarse space builder.
 *
 * Constructs the MsGFEM coarse space by solving a constrained generalized eigenproblem where
 * eigenvectors satisfy an A-harmonicity constraint. This is achieved by formulating a saddle
 * point system with Lagrange multipliers that enforce Au = 0 in the interior.
 *
 * @tparam Mat Matrix type
 * @tparam MaskVec1 Type for Dirichlet mask vector
 * @tparam MaskVec2 Type for subdomain boundary mask vector
 * @tparam Vec Vector type for basis vectors
 */
template <class Mat, class MaskVec1, class MaskVec2, class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class MsGFEMCoarseSpace : public CoarseSpaceBuilder<Vec> {
  template <class M, class MV1, class MV2, class V>
  friend class AlgebraicMsGFEMCoarseSpace;

public:
  /**
   * @brief Construct MsGFEM coarse space.
   *
   * @param A Neumann matrix on the overlapping subdomain.
   * @param pou Partition of unity vector (diagonal of D matrix).
   * @param dirichlet_mask Mask vector indicating Dirichlet boundary DOFs (>0 means Dirichlet).
   * @param subdomain_boundary_mask Mask vector indicating subdomain boundary DOFs (>0 means boundary).
   * @param ptree ParameterTree containing solver and selection parameters.
   * @param taskflow Taskflow instance for parallel execution.
   * @param ptree_prefix Prefix for parameter subtree (default: "msgfem").
   */
  // Note: We intentionally pass shared_ptrs by value to capture them safely in the taskflow lambda
  MsGFEMCoarseSpace(std::shared_ptr<const Mat> A, std::shared_ptr<const PartitionOfUnity> pou, const MaskVec1& dirichlet_mask, const MaskVec2& subdomain_boundary_mask,
                    const Dune::ParameterTree& ptree, tf::Taskflow& taskflow, const std::string& ptree_prefix = "msgfem")
  {
    const auto& subtree = ptree.sub(ptree_prefix);
    Dune::ParameterTree eig_ptree = subtree.sub("eigensolver");

    this->setup_task = taskflow.emplace([A, pou, &dirichlet_mask, &subdomain_boundary_mask, eig_ptree, this] { setup_msgfem_impl(A, A, pou, dirichlet_mask, subdomain_boundary_mask, eig_ptree); })
                           .name("MsGFEM coarse space setup");
  }

  MsGFEMCoarseSpace(std::shared_ptr<const Mat> A_neu, std::shared_ptr<const Mat> A_dir, std::shared_ptr<const PartitionOfUnity> pou, const MaskVec1& dirichlet_mask,
                    const MaskVec2& subdomain_boundary_mask, const Dune::ParameterTree& ptree, tf::Taskflow& taskflow, const std::string& ptree_prefix = "msgfem")
  {
    const auto& subtree = ptree.sub(ptree_prefix);
    Dune::ParameterTree eig_ptree = subtree.sub("eigensolver");

    this->setup_task =
        taskflow.emplace([A_neu, A_dir, pou, &dirichlet_mask, &subdomain_boundary_mask, eig_ptree, this] { setup_msgfem_impl(A_neu, A_dir, pou, dirichlet_mask, subdomain_boundary_mask, eig_ptree); })
            .name("MsGFEM coarse space setup");
  }

  /**
   * @brief Default constructor for deferred task creation.
   */
  MsGFEMCoarseSpace() = default;

protected:
  /**
   * @brief Core implementation of MsGFEM setup - can be called from any task context.
   */
  template <class MV1, class MV2>
  void setup_msgfem_impl(std::shared_ptr<const Mat> A_neu, std::shared_ptr<const Mat> A_dir, std::shared_ptr<const PartitionOfUnity> pou, const MV1& dirichlet_mask, const MV2& subdomain_boundary_mask,
                         const Dune::ParameterTree& eig_ptree)
  {
    logger::info("Setting up MsGFEM coarse space");

    if (A_dir->N() != A_neu->N()) DUNE_THROW(Dune::Exception, "The two matrices must have the same size");

    if (dirichlet_mask.N() != A_dir->N()) DUNE_THROW(Dune::Exception, "The matrix and the Dirichlet mask must have the same size");

    if (pou->size() != A_dir->N()) DUNE_THROW(Dune::Exception, "The matrix and the partition of unity must have the same size");

    // Partition the degrees of freedom
    enum class DOFType : std::uint8_t { Interior, Boundary, Dirichlet };
    std::vector<DOFType> dof_partitioning(A_dir->N());
    std::size_t num_interior = 0;
    std::size_t num_boundary = 0;
    std::size_t num_dirichlet = 0;
    for (std::size_t i = 0; i < A_dir->N(); ++i) {
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
    logger::debug_all("Partitioned dofs, have {} in interior, {} on subdomain boundary, {} on Dirichlet boundary", num_interior, num_boundary, num_dirichlet);

    // Create a reordered index set: first interior dofs, then boundary dofs, then Dirichlet dofs
    std::vector<std::size_t> reordering(A_dir->N());
    std::size_t cnt_interior = 0;
    std::size_t cnt_boundary = num_interior;
    std::size_t cnt_dirichlet = num_interior + num_boundary;
    for (std::size_t i = 0; i < reordering.size(); ++i)
      if (dof_partitioning[i] == DOFType::Interior) reordering[i] = cnt_interior++;
      else if (dof_partitioning[i] == DOFType::Boundary) reordering[i] = cnt_boundary++;
      else reordering[i] = cnt_dirichlet++;

    // Assemble the left-hand side of the eigenproblem
    auto A_lhs = std::make_shared<Mat>();
    const auto n_big = num_interior + num_boundary + num_interior; // size of the big eigenproblem, including the harmonicity constraint
    const auto avg = 2 * (A_dir->nonzeroes() / A_dir->N());
    A_lhs->setBuildMode(Mat::implicit);
    A_lhs->setImplicitBuildModeParameters(avg, 0.2);
    A_lhs->setSize(n_big, n_big);

    // Assemble the part corresponding to the a-harmonic constraint
    for (auto rit = A_dir->begin(); rit != A_dir->end(); ++rit) {
      auto ii = rit.index();
      auto ri = reordering[ii];
      if (dof_partitioning[ii] != DOFType::Interior) continue;

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto jj = cit.index();
        auto rj = reordering[jj];

        if (dof_partitioning[jj] != DOFType::Dirichlet) {
          A_lhs->entry(rj, num_interior + num_boundary + ri) = *cit;
          A_lhs->entry(num_interior + num_boundary + ri, rj) = *cit;
        }
      }
    }

    // Assemble the remaining part of the matrix
    for (auto rit = A_neu->begin(); rit != A_neu->end(); ++rit) {
      auto ii = rit.index();
      auto ri = reordering[ii];
      if (dof_partitioning[ii] == DOFType::Dirichlet) // Skip Dirchlet dofs
        continue;

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto jj = cit.index();
        auto rj = reordering[jj];

        if (dof_partitioning[jj] != DOFType::Dirichlet) A_lhs->entry(ri, rj) = *cit;
      }
    }
    A_lhs->compress();

    // Next, assemble the right-hand side of the eigenproblem
    auto B = std::make_shared<Mat>();
    B->setBuildMode(Mat::implicit);
    B->setImplicitBuildModeParameters(avg, 0.2);
    B->setSize(n_big, n_big);

    for (auto rit = A_neu->begin(); rit != A_neu->end(); ++rit) {
      auto ii = rit.index();
      auto ri = reordering[ii];
      if (dof_partitioning[ii] != DOFType::Interior) continue;

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto jj = cit.index();
        auto rj = reordering[jj];

        if (dof_partitioning[jj] == DOFType::Interior) B->entry(ri, rj) = (*pou)[ii] * (*pou)[jj] * (*cit);
      }
    }
    B->compress();

    // Now we can solve the eigenproblem
    auto eigenvectors = solve_gevp(A_lhs, B, eig_ptree);

    // Finally, extract the actual eigenvectors
    Vec v(A_dir->N());
    v = 0;
    std::vector<Vec> eigenvectors_actual(eigenvectors.size(), v);
    for (std::size_t k = 0; k < eigenvectors.size(); ++k) {
      for (std::size_t i = 0; i < A_dir->N(); ++i)
        if (dof_partitioning[i] != DOFType::Dirichlet) eigenvectors_actual[k][i] = eigenvectors[k][reordering[i]];
    }

    this->basis_ = std::move(eigenvectors_actual);
    detail::finalize_eigenvectors(this->basis_, *pou);
  }
};
#endif

#if 0
#if DUNE_DDM_HAVE_TASKFLOW
/**
 * @brief Algebraic MsGFEM coarse space builder.
 *
 * Constructs an algebraic variant of MsGFEM by first building an algebraic Neumann matrix
 * and then solving the MsGFEM constrained eigenproblem with it. This combines the algebraic
 * Neumann construction from AlgebraicGenEO with the A-harmonic constraint from MsGFEM.
 *
 * @tparam Mat Matrix type
 * @tparam MaskVec1 Type for Dirichlet mask vector
 * @tparam MaskVec2 Type for subdomain boundary mask vector
 * @tparam Vec Vector type for basis vectors
 */
template <class Mat, class MaskVec1, class MaskVec2, class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class AlgebraicMsGFEMCoarseSpace : public CoarseSpaceBuilder<Vec> {
public:
  /**
   * @brief Construct algebraic MsGFEM coarse space.
   *
   * @param A_novlp Non-overlapping Dirichlet matrix.
   * @param A Overlapping Dirichlet matrix.
   * @param pou Partition of unity vector.
   * @param dirichlet_mask Mask vector indicating Dirichlet boundary DOFs (>0 means Dirichlet).
   * @param subdomain_boundary_mask Mask vector indicating subdomain boundary DOFs (>0 means boundary).
   * @param extids Extended remote indices for communication.
   * @param ptree ParameterTree containing solver and selection parameters.
   * @param taskflow Taskflow instance for parallel execution.
   * @param ptree_prefix Prefix for parameter subtree (default: "algebraic_msgfem").
   */
  template <class ExtendedRemoteIndices, class DirichletMask, class BoundaryMask>
  AlgebraicMsGFEMCoarseSpace(std::shared_ptr<const Mat> A_novlp, std::shared_ptr<const Mat> A, std::shared_ptr<const PartitionOfUnity> pou, const DirichletMask& dirichlet_mask,
                             const BoundaryMask& subdomain_boundary_mask, const ExtendedRemoteIndices& extids, const Dune::ParameterTree& ptree, tf::Taskflow& taskflow,
                             const std::string& ptree_prefix = "algebraic_msgfem")
  {
    const auto& subtree = ptree.sub(ptree_prefix);
    Dune::ParameterTree eig_ptree = subtree.sub("eigensolver");

    // Use a subflow to handle the sequential dependencies
    this->setup_task =
        taskflow
            .emplace([A_novlp, A, pou, &extids, &dirichlet_mask, &subdomain_boundary_mask, eig_ptree, this](tf::Subflow& subflow) {
              // Task 1: Build the Neumann matrix
              auto build_neumann_task = subflow.emplace([A_novlp, A, &extids, &dirichlet_mask, this] { Aneu_matrix = detail::build_algebraic_neumann(*A_novlp, *A, extids, dirichlet_mask); })
                                            .name("Build algebraic Neumann matrix");

              // Task 2: Run MsGFEM setup with the Neumann matrix
              auto msgfem_setup_task =
                  subflow.emplace([pou, &dirichlet_mask, &subdomain_boundary_mask, eig_ptree, this] { msgfem.setup_msgfem_impl(Aneu_matrix, pou, dirichlet_mask, subdomain_boundary_mask, eig_ptree); })
                      .name("MsGFEM eigenproblem solve");
              msgfem_setup_task.succeed(build_neumann_task);

              // Task 3: Copy basis vectors
              auto copy_basis_task = subflow.emplace([this] { this->basis_ = std::move(msgfem.basis_); }).name("Copy MsGFEM basis");
              copy_basis_task.succeed(msgfem_setup_task);
            })
            .name("Algebraic MsGFEM coarse space setup");
  }

private:
  std::shared_ptr<Mat> Aneu_matrix;
  MsGFEMCoarseSpace<Mat, MaskVec1, MaskVec2, Vec> msgfem;
};
#endif
#endif

#if DUNE_DDM_HAVE_TASKFLOW
/**
 * @brief MsGFEM ring coarse space builder.
 *
 * Constructs a MsGFEM coarse space by solving the constrained generalized eigenproblem on a ring
 * (overlap region), then extending the eigenvectors energy-minimally to the interior.
 * Combines the A-harmonic constraint from MsGFEM with the computational efficiency of the ring approach.
 *
 * @tparam Mat Matrix type
 * @tparam MaskVec1 Type for Dirichlet mask vector
 * @tparam MaskVec2 Type for subdomain boundary mask vector
 * @tparam Vec Vector type for basis vectors
 */
template <class Mat, class MaskVec1, class MaskVec2, class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class MsGFEMRingCoarseSpace : public CoarseSpaceBuilder<Vec> {
public:
  /**
   * @brief Construct MsGFEM ring coarse space.
   *
   * @param A_dir Dirichlet matrix for energy-minimal extension.
   * @param A Matrix for the eigenproblem (typically the Neumann matrix on the extended overlap region).
   * @param overlap Overlap parameter.
   * @param pou Partition of unity vector.
   * @param dirichlet_mask Mask vector indicating Dirichlet boundary DOFs (>0 means Dirichlet).
   * @param subdomain_boundary_mask Mask vector indicating subdomain boundary DOFs (>0 means boundary).
   * @param ring_to_subdomain Mapping from ring dofs to subdomain indices.
   * @param ptree ParameterTree containing solver and selection parameters.
   * @param taskflow Taskflow instance for parallel execution.
   * @param ptree_prefix Prefix for parameter subtree (default: "msgfem_ring").
   */
  // Note: We intentionally pass shared_ptrs by value to capture them safely in the taskflow lambda
  MsGFEMRingCoarseSpace(std::shared_ptr<const Mat> A_dir, std::shared_ptr<const Mat> A, int overlap, std::shared_ptr<const PartitionOfUnity> pou, const MaskVec1& dirichlet_mask,
                        const MaskVec2& subdomain_boundary_mask, const std::vector<std::size_t>& ring_to_subdomain, const Dune::ParameterTree& ptree, tf::Taskflow& taskflow,
                        const std::string& ptree_prefix = "msgfem_ring")
  {
    const auto& subtree = ptree.sub(ptree_prefix);
    Dune::ParameterTree eig_ptree = subtree.sub("eigensolver");

    this->setup_task =
        taskflow
            .emplace([A_dir, A, overlap, pou, &dirichlet_mask, &subdomain_boundary_mask, ring_to_subdomain, eig_ptree, this](tf::Subflow& subflow) {
              logger::info("Setting up MsGFEM ring coarse space");

              auto setup_boundary_distance_task = subflow
                                                      .emplace([&]() {
                                                        // Similar as in the GenEO coarse space, we start by creating a modification of the
                                                        // pargtition of unity function. Here we identify the different classes of dofs via
                                                        // their distance to the overlapping subdomain boundary, so let's compute that
                                                        // distance first (for sufficiently many layers of dofs).
                                                        boundary_distance.resize(A_dir->N(), std::numeric_limits<int>::max() - 1);
                                                        for (std::size_t i = 0; i < boundary_distance.size(); ++i)
                                                          if (subdomain_boundary_mask[i] > 0) boundary_distance[i] = 0;

                                                        for (int round = 0; round < 2 * overlap + 2; ++round) {
                                                          for (std::size_t i = 0; i < boundary_distance.size(); ++i) {
                                                            for (auto cit = (*A_dir)[i].begin(); cit != (*A_dir)[i].end(); ++cit) {
                                                              auto nb_dist_plus_one = boundary_distance[cit.index()] + 1;
                                                              if (nb_dist_plus_one < boundary_distance[i]) boundary_distance[i] = nb_dist_plus_one;
                                                            }
                                                          }
                                                        }

                                                        ring_width = (2 * overlap) - (2 * pou->get_shrink());
                                                      })
                                                      .name("Setup ring data structures");

              auto solve_eigenproblem_task = subflow
                                                 .emplace([&]() {
                                                   // Handle edge case: empty ring
                                                   if (ring_to_subdomain.empty()) DUNE_THROW(Dune::Exception, "The ring to subdomain mapping is empty, cannot build MsGFEM ring coarse space");

                                                   auto mod_pou = *pou;
                                                   for (std::size_t i = 0; i < mod_pou.size(); ++i)
                                                     if (boundary_distance[i] >= pou->get_shrink() + ring_width) mod_pou[i] = 0;

                                                   std::unordered_set<std::size_t> inside_ring_boundary_dofs;
                                                   for (const auto& i : ring_to_subdomain)
                                                     if (boundary_distance[i] == 2 * overlap) inside_ring_boundary_dofs.insert(i);

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

                                                   logger::debug_all("Partitioned ring dofs, have {} in interior, {} on boundary, {} on Dirichlet boundary", num_interior, num_boundary, num_dirichlet);

                                                   // Create reordered index set: interior, then boundary, then Dirichlet
                                                   std::vector<std::size_t> reordering(ring_to_subdomain.size());
                                                   std::size_t cnt_interior = 0;
                                                   std::size_t cnt_boundary = num_interior;
                                                   std::size_t cnt_dirichlet = num_interior + num_boundary;

                                                   for (std::size_t i = 0; i < reordering.size(); ++i)
                                                     if (dof_partitioning[i] == DOFType::Interior) reordering[i] = cnt_interior++;
                                                     else if (dof_partitioning[i] == DOFType::Boundary) reordering[i] = cnt_boundary++;
                                                     else reordering[i] = cnt_dirichlet++;

                                                   // Assemble left-hand side matrix (constrained system with A-harmonic constraint)
                                                   const auto n_big = num_interior + num_boundary + num_interior; // Include Lagrange multipliers
                                                   const auto avg = 2 * (A->nonzeroes() / A->N());
                                                   auto A_lhs = std::make_shared<Mat>();
                                                   A_lhs->setBuildMode(Mat::implicit);
                                                   A_lhs->setImplicitBuildModeParameters(avg, 0.2);
                                                   A_lhs->setSize(n_big, n_big);

                                                   // Assemble A-harmonic constraint: A*u = 0 for interior DOFs
                                                   for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
                                                     if (dof_partitioning[i] != DOFType::Interior) continue;

                                                     auto ri = reordering[i];

                                                     for (auto cit = (*A)[i].begin(); cit != (*A)[i].end(); ++cit) {
                                                       auto j = cit.index(); // j is also a ring index

                                                       if (dof_partitioning[j] != DOFType::Dirichlet) {
                                                         auto rj = reordering[j];
                                                         // Add constraint entries: A^T on top, A on bottom
                                                         A_lhs->entry(rj, num_interior + num_boundary + ri) = *cit;
                                                         A_lhs->entry(num_interior + num_boundary + ri, rj) = *cit;
                                                       }
                                                     }
                                                   }

                                                   // Assemble main matrix block
                                                   for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
                                                     if (dof_partitioning[i] == DOFType::Dirichlet) continue;

                                                     auto ri = reordering[i];

                                                     for (auto cit = (*A)[i].begin(); cit != (*A)[i].end(); ++cit) {
                                                       auto j = cit.index(); // j is also a ring index

                                                       if (dof_partitioning[j] != DOFType::Dirichlet) {
                                                         auto rj = reordering[j];
                                                         A_lhs->entry(ri, rj) = *cit;
                                                       }
                                                     }
                                                   }
                                                   A_lhs->compress();

                                                   // Assemble right-hand side matrix (weighted with partition of unity)
                                                   auto B = std::make_shared<Mat>();
                                                   B->setBuildMode(Mat::implicit);
                                                   B->setImplicitBuildModeParameters(avg, 0.2);
                                                   B->setSize(n_big, n_big);

                                                   for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
                                                     if (dof_partitioning[i] == DOFType::Dirichlet) continue;

                                                     auto subdomain_ii = ring_to_subdomain[i];
                                                     auto ri = reordering[i];

                                                     for (auto cit = (*A)[i].begin(); cit != (*A)[i].end(); ++cit) {
                                                       auto j = cit.index(); // j is also a ring index
                                                       auto subdomain_jj = ring_to_subdomain[j];

                                                       if (dof_partitioning[j] != DOFType::Dirichlet) {
                                                         auto rj = reordering[j];
                                                         B->entry(ri, rj) = mod_pou[subdomain_ii] * mod_pou[subdomain_jj] * (*cit);
                                                       }
                                                     }
                                                   }
                                                   B->compress();

                                                   // Solve constrained eigenproblem
                                                   auto eigenvectors_constrained = solve_gevp(A_lhs, B, eig_ptree);

                                                   // Extract actual eigenvectors (first part of constrained solution)
                                                   Vec v_ring(ring_to_subdomain.size());
                                                   v_ring = 0;
                                                   eigenvectors_ring.resize(eigenvectors_constrained.size(), v_ring);

                                                   for (std::size_t k = 0; k < eigenvectors_constrained.size(); ++k) {
                                                     for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i)
                                                       if (dof_partitioning[i] != DOFType::Dirichlet) eigenvectors_ring[k][i] = eigenvectors_constrained[k][reordering[i]];
                                                   }
                                                 })
                                                 .name("Solve ring eigenproblem");

              auto setup_harmonic_extension_task = subflow
                                                       .emplace([&]() {
                                                         // Next, we identify the region where we compute the harmonic extension
                                                         extension_interior_to_subdomain.reserve(A_dir->N());
                                                         extension_boundary_to_subdomain.reserve(ring_to_subdomain.size());
                                                         for (std::size_t i = 0; i < A_dir->N(); ++i)
                                                           if (boundary_distance[i] > pou->get_shrink() + ring_width - 1) extension_interior_to_subdomain.push_back(i);
                                                           else if (boundary_distance[i] == pou->get_shrink() + ring_width - 1) extension_boundary_to_subdomain.push_back(i);

                                                         // Set up energy-minimal extension
                                                         ext = std::make_unique<EnergyMinimalExtension<Mat, Vec>>(*A_dir, extension_interior_to_subdomain, extension_boundary_to_subdomain);
                                                       })
                                                       .name("Setup harmonic extension");

              auto compute_harmonic_extension_task = subflow
                                                         .emplace([&]() {
                                                           // Now we have computed a set of eigenvectors on the ring. To obtain basis vectors on the full
                                                           // subdomain, we extend those eigenvectors energy-minimally to the interior. However, we don't
                                                           // extend from the inner ring boundary but from one layer within the ring, as required by the
                                                           // theory.
                                                           // Here we create another map from 'inside ring boundary' to 'ring' to avoid too many hash map lookups below
                                                           std::unordered_map<std::size_t, std::size_t> subdomain_to_ring;
                                                           for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) subdomain_to_ring[ring_to_subdomain[i]] = i;

                                                           std::vector<std::size_t> inside_boundary_to_ring(extension_boundary_to_subdomain.size());
                                                           for (std::size_t i = 0; i < extension_boundary_to_subdomain.size(); ++i)
                                                             inside_boundary_to_ring[i] = subdomain_to_ring[extension_boundary_to_subdomain[i]];

                                                           Vec zero(A_dir->N());
                                                           zero = 0;
                                                           std::vector<Vec> combined_vectors(eigenvectors_ring.size(), zero);

                                                           Vec dirichlet_data(extension_boundary_to_subdomain.size()); // Will be set each iteration
                                                           for (std::size_t k = 0; k < eigenvectors_ring.size(); ++k) {
                                                             const auto& evec = eigenvectors_ring[k];

                                                             for (std::size_t i = 0; i < inside_boundary_to_ring.size(); ++i) dirichlet_data[i] = evec[inside_boundary_to_ring[i]];

                                                             auto interior_vec = ext->extend(dirichlet_data);

                                                             // First set the values in the ring
                                                             for (std::size_t i = 0; i < evec.N(); ++i) combined_vectors[k][ring_to_subdomain[i]] = evec[i];

                                                             // Next fill the interior values (note that the interior and ring now overlap, so this overrides some values)
                                                             for (std::size_t i = 0; i < interior_vec.N(); ++i) combined_vectors[k][extension_interior_to_subdomain[i]] = interior_vec[i];
                                                           }

                                                           this->basis_ = std::move(combined_vectors);
                                                           detail::finalize_eigenvectors(this->basis_, *pou);
                                                         })
                                                         .name("Compute harmonic extension");

              setup_boundary_distance_task.precede(solve_eigenproblem_task, setup_harmonic_extension_task);
              compute_harmonic_extension_task.succeed(solve_eigenproblem_task, setup_harmonic_extension_task);
            })
            .name("MsGFEM ring coarse space setup");
  }

private:
  std::vector<int> boundary_distance;
  int ring_width;
  std::vector<std::size_t> extension_interior_to_subdomain;
  std::vector<std::size_t> extension_boundary_to_subdomain;
  std::vector<Vec> eigenvectors_ring;
  std::unique_ptr<EnergyMinimalExtension<Mat, Vec>> ext;
};
#endif

/**
 * @brief Partition of Unity (POU) coarse space builder.
 *
 * Constructs a simple coarse space consisting of a single basis vector that is
 * constant 1 on each subdomain, scaled by the partition of unity and normalized.
 * This provides a basic coarse space that captures the constant mode on each subdomain.
 *
 * @tparam Vec Vector type for basis vectors
 */
template <class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class POUCoarseSpace : public CoarseSpaceBuilder<Vec> {
public:
#if DUNE_DDM_HAVE_TASKFLOW
  /**
   * @brief Construct POU coarse space.
   *
   * @param pou Partition of unity vector for scaling.
   * @param taskflow Taskflow instance for parallel execution.
   */
  // Note: We intentionally pass shared_ptrs by value to capture them safely in the taskflow lambda
  explicit POUCoarseSpace(std::shared_ptr<const PartitionOfUnity> pou, tf::Taskflow& taskflow)
  {
    this->setup_task = taskflow
                           .emplace([pou, this] {
                             logger::info("Setting up POU coarse space");

                             // Create a single basis vector that is constant 1, scaled by partition of unity
                             this->basis_.resize(1);
                             this->basis_[0].resize(pou->size());

                             // Initialize with constant 1
                             for (std::size_t i = 0; i < pou->size(); ++i) this->basis_[0][i] = 1.0;

                             // Apply partition of unity scaling and normalization
                             detail::finalize_eigenvectors(this->basis_, *pou);
                           })
                           .name("POU coarse space setup");
  }
#endif

  /**
   * @brief Construct POU coarse space.
   *
   * @param pou Partition of unity vector for scaling.
   */
  explicit POUCoarseSpace(const PartitionOfUnity& pou)
  {
    logger::info("Setting up POU coarse space");

    // Create a single basis vector that is constant 1, scaled by partition of unity
    this->basis_.resize(1);
    this->basis_[0].resize(pou.size());

    // Initialize with constant 1
    for (std::size_t i = 0; i < pou.size(); ++i) this->basis_[0][i] = 1.0;

    // Apply partition of unity scaling and normalization
    detail::finalize_eigenvectors(this->basis_, pou);
  }

  POUCoarseSpace(std::vector<Vec>& template_vecs, const PartitionOfUnity& pou)
  {
    this->basis_ = template_vecs;
    detail::finalize_eigenvectors(this->basis_, pou);
  }
};

template <class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class HarmonicExtensionCoarseSpace : public CoarseSpaceBuilder<Vec> {
public:
  template <class Mat, class MaskVec>
  HarmonicExtensionCoarseSpace(std::shared_ptr<Mat> A_ovlp, std::shared_ptr<PartitionOfUnity> pou, std::shared_ptr<std::vector<Vec>> boundary_data, const MaskVec& subdomain_boundary_mask,
                               tf::Taskflow& taskflow)
  {
    this->setup_task = taskflow.emplace([&subdomain_boundary_mask, boundary_data, A_ovlp, pou, this]() {
      logger::info("Setting up coarse space with energy-minimal extension");

      std::vector<std::size_t> interior_to_subdomain;
      std::vector<std::size_t> boundary_to_subdomain;
      interior_to_subdomain.reserve(subdomain_boundary_mask.size());
      boundary_to_subdomain.reserve(subdomain_boundary_mask.size());
      for (std::size_t i = 0; i < subdomain_boundary_mask.size(); ++i)
        if (subdomain_boundary_mask[i]) boundary_to_subdomain.push_back(i);
        else interior_to_subdomain.push_back(i);

      EnergyMinimalExtension<Mat, Vec> ext(*A_ovlp, interior_to_subdomain, boundary_to_subdomain);

      this->basis_.resize(boundary_data->size());
      for (std::size_t i = 0; i < boundary_data->size(); ++i) {
        this->basis_[i].resize(A_ovlp->N());
        for (std::size_t j = 0; j < (*boundary_data)[i].size(); ++j) this->basis_[i][boundary_to_subdomain[j]] = (*boundary_data)[i][j];

        auto interior_solution = ext.extend((*boundary_data)[i]);

        for (std::size_t j = 0; j < interior_solution.size(); ++j) this->basis_[i][interior_to_subdomain[j]] = interior_solution[j];
      }

      detail::finalize_eigenvectors(this->basis_, *pou);
    });
  }
};

template <class Vec = Dune::BlockVector<Dune::FieldVector<double, 1>>>
class SVDCoarseSpace : public CoarseSpaceBuilder<Vec> {
public:
  template <class Mat, class MaskVec, class MaskVec2>
  SVDCoarseSpace(std::shared_ptr<Mat> A_ovlp, std::shared_ptr<PartitionOfUnity> pou, const MaskVec& subdomain_boundary_mask, const MaskVec2& dirichlet_boundary_mask, const Dune::ParameterTree& ptree,
                 tf::Taskflow& taskflow, const std::string& ptree_prefix = "svd_coarse_space")
  {
    int n = ptree.sub(ptree_prefix).get("n", 10);
    bool mult_pou = ptree.sub(ptree_prefix).get("mult_pou", false);

    this->setup_task = taskflow.emplace([subdomain_boundary_mask, dirichlet_boundary_mask, A_ovlp, pou, n, mult_pou, this]() {
      logger::info("Setting up SVD coarse space");

      Dune::Timer timer;

      assert(A_ovlp->N() == subdomain_boundary_mask.size());
      assert(subdomain_boundary_mask.size() == dirichlet_boundary_mask.size());

      // Start by partitioning the dofs
      enum class DOFType : std::uint8_t { Interior, Boundary, Dirichlet };
      std::vector<DOFType> dof_partitioning(A_ovlp->N());
      std::size_t n_interior = 0;
      std::size_t n_boundary = 0;
      std::size_t n_dirichlet = 0;
      for (std::size_t i = 0; i < A_ovlp->N(); ++i) {
        if (dirichlet_boundary_mask[i] > 0) {
          dof_partitioning[i] = DOFType::Dirichlet;
          n_dirichlet++;
        }
        else if (subdomain_boundary_mask[i]) {
          dof_partitioning[i] = DOFType::Boundary;
          n_boundary++;
        }
        else {
          dof_partitioning[i] = DOFType::Interior;
          n_interior++;
        }
      }
      logger::debug_all("[SVD] Partitioned dofs, have {} in interior, {} on subdomain boundary, {} on Dirichlet boundary", n_interior, n_boundary, n_dirichlet);

      // Now create a subdomain -> interior mapping and a subdomain -> boundary mapping
      std::unordered_map<std::size_t, std::size_t> subdomain_to_interior;
      std::unordered_map<std::size_t, std::size_t> subdomain_to_boundary;
      subdomain_to_interior.reserve(n_interior);
      subdomain_to_boundary.reserve(n_boundary);
      std::size_t cnt_interior = 0;
      std::size_t cnt_boundary = 0;
      for (std::size_t i = 0; i < A_ovlp->N(); ++i)
        if (dof_partitioning[i] == DOFType::Interior) subdomain_to_interior[i] = cnt_interior++;
        else if (dof_partitioning[i] == DOFType::Boundary) subdomain_to_boundary[i] = cnt_boundary++;

      // Now create the matrix A_{i, \\Gamma_i}
      logger::info("[SVD] Create columns of A_{i, \\Gamma_i}");
      timer.reset();

      std::vector<Vec> A_igamma(n_boundary, Vec(n_interior));
      for (auto& column : A_igamma) column = 0; // zero initialise
      for (auto ri = A_ovlp->begin(); ri != A_ovlp->end(); ++ri) {
        if (dof_partitioning[ri.index()] != DOFType::Interior) continue;
        for (auto ci = ri->begin(); ci != ri->end(); ++ci)
          if (dof_partitioning[ci.index()] == DOFType::Boundary) A_igamma[subdomain_to_boundary[ci.index()]][subdomain_to_interior[ri.index()]] = *ci;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      logger::info("[SVD] Done creating columns of A_{i, \\Gamma_i}, took {}s", timer.elapsed());
      timer.reset();
      logger::info("[SVD] Computing T matrix");
      // ################################################################

      Mat A_int;
      auto avg = A_ovlp->nonzeroes() / A_ovlp->N() + 2;
      A_int.setBuildMode(Mat::implicit);
      A_int.setImplicitBuildModeParameters(avg, 0.2);
      A_int.setSize(n_interior, n_interior);
      for (auto ri = A_ovlp->begin(); ri != A_ovlp->end(); ++ri) {
        if (dof_partitioning[ri.index()] != DOFType::Interior) continue;
        for (auto ci = ri->begin(); ci != ri->end(); ++ci)
          if (dof_partitioning[ci.index()] == DOFType::Interior) A_int.entry(subdomain_to_interior[ri.index()], subdomain_to_interior[ci.index()]) = *ci;
      }
      A_int.compress();

      // Create the solver and solve for all right hand sides
      Dune::UMFPack<Mat> solver(A_int);
      Eigen::MatrixXd T(n_interior, n_boundary);

      Dune::InverseOperatorResult res;
      for (std::size_t i = 0; i < A_igamma.size(); ++i) {
        auto& rhs = A_igamma[i];
        Vec x(rhs.size());
        solver.apply(x, rhs, res);

        // Put solution into T
        for (std::size_t j = 0; j < x.size(); ++j) T((int)j, (int)i) = x[j];
      }

      // Scale T by partition of unity: T <- D * T
      for (std::size_t i = 0; i < pou->size(); ++i) {
        if (dof_partitioning[i] == DOFType::Interior) {
          const auto p = pou->vector()[i];
          const auto ii = subdomain_to_interior[i];
          for (std::size_t j = 0; j < n_boundary; ++j) T((int)ii, (int)j) *= p;
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      logger::info("[SVD] Done computing T matrix, took {}s", timer.elapsed());

      logger::info("[SVD] Computing SVD of T");
      timer.reset();

      auto svd = T.bdcSvd(Eigen::ComputeThinU);

      const auto& singular_values = svd.singularValues();
      std::ostringstream os;
      std::copy(singular_values.begin(), singular_values.end() - 1, std::ostream_iterator<double>(os, ","));
      os << singular_values[singular_values.size() - 1];
      // for (std::size_t i = 0; i < singular_values.size(); ++i) logger::info_all("[SVD] {}: {}", i, singular_values[i]);
      logger::info_all("[SVD] Computes singular values: [{}]", os.str());

      int rank{};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      std::ofstream sv_file("singular_values_" + std::to_string(rank) + ".txt");
      std::copy(singular_values.begin(), singular_values.end() - 1, std::ostream_iterator<double>(sv_file, ","));
      sv_file << singular_values[singular_values.size() - 1];

      this->basis_.resize(n);
      const auto& U = svd.matrixU();
      for (std::size_t i = 0; i < this->basis_.size(); ++i) {
        auto& b = this->basis_[i];
        b.resize(A_ovlp->N());
        b = 0;

        for (std::size_t j = 0; j < A_ovlp->N(); ++j)
          if (dof_partitioning[j] == DOFType::Interior) b[j] = U((int)(subdomain_to_interior[j]), (int)(i));
      }

      if (mult_pou) detail::finalize_eigenvectors(this->basis_, *pou);
    });
  }
};
