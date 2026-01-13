#pragma once

#include "dune/ddm/datahandles.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/pdelab_helper.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/grid/common/gridenums.hh>
#include <dune/istl/io.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/pdelab.hh>
#include <dune/pdelab/backend/istl/parallelhelper.hh>

/** @brief Determines the region where the "restricted Neumann" matrix is defined (to be used with the function PoissonProblem#assemble_overlapping_matrices()).
 */
enum class NeumannRegion : std::uint8_t {
  Overlap,         ///< Restricted Neumann matrix is only defined in the overlapping region
  ExtendedOverlap, ///< Restricted Neumann matrix is defined in the overlap + one additional layer of finite elements towards the interior
  All              ///< Restricted Neumann matrix is defined on the whole subdomain
};

/** @brief Symmetrically eliminate Dirichlet degrees of freedom from a matrix
 *
 *  For Dirichlet DOFs (where dirichlet_mask > 0), sets the diagonal entry to 1.0
 *  and all off-diagonal entries in that row to 0.0. Also zeros out the column
 *  entries corresponding to Dirichlet DOFs in other rows. This maintains the
 *  symmetric structure of the matrix while enforcing Dirichlet constraints.
 *
 *  @param A Matrix to modify (modified in place)
 *  @param dirichlet_mask Vector with non-zero values at Dirichlet DOF indices
 */
template <class Mat, class Vec>
void eliminate_dirichlet(Mat& A, const Vec& dirichlet_mask, bool symmetrically = true)
{
  for (auto ri = A.begin(); ri != A.end(); ++ri) {
    if (dirichlet_mask[ri.index()] > 0) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) *ci = (ci.index() == ri.index()) ? 1.0 : 0.0;
    }
    else {
      if (symmetrically)
        for (auto ci = ri->begin(); ci != ri->end(); ++ci)
          if (dirichlet_mask[ci.index()] > 0) *ci = 0.0;
    }
  }
}

/** @brief Symmetrically eliminate Dirichlet degrees of freedom using index mapping
 *
 *  Similar to eliminate_dirichlet() but uses an index mapping to translate between
 *  matrix indices and mask indices. This is used when the matrix has different
 *  dimensions than the mask (e.g., for restricted Neumann matrices).
 *
 *  @param A Matrix to modify (modified in place)
 *  @param dirichlet_mask Vector with non-zero values at Dirichlet DOF indices
 *  @param map Index mapping from matrix indices to mask indices
 */
template <class Mat, class Vec>
void eliminate_dirichlet(Mat& A, const Vec& dirichlet_mask, const std::vector<std::size_t>& map)
{
  for (auto ri = A.begin(); ri != A.end(); ++ri) {
    if (dirichlet_mask[map[ri.index()]] > 0) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) *ci = (ci.index() == ri.index()) ? 1.0 : 0.0;
    }
    else {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci)
        if (dirichlet_mask[map[ci.index()]] > 0) *ci = 0.0;
    }
  }
}

/** @brief Wrapper for the overlapping matrices assembled by assemble_overlapping_matrices().
 *
 *  @see assemble_overlapping_matrices() for explanations of the different matrices.
 */
template <class Mat>
struct OverlappingMatrices {
  std::shared_ptr<Mat> A_dir; ///< The overlapping Dirichlet matrix
  std::shared_ptr<Mat> A_neu; ///< The "first" Neumann matrix
  std::shared_ptr<Mat> B_neu; ///< The "second" Neumann matrix
};

/** @brief Assemble the overlapping Dirichlet and Neumann matrices

     This function assembles the stiffness matrix locally and then creates three matrices:
     - The "Dirichlet" matrix which is obtained by extracting a sub-matrix out of the global
       stiffness matrix on the overlapping subdomain that is described by \p extids. It is
       called the Dirichlet matrix because it corresponds to a problem with Dirichlet boundary
       conditions at the overlapping subdomain boundary. This is the matrix that should be used
       on the fine level in a two-level Schwarz method.
     - The first "Neumann" matrix which might also be defined on the whole overlapping subdomain,
       or on a smaller sub-region. The region is determined by the second parameter \p first_neumann_region.
     - The second "Neumann" matrix which might also be defined on the whole overlapping subdomain,
       or on a smaller sub-region. The region is determined by the third parameter \p second_neumann_region.

     If \p first_neumann_region and \p second_neumann_region are the same, then only one matrix
     will be assembled and the returned pointers point to the same matrix.

     The parameter \p matrix_size_eq_subdomain can be used to control the size of the assembled Neumann matrices.
     If it is true, then the Neumann matrices will have as many rows as there are dofs in the subdomain. If it is false
     and, e.g., \p first_neumann_region equals NeumannRegion::Overlap, then the Neumann matrix will only have
     as many rows/columns as there are degrees of freedom in the overlap region. This can be used to assemble
     the matrices for the "ring" coarse spaces.

     If the parameter \p call_make_additive is true, then the function make_additive will be called after assembly.
     In this case the last parameter novlp_comm must not be nullptr. This parameter must be set if the matrix
     that PDELab assembles locally is already consistent after assembly (which is the case if the EntitySet
     contains ghost elements, for example). Otherwise the assembled overlapping matrices will contain incorrect
     values.

     @see NeumannRegion for the options that can be used for \p first_neumann_region and \p second_neumann_region.
  */
template <class PDELabMat, class PDELabVec, class Vec, class GO, class Communication>
[[nodiscard]] std::tuple<OverlappingMatrices<Dune::PDELab::Backend::Native<PDELabMat>>, // Overlapping matrices
                         std::shared_ptr<Vec>,                                          // Overlapping Dirichlet mask
                         std::vector<std::size_t>>                                      // Mapping from the Neumann region to the whole subdomain (might be empty)
assemble_overlapping_matrices(PDELabMat& As, PDELabVec& x, const GO& go, const Vec& dirichlet_mask, Communication& comm, NeumannRegion first_neumann_region, NeumannRegion second_neumann_region,
                              int overlap, bool matrix_size_eq_subdomain = true, bool call_make_additive = false, const Communication* novlp_comm = nullptr)
{
  using Dune::PDELab::Backend::native;
  using Dune::PDELab::Backend::Native;
  using Mat = Native<PDELabMat>;
  logger::info("Assembling overlapping Dirichlet and Neumann matrices");

  int ownrank = comm.communicator().rank();
  const auto& paridxs = comm.indexSet();

  // Create variable size communicator for the overlapping index sets
  typename Communication::AllSet allset;
  Dune::Interface interface_ext;
  interface_ext.build(comm.remoteIndices(), allset, allset);
  Dune::VariableSizeCommunicator varcomm(interface_ext);

  // Create the (at this point still empty) overlapping subdomain matrix
  const auto& nAs = native(As);
  CreateMatrixDataHandle cmdh(nAs, comm.indexSet());
  varcomm.forward(cmdh);
  auto A_dir = std::make_shared<Mat>(cmdh.getOverlappingMatrix());

  // Identify the boundary of the overlapping subdomain
  IdentifyBoundaryDataHandle ibdh(*A_dir, paridxs);
  varcomm.forward(ibdh);

  // Now we know *our* subdomain boundary. We'll now create a vector on the overlapping
  // subdomain that contains the value 1 on the subdomain boundary and the value 2 one
  // layer into the subdomain. We'll communicate this vector with our neighbours so
  // that they can find out which elements they have to integrate separately in order
  // to send us the "Neumann correction terms".
  const auto& boundary_mask = ibdh.get_boundary_mask();

  std::vector<int> boundary_dst(boundary_mask.size(), std::numeric_limits<int>::max() - 1);
  for (std::size_t i = 0; i < boundary_dst.size(); ++i)
    if (boundary_mask[i]) boundary_dst[i] = 0;

  for (int round = 0; round <= 4 * overlap; ++round) {
    for (std::size_t i = 0; i < boundary_dst.size(); ++i)
      for (auto cit = (*A_dir)[i].begin(); cit != (*A_dir)[i].end(); ++cit) boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cit.index()] + 1); // Increase distance from boundary by one
  }

  std::vector<std::uint8_t> boundary_indicator(A_dir->N(), 0);
  for (std::size_t i = 0; i < boundary_indicator.size(); ++i)
    if (boundary_dst[i] == 0) boundary_indicator[i] = 1;
    else boundary_indicator[i] = 2;

  CopyVectorDataHandleWithRank cvdh(boundary_indicator, ownrank);
  varcomm.forward(cvdh);

  // Next, we turn this vector into two boolean masks for each rank.
  std::map<int, std::vector<bool>> on_boundary_mask_for_rank;
  std::map<int, std::vector<bool>> inside_boundary_mask_for_rank;
  for (const auto& [rank, copied_vec] : cvdh.copied_vecs) {
    on_boundary_mask_for_rank[rank].resize(nAs.N(), false);
    inside_boundary_mask_for_rank[rank].resize(nAs.N(), false);

    for (std::size_t i = 0; i < nAs.N(); ++i) {
      on_boundary_mask_for_rank[rank][i] = copied_vec[i] == 1;
      inside_boundary_mask_for_rank[rank][i] = copied_vec[i] == 2;
    }
  }

  std::vector<bool> on_boundary_mask(nAs.N(), false);
  std::vector<bool> outside_boundary_mask(nAs.N(), false);
  if (first_neumann_region != NeumannRegion::All and first_neumann_region != second_neumann_region)
    DUNE_THROW(Dune::NotImplemented, "Two different Neumann regions are only supported if the first is NeumannRegion::All");
  if (first_neumann_region == NeumannRegion::Overlap or second_neumann_region == NeumannRegion::Overlap) {
    for (std::size_t i = 0; i < nAs.N(); ++i) {
      on_boundary_mask[i] = boundary_dst[i] == (2 * overlap);
      outside_boundary_mask[i] = boundary_dst[i] == (2 * overlap + 1);
    }
  }
  else if (first_neumann_region == NeumannRegion::ExtendedOverlap or second_neumann_region == NeumannRegion::ExtendedOverlap) {
    for (std::size_t i = 0; i < nAs.N(); ++i) {
      on_boundary_mask[i] = boundary_dst[i] == (2 * overlap + 1);
      outside_boundary_mask[i] = boundary_dst[i] == (2 * overlap + 2);
    }
  }

  // No we can assemble while keeping track of the contributions that later need to be sent to other ranks so that they can assemble the
  // Neumann matrices they need.
  auto& wrapper = go.localAssembler().localOperator();
  wrapper.set_masks(nAs, &on_boundary_mask_for_rank, &inside_boundary_mask_for_rank, &on_boundary_mask, &outside_boundary_mask);
  go.jacobian(x, As);

  if (call_make_additive) {
    if (novlp_comm == nullptr) DUNE_THROW(Dune::Exception, "Need non-overlapping communicator for DG assembly");
    make_additive(As, *novlp_comm);
  }

  comm.buildGlobalLookup();
  const auto& glis = comm.globalLookup();
  auto triples_for_rank = wrapper.get_correction_triples(glis);

  // Now we have a set of {row, col, value} triples that represent corrections that remote ranks have to apply after overlap extension to turn
  // the matrices that they've obtained into matrices that correspond to a PDE with Neumann boundary conditions at the overlapping
  // subdomain boundary. Let's exchange this info with our neighbours.
  constexpr int nitems = 4;
  std::array<int, nitems> blocklengths = {1, 1, 1, 1};
  std::array<MPI_Datatype, nitems> types = {MPI_INT, MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_DOUBLE};
  MPI_Datatype triple_type = MPI_DATATYPE_NULL;
  std::array<MPI_Aint, nitems> offsets{0};
  offsets[0] = offsetof(TripleWithRank, rank);
  offsets[1] = offsetof(TripleWithRank, row);
  offsets[2] = offsetof(TripleWithRank, col);
  offsets[3] = offsetof(TripleWithRank, val);
  MPI_Type_create_struct(nitems, blocklengths.data(), offsets.data(), types.data(), &triple_type);
  MPI_Type_commit(&triple_type);

  std::vector<MPI_Request> requests;
  requests.reserve(triples_for_rank.size());
  for (const auto& [rank, triples] : triples_for_rank) {
    if (rank < 0) {
      // rank == -1 corresponds to corrections that we have to apply locally, so we can skip them here
      continue;
    }

    MPI_Isend(triples.data(), triples.size(), triple_type, rank, 0, comm.communicator(), &requests.emplace_back());
  }

  std::map<int, std::vector<TripleWithRank>> remote_triples;
  for (const auto& [rank, triples] : triples_for_rank) {
    if (rank < 0) {
      // rank < 0 corresponds to corrections that we have to apply locally, so we can skip them here
      continue;
    }

    MPI_Status status;
    int count{};

    MPI_Probe(rank, 0, comm.communicator(), &status);
    MPI_Get_count(&status, triple_type, &count);

    remote_triples[rank].resize(count);
    MPI_Recv(remote_triples[rank].data(), count, triple_type, rank, 0, comm.communicator(), MPI_STATUS_IGNORE);
  }
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

  // Log correction statistics before communication
  logger::debug_all("Rank {}: Sending corrections to {} ranks", ownrank, triples_for_rank.size() - 1);
  for (const auto& [rank, triples] : triples_for_rank)
    if (rank >= 0) logger::debug_all("Rank {}: Sending {} correction entries to rank {}", ownrank, triples.size(), rank);
    else logger::debug_all("Rank {}: {} local correction entries", ownrank, triples.size());

  // Now we can assemble the overlapping matrix

  AddMatrixDataHandle amdh(nAs, *A_dir, comm.indexSet());
  varcomm.forward(amdh);

  // Next, make sure that Dirichlet dofs are eliminated symmetrically
  auto dirichlet_mask_ovlp = std::make_shared<Vec>(A_dir->N());
  *dirichlet_mask_ovlp = 0;
  for (std::size_t i = 0; i < dirichlet_mask.N(); ++i) (*dirichlet_mask_ovlp)[i] = dirichlet_mask[i];
  AddVectorDataHandle<Vec> advdh;
  advdh.setVec(*dirichlet_mask_ovlp);
  varcomm.forward(advdh);

  // Next, assemble the Neumann matrices
  std::shared_ptr<Mat> A_neu;
  std::shared_ptr<Mat> B_neu;
  std::vector<std::size_t> neumann_region_to_subdomain;

  if (first_neumann_region == NeumannRegion::All) {
    A_neu = std::make_shared<Mat>(*A_dir);
    int corrections_applied = 0;
    int corrections_skipped = 0;
    double max_correction = 0.0;
    double sum_abs_corrections = 0.0;
    for (const auto& [rank, triples] : remote_triples) {
      for (const auto& triple : triples) {
        if (paridxs.exists(triple.row) && paridxs.exists(triple.col)) {
          auto lrow = paridxs[triple.row].local();
          auto lcol = paridxs[triple.col].local();

          (*A_neu)[lrow][lcol] -= triple.val;
          ++corrections_applied;
          max_correction = std::max(max_correction, std::abs(triple.val));
          sum_abs_corrections += std::abs(triple.val);
        }
        else {
          logger::warn_all("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
          ++corrections_skipped;
        }
      }
    }
    eliminate_dirichlet(*A_neu, *dirichlet_mask_ovlp);
  }
  else if (first_neumann_region == NeumannRegion::ExtendedOverlap or first_neumann_region == NeumannRegion::Overlap) {
    auto neumann_region_width = first_neumann_region == NeumannRegion::ExtendedOverlap ? 2 * overlap + 1 : 2 * overlap;

    if (matrix_size_eq_subdomain) {
      // First create a copy of the Dirichlet matrix, but only those entries in the extended overlap region
      auto avg = A_dir->nonzeroes() / A_dir->N() + 2;
      A_neu = std::make_shared<Mat>(A_dir->N(), A_dir->N(), avg, 0.2, Mat::implicit);
      for (auto ri = A_dir->begin(); ri != A_dir->end(); ++ri) {
        if (boundary_dst[ri.index()] > neumann_region_width) continue;

        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if (boundary_dst[ci.index()] > neumann_region_width) continue;

          A_neu->entry(ri.index(), ci.index()) = *ci;
        }
      }
      A_neu->compress();

      // Then apply the outer and inner Neumann corrections
      for (const auto& [rank, triples] : remote_triples) {
        for (const auto& triple : triples) {
          if (paridxs.exists(triple.row) && paridxs.exists(triple.col)) {
            auto lrow = paridxs[triple.row].local();
            auto lcol = paridxs[triple.col].local();

            (*A_neu)[lrow][lcol] -= triple.val;
          }
          else {
            logger::warn_all("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
          }
        }
      }

      for (const auto& triple : triples_for_rank[-1]) (*A_neu)[triple.row][triple.col] -= triple.val;

      eliminate_dirichlet(*A_neu, *dirichlet_mask_ovlp);
    }
    else {
      // In this case, we essentially do the same thing as above but now the matrix is only defined on the overlap region
      // (whereas above it is just zero everywhere else). We start by creating an index set for the overlap region, i.e.,
      // a vector with as many entries as there are dofs in the overlap regions that maps 'overlap region index' -> 'subdomain index'.
      auto n = std::count_if(boundary_dst.begin(), boundary_dst.end(), [&](auto&& x) { return x <= neumann_region_width; });
      neumann_region_to_subdomain.resize(n);
      std::size_t cnt = 0;
      for (std::size_t i = 0; i < boundary_dst.size(); ++i)
        if (boundary_dst[i] <= neumann_region_width) neumann_region_to_subdomain[cnt++] = i;

      // Also create the inverse mapping.
      // TODO: Check if it's faster to create a vector that just has bogus values outside the interesting region
      std::unordered_map<std::size_t, std::size_t> subdomain_to_neumann_region;
      for (std::size_t i = 0; i < neumann_region_to_subdomain.size(); ++i) subdomain_to_neumann_region[neumann_region_to_subdomain[i]] = i;

      // Then, build the matrix
      auto avg = A_dir->nonzeroes() / A_dir->N() + 2;
      A_neu = std::make_shared<Mat>(n, n, avg, 0.2, Mat::implicit);

      for (std::size_t i = 0; i < neumann_region_to_subdomain.size(); ++i) {
        auto ri = neumann_region_to_subdomain[i];
        for (auto ci = (*A_dir)[ri].begin(); ci != (*A_dir)[ri].end(); ++ci) {
          if (boundary_dst[ci.index()] > neumann_region_width) continue;

          A_neu->entry(i, subdomain_to_neumann_region[ci.index()]) = *ci;
        }
      }
      A_neu->compress();

      // Then apply the outer and inner Neumann corrections
      for (const auto& [rank, triples] : remote_triples) {
        for (const auto& triple : triples) {
          if (paridxs.exists(triple.row) && paridxs.exists(triple.col)) {
            auto lrow = paridxs[triple.row].local();
            auto lcol = paridxs[triple.col].local();

            assert(subdomain_to_neumann_region.contains(lrow) && subdomain_to_neumann_region.contains(lcol) && "Remote corrections should be applicable to ring matrix");

            (*A_neu)[subdomain_to_neumann_region[lrow]][subdomain_to_neumann_region[lcol]] -= triple.val;
          }
          else {
            logger::warn_all("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
          }
        }
      }

      for (const auto& triple : triples_for_rank[-1]) {
        assert(subdomain_to_neumann_region.contains(triple.row) && subdomain_to_neumann_region.contains(triple.col) && "Own corrections should be applicable to ring matrix");

        (*A_neu)[subdomain_to_neumann_region[triple.row]][subdomain_to_neumann_region[triple.col]] -= triple.val;
      }
    }
  }
  else {
    DUNE_THROW(Dune::NotImplemented, "Only NeumannRegion::All and NeumannRegion::ExtendedOverlap implemented for first Neumann matrix currently");
  }

  // Lastly, assemble then second Neumann matrix
  if (second_neumann_region == first_neumann_region) { B_neu = A_neu; }
  else if (second_neumann_region == NeumannRegion::Overlap) {
    // First create a copy of the Neumann matrix, but only those entries in the overlap region
    auto avg = A_dir->nonzeroes() / A_dir->N() + 2;
    B_neu = std::make_shared<Mat>(A_dir->N(), A_dir->N(), avg, 0.2, Mat::implicit);
    for (auto ri = A_neu->begin(); ri != A_neu->end(); ++ri) {
      if (boundary_dst[ri.index()] > 2 * overlap) continue;

      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        if (boundary_dst[ci.index()] > 2 * overlap) continue;

        B_neu->entry(ri.index(), ci.index()) = *ci;
      }
    }
    B_neu->compress();

    // Finally, apply the "inner" Neumann corrections
    for (const auto& triple : triples_for_rank[-1]) (*B_neu)[triple.row][triple.col] -= triple.val;

    eliminate_dirichlet(*B_neu, *dirichlet_mask_ovlp);
  }
  else {
    DUNE_THROW(Dune::NotImplemented, "Unknown neumann_region type");
  }

  // Symmetrically eliminate global Dirichlet dofs and subdomain boundary dofs in A_dir
  eliminate_dirichlet(*A_dir, *dirichlet_mask_ovlp);
  eliminate_dirichlet(*A_dir, boundary_mask);

  OverlappingMatrices<Dune::PDELab::Backend::Native<PDELabMat>> matrices;
  matrices.A_dir = std::move(A_dir);
  matrices.A_neu = std::move(A_neu);
  matrices.B_neu = std::move(B_neu);
  return {matrices, dirichlet_mask_ovlp, neumann_region_to_subdomain};
}
