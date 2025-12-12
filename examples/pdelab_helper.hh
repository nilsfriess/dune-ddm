#pragma once

#include "dune/ddm/helpers.hh"
#include "dune/ddm/overlap_extension.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/grid/common/gridenums.hh>
#include <dune/istl/io.hh>
#include <dune/pdelab.hh>

/** @brief Determines the region where the "restricted Neumann" matrix is defined (to be used with the function PoissonProblem#assemble_overlapping_matrices()).
 */
enum class NeumannRegion : std::uint8_t {
  Overlap,         ///< Restricted Neumann matrix is only defined in the overlapping region
  ExtendedOverlap, ///< Restricted Neumann matrix is defined in the overlap + one additional layer of finite elements towards the interior
  All              ///< Restricted Neumann matrix is defined on the whole subdomain
};

template <class GFS>
auto make_remote_indices(const GFS& gfs, const Dune::MPIHelper& helper)
{
  using Dune::PDELab::Backend::native;

  // Using the grid function space, we can generate a globally unique numbering
  // of the dofs. This is done by taking the local index, shifting it to the
  // upper 32 bits of a 64 bit number and taking our MPI rank as the lower 32
  // bits.
  using GlobalIndexVec = Dune::PDELab::Backend::Vector<GFS, std::uint64_t>;
  GlobalIndexVec giv(gfs);
  for (std::size_t i = 0; i < giv.N(); ++i) native(giv)[i] = (static_cast<std::uint64_t>(i + 1) << 32ULL) + helper.rank();

  // Now we have a unique global indexing scheme in the interior of each process
  // subdomain; at the process boundary we take the smallest among all
  // processes.
  GlobalIndexVec giv_before(gfs);
  giv_before = giv; // Copy the vector so that we can find out if we are the
                    // owner of a border index after communication
  Dune::PDELab::MinDataHandle mindh(gfs, giv);
  gfs.gridView().communicate(mindh, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);

  using BooleanVec = Dune::PDELab::Backend::Vector<GFS, bool>;
  BooleanVec isPublic(gfs);
  isPublic = false;
  Dune::PDELab::SharedDOFDataHandle shareddh(gfs, isPublic);
  gfs.gridView().communicate(shareddh, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);

  using AttributeLocalIndex = Dune::ParallelLocalIndex<Attribute>;
  using GlobalIndex = std::uint64_t;
  using ParallelIndexSet = Dune::ParallelIndexSet<GlobalIndex, AttributeLocalIndex>;
  using RemoteIndices = Dune::RemoteIndices<ParallelIndexSet>;

  auto paridxs = std::make_shared<ParallelIndexSet>();
  paridxs->beginResize();
  int public_count = 0;
  for (std::size_t i = 0; i < giv.N(); ++i) {
    paridxs->add(native(giv)[i],
                 {i,                                                                            // Local index is just i
                  native(giv)[i] == native(giv_before)[i] ? Attribute::owner : Attribute::copy, // If the index didn't change above, we own it
                  native(isPublic)[i]}                                                          // SharedDOFDataHandle determines if an index is public
    );
    if (native(isPublic)[i]) ++public_count;
  }
  paridxs->endResize();

  std::set<int> neighboursset;
  Dune::PDELab::GFSNeighborDataHandle nbdh(gfs, helper.rank(), neighboursset);
  gfs.gridView().communicate(nbdh, Dune::All_All_Interface, Dune::ForwardCommunication);
  std::vector<int> neighbours(neighboursset.begin(), neighboursset.end());

  auto* remoteindices = new RemoteIndices(*paridxs, *paridxs, helper.getCommunicator(), neighbours);
  remoteindices->rebuild<false>();

  // RemoteIndices store a reference to the paridxs that are passed to the constructor.
  // In order to avoid dangling references, we capture the paridxs shared_ptr in the lambda
  // to increase the reference count which ensures that it will be deleted as soon as the
  // remoteindices are deleted.
  return std::shared_ptr<RemoteIndices>(remoteindices, [paridxs](auto* ptr) mutable { delete ptr; });
}

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
void eliminate_dirichlet(Mat& A, const Vec& dirichlet_mask)
{
  for (auto ri = A.begin(); ri != A.end(); ++ri) {
    if (dirichlet_mask[ri.index()] > 0) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) *ci = (ci.index() == ri.index()) ? 1.0 : 0.0;
    }
    else {
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

     The created matrices can be accessed via the methods get_dirichlet_matrix(), get_first_neumann_matrix(),
     and get_second_neumann_matrix(). If \p first_neumann_region and \p second_neumann_region are
     the same, then only two matrices will be assembled and get_first_neumann_matrix() and get_second_neumann_matrix()
     return pointers to the same matrix.

     The last parameter \p neumann_size_as_dirichlet can be used to control the size of the Neumann matrix.
     If it is true, then the Neumann matrices will be of the same size as the Dirichlet matrix. If it is false
     and, e.g., \p first_neumann_region equals NeumannRegion::Overlap, then the Neumann matrix will only have
     as many rows/columns as there are degrees of freedom in the overlap region. This can be used to assemble
     the matrices for the "ring" coarse spaces.

     @see NeumannRegion for the options that can be used for \p first_neumann_region and \p second_neumann_region.
  */
template <class RemoteIndices, class PDELabMat, class PDELabVec, class Vec, class GO>
[[nodiscard]] std::tuple<std::shared_ptr<Dune::PDELab::Backend::Native<PDELabMat>>, // The dirichlet matrix
                         std::shared_ptr<Dune::PDELab::Backend::Native<PDELabMat>>, // The "first" Neumann matrix
                         std::shared_ptr<Dune::PDELab::Backend::Native<PDELabMat>>, // The "second" Neumann matrix
                         std::shared_ptr<Vec>,                                      // The dirichlet mask extended to the overlapping subdomain
                         std::vector<std::size_t>>                                  // Mapping from the Neumann region to the whole subdomain (might be empty)
assemble_overlapping_matrices(PDELabMat& As, PDELabVec& x, const GO& go, const Vec& dirichlet_mask, const ExtendedRemoteIndices<RemoteIndices, Dune::PDELab::Backend::Native<PDELabMat>>& extids,
                              NeumannRegion first_neumann_region, NeumannRegion second_neumann_region, bool neumann_size_as_dirichlet = true)
{
  using Dune::PDELab::Backend::native;
  using Dune::PDELab::Backend::Native;
  using Mat = Native<PDELabMat>;
  logger::info("Assembling overlapping Dirichlet and Neumann matrices");

  int ownrank{};
  MPI_Comm_rank(extids.get_remote_indices().communicator(), &ownrank);
  auto ovlp_paridxs = extids.get_parallel_index_set();
  auto varcomm_ext = extids.get_overlapping_communicator();

  // Create the (at this point still empty) overlapping subdomain matrix
  const auto& nAs = native(As);
  auto A_dir = std::make_shared<Mat>(extids.create_overlapping_matrix(nAs));

  // Identify the boundary of the overlapping subdomain
  const auto& paridxs = extids.get_parallel_index_set();
  IdentifyBoundaryDataHandle ibdh(*A_dir, paridxs);
  varcomm_ext.forward(ibdh);

  // Now we know *our* subdomain boundary. We'll now create a vector on the overlapping
  // subdomain that contains the value 1 on the subdomain boundary and the value 2 one
  // layer into the subdomain. We'll communicate this vector with our neighbours so
  // that they can find out which elements they have to integrate separately in order
  // to send us the "Neumann correction terms".
  // const auto& boundary_mask = ibdh.get_boundary_mask();
  const auto& boundary_mask = extids.get_overlapping_boundary_mask();

  std::vector<int> boundary_dst(boundary_mask.size(), std::numeric_limits<int>::max() - 1);
  for (std::size_t i = 0; i < boundary_dst.size(); ++i)
    if (boundary_mask[i]) boundary_dst[i] = 0;

  int overlap = extids.get_overlap();
  for (int round = 0; round <= overlap; ++round) {
    for (std::size_t i = 0; i < boundary_dst.size(); ++i)
      for (auto cit = (*A_dir)[i].begin(); cit != (*A_dir)[i].end(); ++cit) boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cit.index()] + 1); // Increase distance from boundary by one
  }

  std::vector<std::uint8_t> boundary_indicator(extids.size(), 0);
  for (std::size_t i = 0; i < extids.size(); ++i)
    if (boundary_dst[i] == 0) boundary_indicator[i] = 1;
    else if (boundary_dst[i] == 1) boundary_indicator[i] = 2;

  int rank{};
  MPI_Comm_rank(extids.get_remote_indices().communicator(), &rank);
  CopyVectorDataHandleWithRank cvdh(boundary_indicator, rank);
  varcomm_ext.forward(cvdh);

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

  Dune::GlobalLookupIndexSet glis(extids.get_parallel_index_set());
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

    MPI_Isend(triples.data(), triples.size(), triple_type, rank, 0, extids.get_remote_indices().communicator(), &requests.emplace_back());
  }

  std::map<int, std::vector<TripleWithRank>> remote_triples;
  for (const auto& [rank, triples] : triples_for_rank) {
    if (rank < 0) {
      // rank < 0 corresponds to corrections that we have to apply locally, so we can skip them here
      continue;
    }

    MPI_Status status;
    int count{};

    MPI_Probe(rank, 0, extids.get_remote_indices().communicator(), &status);
    MPI_Get_count(&status, triple_type, &count);

    remote_triples[rank].resize(count);
    MPI_Recv(remote_triples[rank].data(), count, triple_type, rank, 0, extids.get_remote_indices().communicator(), MPI_STATUS_IGNORE);
  }
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

  // Now we can assemble the overlapping matrix

  AddMatrixDataHandle amdh(nAs, *A_dir, extids.get_parallel_index_set());
  extids.get_overlapping_communicator().forward(amdh);

  // Next, make sure that Dirichlet dofs are eliminated symmetrically
  auto dirichlet_mask_ovlp = std::make_shared<Vec>(A_dir->N());
  *dirichlet_mask_ovlp = 0;
  for (std::size_t i = 0; i < dirichlet_mask.N(); ++i) (*dirichlet_mask_ovlp)[i] = dirichlet_mask[i];
  AddVectorDataHandle<Vec> advdh;
  advdh.setVec(*dirichlet_mask_ovlp);
  varcomm_ext.forward(advdh);

  // Check A_dir before Dirichlet elimination
  double max_diag = 0.0, min_diag = 1e100;
  int zero_diags = 0, zero_rows = 0;
  std::vector<std::size_t> zero_row_indices_before;
  for (auto ri = A_dir->begin(); ri != A_dir->end(); ++ri) {
    bool row_is_zero = true;
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      if (std::abs((*ci)[0][0]) > 1e-14) {
        row_is_zero = false;
        break;
      }
    }
    if (row_is_zero) {
      ++zero_rows;
      if (zero_rows <= 10) zero_row_indices_before.push_back(ri.index());
    }

    if (ri->find(ri.index()) != ri->end()) {
      auto diag_val = (*ri)[ri.index()][0][0];
      if (std::abs(diag_val) < 1e-14) ++zero_diags;
      max_diag = std::max(max_diag, std::abs(diag_val));
      min_diag = std::min(min_diag, std::abs(diag_val));
    }
  }

  eliminate_dirichlet(*A_dir, *dirichlet_mask_ovlp);

  // Check A_dir after Dirichlet elimination
  max_diag = 0.0;
  min_diag = 1e100;
  zero_diags = 0;
  zero_rows = 0;
  std::vector<std::size_t> zero_row_indices;
  for (auto ri = A_dir->begin(); ri != A_dir->end(); ++ri) {
    bool row_is_zero = true;
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      if (std::abs((*ci)[0][0]) > 1e-14) {
        row_is_zero = false;
        break;
      }
    }
    if (row_is_zero) {
      ++zero_rows;
      if (zero_rows <= 10) zero_row_indices.push_back(ri.index());
    }

    if (ri->find(ri.index()) != ri->end()) {
      auto diag_val = (*ri)[ri.index()][0][0];
      if (std::abs(diag_val) < 1e-14) ++zero_diags;
      max_diag = std::max(max_diag, std::abs(diag_val));
      min_diag = std::min(min_diag, std::abs(diag_val));
    }
  }

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
        if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
          auto lrow = ovlp_paridxs[triple.row].local();
          auto lcol = ovlp_paridxs[triple.col].local();

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
    // Just to be sure
    eliminate_dirichlet(*A_neu, *dirichlet_mask_ovlp);
  }
  else if (first_neumann_region == NeumannRegion::ExtendedOverlap or first_neumann_region == NeumannRegion::Overlap) {
    auto neumann_region_width = first_neumann_region == NeumannRegion::ExtendedOverlap ? 2 * overlap + 1 : 2 * overlap;

    if (neumann_size_as_dirichlet) {
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
          if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
            auto lrow = ovlp_paridxs[triple.row].local();
            auto lcol = ovlp_paridxs[triple.col].local();

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
          if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
            auto lrow = ovlp_paridxs[triple.row].local();
            auto lcol = ovlp_paridxs[triple.col].local();

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

  return {A_dir, A_neu, B_neu, dirichlet_mask_ovlp, neumann_region_to_subdomain};
}
