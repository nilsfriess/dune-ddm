#pragma once

#include "dune/ddm/logger.hh"

#include <cnpy.h>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/matrixredistribute.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/paamg/graph.hh>
#include <dune/istl/repartition.hh>
#include <string>
#include <unordered_set>

namespace Dune {

template <class Scalar, class GlobalIndex = std::size_t, class LocalIndex = int>
struct ParallelMatrixData {
  std::shared_ptr<BCRSMatrix<FieldMatrix<Scalar, 1, 1>>> matrix;
  std::shared_ptr<OwnerOverlapCopyCommunication<GlobalIndex, LocalIndex>> communication;
};

/** @brief Load a sparse matrix file and distribute it across MPI ranks.
 *
 *  Supports Matrix Market (.mtx/.mm) and scipy CSR sparse .npz files.
 *  The format is auto-detected from the file extension.
 *
 *  Rank 0 loads the full matrix from disk, serializes it to CSR arrays,
 *  and scatters each rank's contiguous block of rows via MPI_Scatterv.
 *  Each rank then builds a local matrix with a trivial contiguous
 *  decomposition (rank r owns rows [r*N/P, (r+1)*N/P)) plus one layer
 *  of copy (overlap) indices for cross-partition connectivity.
 *
 *  DUNE's graphRepartition (ParMETIS) then improves this initial
 *  decomposition and redistributeMatrix moves each rank's data to its
 *  final partition.
 *
 *  Peak memory is O(nnz) on rank 0 during loading; other ranks only
 *  ever hold O(nnz/P) data. After redistribution every rank holds
 *  only its partition.
 *
 *  @param helper   The MPIHelper instance.
 *  @param filename Path to the matrix file (.mtx/.mm for Matrix Market, .npz for scipy CSR).
 *  @return ParallelMatrixData with partitioned matrix and communication.
 */
template <class Scalar = double>
ParallelMatrixData<Scalar> readMatrixMarketParallel(const MPIHelper& helper, const std::string& filename, bool make_additive = true)
{
  using GlobalIndex = std::size_t;
  using LocalIndex = int;
  using Matrix = BCRSMatrix<FieldMatrix<Scalar, 1, 1>>;
  using Communication = OwnerOverlapCopyCommunication<GlobalIndex, LocalIndex>;

  const int size = helper.size();
  const int rank = helper.rank();

  Communication communication;
  MPI_Comm comm = communication.communicator();

  // ---- Phase 1: Rank 0 loads the matrix and serializes to CSR ----
  GlobalIndex globalN = 0;
  std::vector<GlobalIndex> all_row_sizes;
  std::vector<GlobalIndex> all_col_indices;
  std::vector<Scalar> all_values;

  // Detect file format from extension
  const bool is_npz = filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".npz") == 0;

  if (rank == 0) {
    if (is_npz) {
      // Load scipy CSR sparse matrix from .npz (contains indptr, indices, data, shape)
      cnpy::npz_t npz = cnpy::npz_load(filename);

      auto& shape_arr = npz.at("shape");
      auto& indptr_arr = npz.at("indptr");
      auto& indices_arr = npz.at("indices");
      auto& data_arr = npz.at("data");

      // Read matrix dimensions
      if (shape_arr.word_size == sizeof(int64_t)) globalN = static_cast<GlobalIndex>(shape_arr.data<int64_t>()[0]);
      else globalN = static_cast<GlobalIndex>(shape_arr.data<int32_t>()[0]);

      // Convert indptr to per-row sizes
      all_row_sizes.resize(globalN);
      if (indptr_arr.word_size == sizeof(int64_t)) {
        const auto* indptr = indptr_arr.data<int64_t>();
        for (GlobalIndex i = 0; i < globalN; ++i) all_row_sizes[i] = static_cast<GlobalIndex>(indptr[i + 1] - indptr[i]);
      }
      else {
        const auto* indptr = indptr_arr.data<int32_t>();
        for (GlobalIndex i = 0; i < globalN; ++i) all_row_sizes[i] = static_cast<GlobalIndex>(indptr[i + 1] - indptr[i]);
      }

      // Copy column indices
      const size_t total_nnz = indices_arr.num_vals;
      all_col_indices.resize(total_nnz);
      if (indices_arr.word_size == sizeof(int64_t)) {
        const auto* cols = indices_arr.data<int64_t>();
        for (size_t i = 0; i < total_nnz; ++i) all_col_indices[i] = static_cast<GlobalIndex>(cols[i]);
      }
      else {
        const auto* cols = indices_arr.data<int32_t>();
        for (size_t i = 0; i < total_nnz; ++i) all_col_indices[i] = static_cast<GlobalIndex>(cols[i]);
      }

      // Copy values
      all_values.resize(total_nnz);
      if (data_arr.word_size == sizeof(double)) {
        const auto* vals = data_arr.data<double>();
        for (size_t i = 0; i < total_nnz; ++i) all_values[i] = static_cast<Scalar>(vals[i]);
      }
      else {
        const auto* vals = data_arr.data<float>();
        for (size_t i = 0; i < total_nnz; ++i) all_values[i] = static_cast<Scalar>(vals[i]);
      }

      logger::info("Loaded .npz matrix, size {}x{}, nnz {}", globalN, globalN, total_nnz);
    }
    else {
      // Load Matrix Market format
      Matrix A;
      loadMatrixMarket(A, filename);
      globalN = A.N();
      logger::info("Loaded matrix, size {}x{}", A.N(), A.M());

      all_row_sizes.resize(globalN);
      all_col_indices.reserve(A.nonzeroes());
      all_values.reserve(A.nonzeroes());
      for (GlobalIndex i = 0; i < globalN; ++i) {
        all_row_sizes[i] = A[i].size();
        for (auto ci = A[i].begin(); ci != A[i].end(); ++ci) {
          all_col_indices.push_back(ci.index());
          all_values.push_back((*ci)[0][0]);
        }
      }
    }
  }

  // ---- Phase 2: Broadcast global size and scatter owned rows ----
  MPI_Bcast(&globalN, 1, MPITraits<GlobalIndex>::getType(), 0, comm);

  const GlobalIndex start = rank * globalN / size;
  const GlobalIndex end = (rank + 1) * globalN / size;
  const auto n_owned = end - start;
  logger::info_all("Local size: starts at {}, ends at {}, n_owned {}", start, end, n_owned);

  const auto getOwner = [&](GlobalIndex idx) -> int {
    if (idx >= globalN) return -1;
    return static_cast<int>((idx * size + size - 1) / globalN);
  };

  // Compute scatter parameters (only meaningful on rank 0)
  std::vector<int> row_sendcounts(size, 0);
  std::vector<int> row_senddispls(size, 0);
  std::vector<int> entry_sendcounts(size, 0);
  std::vector<int> entry_senddispls(size, 0);

  if (rank == 0) {
    for (int r = 0; r < size; ++r) {
      GlobalIndex rs = r * globalN / size;
      GlobalIndex re = (r + 1) * globalN / size;
      row_sendcounts[r] = static_cast<int>(re - rs);
      row_senddispls[r] = static_cast<int>(rs);
    }
    GlobalIndex offset = 0;
    for (int r = 0; r < size; ++r) {
      entry_senddispls[r] = static_cast<int>(offset);
      GlobalIndex rs = r * globalN / size;
      GlobalIndex re = (r + 1) * globalN / size;
      GlobalIndex entries = 0;
      for (GlobalIndex i = rs; i < re; ++i) entries += all_row_sizes[i];
      entry_sendcounts[r] = static_cast<int>(entries);
      offset += entries;
    }
  }

  // Scatter row sizes for owned rows
  std::vector<GlobalIndex> local_row_sizes(n_owned);
  MPI_Scatterv(rank == 0 ? all_row_sizes.data() : nullptr, row_sendcounts.data(), row_senddispls.data(), MPITraits<GlobalIndex>::getType(), local_row_sizes.data(), static_cast<int>(n_owned),
               MPITraits<GlobalIndex>::getType(), 0, comm);

  GlobalIndex local_nnz = 0;
  for (auto s : local_row_sizes) local_nnz += s;

  // Scatter column indices and values
  std::vector<GlobalIndex> local_col_indices(local_nnz);
  std::vector<Scalar> local_values(local_nnz);

  MPI_Scatterv(rank == 0 ? all_col_indices.data() : nullptr, entry_sendcounts.data(), entry_senddispls.data(), MPITraits<GlobalIndex>::getType(), local_col_indices.data(), static_cast<int>(local_nnz),
               MPITraits<GlobalIndex>::getType(), 0, comm);

  MPI_Scatterv(rank == 0 ? all_values.data() : nullptr, entry_sendcounts.data(), entry_senddispls.data(), MPITraits<Scalar>::getType(), local_values.data(), static_cast<int>(local_nnz),
               MPITraits<Scalar>::getType(), 0, comm);

  // Free rank 0 CSR data
  std::vector<GlobalIndex>().swap(all_row_sizes);
  std::vector<GlobalIndex>().swap(all_col_indices);
  std::vector<Scalar>().swap(all_values);

  // ---- Phase 3: Build index set and local matrix from scattered data ----
  std::unordered_set<int> neighbors;
  std::unordered_set<GlobalIndex> copy_indices;
  std::unordered_map<GlobalIndex, GlobalIndex> global_to_local;

  // Single pass over owned rows: determine copy indices, neighbors, and index publicity
  GlobalIndex cnt = 0;
  communication.indexSet().beginResize();
  {
    GlobalIndex entry_offset = 0;
    for (GlobalIndex i = start; i < end; ++i) {
      bool pub = false;
      GlobalIndex row_nnz = local_row_sizes[i - start];
      for (GlobalIndex j = 0; j < row_nnz; ++j) {
        GlobalIndex col = local_col_indices[entry_offset + j];
        if (int owner = getOwner(col); owner != rank) {
          if (owner == -1) DUNE_THROW(Dune::Exception, "Owner of index " << col << " could not be found");
          neighbors.insert(owner);
          copy_indices.insert(col);
          pub = true;
        }
      }
      global_to_local[i] = cnt;
      communication.indexSet().add(i, {cnt++, OwnerOverlapCopyAttributeSet::AttributeSet::owner, pub});
      entry_offset += row_nnz;
    }
  }
  for (auto cidx : copy_indices) {
    global_to_local[cidx] = cnt;
    communication.indexSet().add(cidx, {cnt++, OwnerOverlapCopyAttributeSet::AttributeSet::copy, true});
  }
  communication.indexSet().endResize();

  communication.remoteIndices().setNeighbours(neighbors);
  communication.remoteIndices().template rebuild<false>();

  // Build the local matrix: owned rows get their full entries, copy rows are empty
  // (copy row values reside on their owning rank and are redistributed from there).
  GlobalIndex max_rs = 1;
  for (auto s : local_row_sizes) max_rs = std::max(max_rs, s);

  Matrix A_local(global_to_local.size(), global_to_local.size(), max_rs, 0.2, Matrix::implicit);
  {
    GlobalIndex entry_offset = 0;
    for (GlobalIndex i = start; i < end; ++i) {
      GlobalIndex row_nnz = local_row_sizes[i - start];
      GlobalIndex local_row = global_to_local[i];
      for (GlobalIndex j = 0; j < row_nnz; ++j) {
        GlobalIndex col = local_col_indices[entry_offset + j];
        if (global_to_local.contains(col)) A_local.entry(local_row, global_to_local[col]) = local_values[entry_offset + j];
      }
      entry_offset += row_nnz;
    }
  }
  A_local.compress();

  // ---- Phase 4: Repartition with ParMETIS and redistribute ----
  std::shared_ptr<Communication> outcomm;
  RedistributeInformation<Communication> rinfo;

  graphRepartition(Amg::MatrixGraph<Matrix>(A_local), communication, static_cast<Metis::idx_t>(size), outcomm, rinfo.getInterface(), true);
  rinfo.setSetup();

  Matrix parallel_A;
  redistributeMatrix(A_local, parallel_A, communication, *outcomm, rinfo);

  logger::info_all("Partitioned matrix: {}x{}, nnz {}", parallel_A.N(), parallel_A.M(), parallel_A.nonzeroes());

  // ---- Phase 5 (optional): Make the matrix additive ----
  // For copy DOF rows, zero out entries whose columns reference non-owner DOFs
  // to avoid double-counting when using addOwnerCopyToOwnerCopy communication.
  if (make_additive) {
    const auto& indexSet = outcomm->indexSet();
    std::vector<bool> is_owner(parallel_A.N(), false);
    for (auto it = indexSet.begin(); it != indexSet.end(); ++it)
      if (it->local().attribute() == OwnerOverlapCopyAttributeSet::AttributeSet::owner) is_owner[it->local().local()] = true;
    for (auto it = indexSet.begin(); it != indexSet.end(); ++it) {
      if (it->local().attribute() == OwnerOverlapCopyAttributeSet::AttributeSet::copy) {
        auto local_idx = it->local().local();
        for (auto ci = parallel_A[local_idx].begin(); ci != parallel_A[local_idx].end(); ++ci)
          if (!is_owner[ci.index()]) *ci = 0;
      }
    }
  }

  ParallelMatrixData<Scalar, GlobalIndex, LocalIndex> data;
  data.matrix = std::make_shared<Matrix>(std::move(parallel_A));
  data.communication = outcomm;
  return data;
}

} // namespace Dune
