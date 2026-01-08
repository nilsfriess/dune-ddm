#pragma once

#include "dune/ddm/logger.hh"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <dune/common/parallel/indexset.hh>
#include <dune/istl/io.hh>
#include <dune/istl/scalarproducts.hh>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

template <class Vec>
class CopyVectorDataHandle {
public:
  using DataType = typename Vec::block_type;

  bool fixedSize() { return true; }
  std::size_t size(int) { return 1; }

  template <class Buffer>
  void gather(Buffer& buffer, int i)
  {
    buffer.write((*vec)[i]);
  }

  template <class Buffer>
  void scatter(Buffer& buffer, int i, int)
  {
    DataType data;
    buffer.read(data);
    (*vec)[i] = data;
  }

  void setVec(Vec& v) { vec = &v; }

private:
  Vec* vec;
};

template <class Vec>
class AddVectorDataHandle {
public:
  using DataType = typename Vec::block_type;

  bool fixedSize() { return true; }
  std::size_t size(int) { return 1; }

  template <class Buffer>
  void gather(Buffer& buffer, int i)
  {
    buffer.write(sourcevec[i]);
  }

  template <class Buffer>
  void scatter(Buffer& buffer, int i, int)
  {
    DataType data;
    buffer.read(data);
    (*targetvec)[i] += data;
  }

  void setVec(Vec& v)
  {
    // We have to copy the vector because Dune's VariableSizeCommunicator does not ensure that gather is called before scatter.
    // If we would use the same vector, we might add values from rank A into our vec during scatter, which are then send
    // to rank B. If A and B are in the same overlap region, then B might have already gotten the values from A, so that
    // together with our values, it effectively added the values from A twice.
    sourcevec = v;
    targetvec = &v;
  }

private:
  Vec sourcevec;
  Vec* targetvec;
};

template <class Vec>
class CopyVectorDataHandleWithRank {
public:
  using DataType = std::pair<int, typename Vec::value_type>;

  explicit CopyVectorDataHandleWithRank(const Vec& v, int rank)
      : sourcevec(&v)
      , rank(rank)
  {
  }

  bool fixedSize() { return true; }
  std::size_t size(int) { return 1; }

  template <class Buffer>
  void gather(Buffer& buffer, int i)
  {
    buffer.write(std::make_pair(rank, (*sourcevec)[i]));
  }

  template <class Buffer>
  void scatter(Buffer& buffer, int i, int)
  {
    DataType data;
    buffer.read(data);
    const auto& [from_rank, vec] = data;

    if (not copied_vecs.contains(from_rank)) {
      copied_vecs.emplace(from_rank, *sourcevec);
      std::fill(copied_vecs[from_rank].begin(), copied_vecs[from_rank].end(), 0);
    }
    copied_vecs[from_rank][i] = vec;
  }

  std::map<int, Vec> copied_vecs;

private:
  const Vec* sourcevec;
  int rank;
};

template <class Mat, class ParallelIndexSet>
class IdentifyBoundaryDataHandle {
public:
  IdentifyBoundaryDataHandle(const Mat& A, const ParallelIndexSet& paridxs)
      : boundary_mask(paridxs.size(), false)
      , paridxs{paridxs}
      , A{A}
      , glis{paridxs}
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  IdentifyBoundaryDataHandle(const IdentifyBoundaryDataHandle&) = delete;
  IdentifyBoundaryDataHandle(IdentifyBoundaryDataHandle&&) = delete;
  IdentifyBoundaryDataHandle& operator=(const IdentifyBoundaryDataHandle&) = delete;
  IdentifyBoundaryDataHandle& operator=(IdentifyBoundaryDataHandle&&) = delete;
  ~IdentifyBoundaryDataHandle() = default;

  using DataType = typename ParallelIndexSet::GlobalIndex;

  bool fixedSize() { return false; }

  std::size_t size(std::size_t i)
  {
    int count = 1; // send at least a dummy value
    for (auto cit = A[i].begin(); cit != A[i].end(); ++cit)
      if (cit.index() != i) ++count;
    return count;
  }

  template <class B>
  void gather(B& buffer, std::size_t i)
  {
    buffer.write(rank);
    for (auto cit = A[i].begin(); cit != A[i].end(); ++cit)
      if (cit.index() != i) buffer.write(glis.pair(cit.index())->global());
  }

  template <class Buffer>
  void scatter(Buffer& buffer, std::size_t i, std::size_t size)
  {
    DataType otherrank;
    buffer.read(otherrank); // Read the dummy value
    boundaryMaskForRank[otherrank].resize(paridxs.size(), false);

    DataType gi;

    for (std::size_t k = 1; k < size; k++) {
      buffer.read(gi);
      if (not paridxs.exists(gi)) { // If we don't know the received global index, i is a boundary index
        boundaryMaskForRank[otherrank][i] = true;
        boundary_mask[i] = true;
      }
    }
  }

  const std::vector<bool>& get_boundary_mask() const
  {
    return boundary_mask;
  }

  const std::map<int, std::vector<bool>>& getBoundaryMaskForRank() const { return boundaryMaskForRank; }

private:
  std::map<int, std::vector<bool>> boundaryMaskForRank;
  std::vector<bool> boundary_mask;
  const ParallelIndexSet& paridxs;
  const Mat& A;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;

  int rank;
};

struct RankTuple {
  using value_type = int;

  std::vector<std::set<int>> rankmap;
  int rank;
};

class RankDataHandle {
public:
  using DataType = int;

  static int gather(const RankTuple& rt, int) { return rt.rank; }
  static void scatter(RankTuple& rt, int otherrank, std::size_t i) { rt.rankmap[i].insert(otherrank); }
};

template <class Mat, class GlobalIndex>
class IndexsetExtensionMatrixGraphDataHandle {
public:
  using DataType = GlobalIndex;

  IndexsetExtensionMatrixGraphDataHandle(int rank, const Mat& A, std::unordered_set<GlobalIndex>& gis)
      : rank(rank)
      , A(A)
      , gis(gis)
  {
  }

  void set_index_set(std::vector<GlobalIndex>& ltg_)
  {
    ltg = &ltg_;
    ltg_copy = *ltg;
  }

  IndexsetExtensionMatrixGraphDataHandle(const IndexsetExtensionMatrixGraphDataHandle&) = delete;
  IndexsetExtensionMatrixGraphDataHandle(IndexsetExtensionMatrixGraphDataHandle&&) = delete;
  IndexsetExtensionMatrixGraphDataHandle& operator=(const IndexsetExtensionMatrixGraphDataHandle&) = delete;
  IndexsetExtensionMatrixGraphDataHandle& operator=(IndexsetExtensionMatrixGraphDataHandle&&) = delete;
  ~IndexsetExtensionMatrixGraphDataHandle() = default;

  bool fixedSize() { return false; }

  std::size_t size(std::size_t i)
  {
    std::size_t count = 1;
    if (i < A.N()) {
      for (auto cit = A[i].begin(); cit != A[i].end(); ++cit)
        if (cit.index() != i) count++;
    }
    return count;
  }

  template <class Buffer>
  void gather(Buffer& b, std::size_t i)
  {
    b.write(0);
    if (i < A.N()) {
      for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
        if (cit.index() != i) {
          b.write(ltg_copy[cit.index()]);

          // All ranks that previously already knew index `i` will now also know index `cit.index()`,
          // so we update the rankmap accordingly.
          for (const auto& p : rankmap[i]) updated_rankmap[cit.index()].insert(p);
        }
      }
    }
  }

  template <class Buffer>
  void scatter(Buffer& b, std::size_t, std::size_t size)
  {
    GlobalIndex gi;
    b.read(gi);
    for (std::size_t k = 1; k < size; ++k) {
      b.read(gi);
      if (not gis.count(gi)) {
        ltg->push_back(gi);
        gis.insert(gi);
      }
    }
  }

  std::vector<std::set<int>> rankmap;
  std::vector<std::set<int>> updated_rankmap;

private:
  int rank;
  const Mat& A;

  std::vector<GlobalIndex>* ltg;
  std::unordered_set<GlobalIndex>& gis;

  std::vector<GlobalIndex> ltg_copy;
};

class UpdateRankInfoDataHandle {
public:
  void set_rankmap(std::vector<std::set<int>>& rankmap_)
  {
    rankmap = &rankmap_;
    rankmap_send = rankmap_;
  }

  UpdateRankInfoDataHandle(int rank)
      : rank(rank)
  {
  }

  UpdateRankInfoDataHandle(const UpdateRankInfoDataHandle&) = delete;
  UpdateRankInfoDataHandle(UpdateRankInfoDataHandle&&) = delete;
  UpdateRankInfoDataHandle& operator=(const UpdateRankInfoDataHandle&) = delete;
  UpdateRankInfoDataHandle& operator=(UpdateRankInfoDataHandle&&) = delete;
  ~UpdateRankInfoDataHandle() = default;

  using DataType = int;

  bool fixedSize() { return false; }
  std::size_t size(std::size_t i) { return 1 + (*rankmap)[i].size(); }

  template <class Buffer>
  void gather(Buffer& buffer, std::size_t i)
  {
    buffer.write(0);
    for (auto&& r : rankmap_send[i]) buffer.write(r);
  }

  template <class Buffer>
  void scatter(Buffer& buffer, std::size_t i, std::size_t size)
  {
    int curr_r = -1;
    buffer.read(curr_r);
    for (std::size_t r = 1; r < size; ++r) {
      buffer.read(curr_r);
      if (curr_r != rank) (*rankmap)[i].insert(curr_r);
    }
  }

private:
  int rank;
  std::vector<std::set<int>>* rankmap{nullptr};
  std::vector<std::set<int>> rankmap_send; // Scatter is not guaranteed to happen after gather, so we need to copy the map before sending
};

/* A data handle to exchange global indices */
template <class Mat, class ParallelIndexSet>
class DataHandle {
public:
  DataHandle(const Mat& A, const ParallelIndexSet& paridxs, const std::map<int, std::set<int>>& connected_indices, const std::map<int, std::set<int>>& connected_ranks, int rank)
      : A(A)
      , glis(paridxs)
      , paridxs(paridxs)
      , connected_indices(connected_indices)
      , connected_ranks(connected_ranks)
      , rank(rank)
  {
  }
  DataHandle(const DataHandle&) = delete;
  DataHandle(DataHandle&&) = delete;
  DataHandle& operator=(const DataHandle&) = delete;
  DataHandle& operator=(DataHandle&&) = delete;
  ~DataHandle() = default;

  using DataType = typename ParallelIndexSet::GlobalIndex;

  bool fixedSize() { return false; }
  std::size_t size(int i)
  {
    // assert(connected_indices.count(i) > 0);

    std::size_t count = 0;
    count += 1; // The number of global indices and connected ranks we send for this local index

    if (connected_indices.count(i)) {
      for (const auto& idx : connected_indices.at(i)) {
        count += 1; // The actual global index

        count += 1;                                                              // The number of ranks that also know this index
        if (connected_ranks.count(idx)) count += connected_ranks.at(idx).size(); // The ranks
      }
    }

    return count;
  }

  template <class Buffer>
  void gather(Buffer& buffer, int i)
  {
    if (connected_indices.count(i)) {
      buffer.write(connected_indices.at(i).size());
      for (const auto& li : connected_indices.at(i)) {
        buffer.write(glis.pair(li)->global());

        if (connected_ranks.count(li)) {
          buffer.write(connected_ranks.at(li).size());
          for (const auto& r : connected_ranks.at(li)) buffer.write(r);
        }
        else {
          buffer.write(0);
        }
      }
    }
    else {
      buffer.write(0);
    }
  }

  template <class Buffer>
  void scatter(Buffer& buffer, int, int)
  {
    DataType data;

    buffer.read(data); // Number of global indices and connected ranks

    for (std::size_t i = 0; i < data; ++i) {
      DataType gi;
      buffer.read(gi);
      gis.insert(gi);

      // Read number of connected ranks
      DataType count;
      buffer.read(count);

      for (std::size_t j = 0; j < count; ++j) {
        DataType remoterank;
        buffer.read(remoterank);
        neighbours_for_gidx[gi].insert(static_cast<int>(remoterank));
      }
    }
  }

  std::map<DataType, std::set<int>> neighbours_for_gidx;
  std::set<DataType> gis;

private:
  const Mat& A;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;
  const ParallelIndexSet& paridxs;
  const std::map<int, std::set<int>>& connected_indices;
  const std::map<int, std::set<int>>& connected_ranks;
  int rank;
};

template <class Mat, class ParallelIndexSet>
class CreateMatrixDataHandle {
public:
  CreateMatrixDataHandle(const Mat& A, const ParallelIndexSet& paridxs)
      : A(A)
      , paridxs(paridxs)
      , glis(paridxs)
      , Aovlp(paridxs.size(), paridxs.size(), static_cast<double>(A.nonzeroes()) / A.N(), 0.6, Mat::implicit)
  {
    for (auto rIt = A.begin(); rIt != A.end(); ++rIt)
      for (auto cIt = rIt->begin(); cIt != rIt->end(); ++cIt) Aovlp.entry(rIt.index(), cIt.index()) = 0.0;
  }
  CreateMatrixDataHandle(const CreateMatrixDataHandle&) = delete;
  CreateMatrixDataHandle(CreateMatrixDataHandle&&) = delete;
  CreateMatrixDataHandle& operator=(const CreateMatrixDataHandle&) = delete;
  CreateMatrixDataHandle& operator=(CreateMatrixDataHandle&&) = delete;
  ~CreateMatrixDataHandle() = default;

  using DataType = typename ParallelIndexSet::GlobalIndex;

  bool fixedSize() { return false; }
  std::size_t size(int i)
  {
    std::size_t count = 1;
    if (static_cast<std::size_t>(i) < A.N()) // Only if the index is part of the original matrix
      for (auto cIt = A[i].begin(); cIt != A[i].end(); ++cIt) count++;
    return count;
  }

  template <class Buffer>
  void gather(Buffer& buffer, std::size_t i)
  {
    buffer.write(1); // send dummy data
    if (i < A.N())
      for (auto cIt = A[i].begin(); cIt != A[i].end(); ++cIt) buffer.write(glis.pair(cIt.index())->global());
  }

  template <class Buffer>
  void scatter(Buffer& buffer, int i, int size)
  {
    DataType gi;
    buffer.read(gi); // read dummy data
    for (int k = 0; k < size - 1; k++) {
      buffer.read(gi); // read global index
      if (paridxs.exists(gi)) Aovlp.entry(i, paridxs[gi].local()) = 0.0;
    }
  }

  Mat&& getOverlappingMatrix()
  {
    Aovlp.compress();
    return std::move(Aovlp);
  }

private:
  const Mat& A;
  const ParallelIndexSet& paridxs;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;

  Mat Aovlp;
};

// TODO: This is a bit cursed currently. We have to send an index and a corresponding matrix entry.
//       We do this by reinterpeting the entry using memcpy as an index, send both values as the same
//       type and reinterpret back on the receiver side. We should probably just split the communication
//       of the indices and the communication of the entries.
//       We do not send them as std::pairs because that was very inefficient.
template <class Mat, class ParallelIndexSet>
class AddMatrixDataHandle {
  using GlobalIndex = typename ParallelIndexSet::GlobalIndex;
  using MatrixEntry = typename Mat::block_type;

public:
  // using DataType = std::pair<GlobalIndex, MatrixEntry>;
  using DataType = GlobalIndex;
  static_assert(sizeof(GlobalIndex) == sizeof(MatrixEntry));

  AddMatrixDataHandle(const Mat& A, Mat& Aovlp, const ParallelIndexSet& paridxs)
      : Asource(A)
      , Atarget(Aovlp)
      , paridxs(paridxs)
      , glis(paridxs)
  {
    for (auto rIt = A.begin(); rIt != A.end(); ++rIt)
      for (auto cIt = rIt->begin(); cIt != rIt->end(); ++cIt) Atarget[rIt.index()][cIt.index()] = Asource[rIt.index()][cIt.index()];

    scatter_event = Logger::get().registerOrGetEvent("OverlapExtension", "add Matrix scatter");
    gather_event = Logger::get().registerOrGetEvent("OverlapExtension", "add Matrix gather");
  }
  AddMatrixDataHandle(const AddMatrixDataHandle&) = delete;
  AddMatrixDataHandle(AddMatrixDataHandle&&) = delete;
  AddMatrixDataHandle& operator=(const AddMatrixDataHandle&) = delete;
  AddMatrixDataHandle& operator=(AddMatrixDataHandle&&) = delete;
  ~AddMatrixDataHandle() = default;

  bool fixedSize() { return false; }
  std::size_t size(int i)
  {
    std::size_t count = 1; // Send some dummy data
    if (static_cast<std::size_t>(i) < Asource.N())
      for (auto cIt = Asource[i].begin(); cIt != Asource[i].end(); ++cIt) count += 2; // We send an index and an entry
    return count;
  }

  template <class Buffer>
  void gather(Buffer& buffer, int i)
  {
    Logger::ScopedLog sl{gather_event};

    buffer.write(0); // Send dummy data

    if (static_cast<std::size_t>(i) < Asource.N())
      for (auto cIt = Asource[i].begin(); cIt != Asource[i].end(); ++cIt) {
        // buffer.write(DataType(glis.pair(cIt.index())->global(), *cIt));
        buffer.write(glis.pair(cIt.index())->global());

        GlobalIndex entry;
        double d = (*cIt)[0][0];
        std::memcpy(&entry, &d, sizeof(entry));
        buffer.write(entry);
      }
  }

  template <class Buffer>
  void scatter(Buffer& buffer, int i, std::size_t size)
  {
    Logger::ScopedLog sl{scatter_event};

    DataType idx;
    buffer.read(idx); // read dummy data

    DataType entry;
    MatrixEntry actual_entry;
    std::size_t count = 1;
    while (count < size) {
      buffer.read(idx);
      buffer.read(entry);

      std::memcpy(&actual_entry, &entry, sizeof(entry));

      if (paridxs.exists(idx)) Atarget[i][paridxs[idx].local()] += actual_entry;

      count += 2;
    }
  }

private:
  const Mat& Asource;
  Mat& Atarget;

  const ParallelIndexSet& paridxs;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;

  Logger::Event* scatter_event;
  Logger::Event* gather_event;
};

template <class ParallelIndexSet>
class MarkIndicesForRank {
  using GlobalIndex = typename ParallelIndexSet::GlobalIndex;

public:
  using DataType = std::pair<int, bool>; // Our rank and whether the index is marked or not

  MarkIndicesForRank(const std::vector<std::size_t>& marked_indices, int rank, const ParallelIndexSet& paridxs)
      : mask(paridxs.size(), false)
      , rank(rank)
      , paridxs(paridxs)
      , glis(paridxs)
  {
    for (auto i : marked_indices) mask[i] = true;
  }
  MarkIndicesForRank(const MarkIndicesForRank&) = delete;
  MarkIndicesForRank(MarkIndicesForRank&&) = delete;
  MarkIndicesForRank& operator=(const MarkIndicesForRank&) = delete;
  MarkIndicesForRank& operator=(MarkIndicesForRank&&) = delete;
  ~MarkIndicesForRank() = default;

  bool fixedSize() { return true; }
  std::size_t size(int) { return 1; }

  template <class Buffer>
  void gather(Buffer& buffer, int i)
  {
    buffer.write({rank, mask[i]});
  }

  template <class Buffer>
  void scatter(Buffer& buffer, int i, int)
  {
    DataType d;
    buffer.read(d);

    if (not mask_from_rank.count(d.first)) mask_from_rank[d.first].resize(paridxs.size(), false); // Initialise masks with false

    mask_from_rank[d.first][i] = d.second;
  }

  std::map<int, std::vector<bool>> mask_from_rank;

private:
  std::vector<bool> mask;
  int rank;

  const ParallelIndexSet& paridxs;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;
};
