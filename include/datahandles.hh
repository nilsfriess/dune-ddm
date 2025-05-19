#pragma once

#include <dune/common/parallel/indexset.hh>

#include <cstddef>
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
  void gather(Buffer &buffer, int i)
  {
    buffer.write((*vec)[i]);
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int)
  {
    DataType data;
    buffer.read(data);
    (*vec)[i] = data;
  }

  void setVec(Vec &v) { vec = &v; }

private:
  Vec *vec;
};

template <class Vec>
class AddVectorDataHandle {
public:
  using DataType = typename Vec::block_type;

  bool fixedSize() { return true; }
  std::size_t size(int) { return 1; }

  template <class Buffer>
  void gather(Buffer &buffer, int i)
  {
    buffer.write(sourcevec[i]);
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int)
  {
    DataType data;
    buffer.read(data);
    (*targetvec)[i] += data;
  }

  void setVec(Vec &v)
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
  Vec *targetvec;
};

template <class Vec>
class CopyVectorDataHandleWithRank {
public:
  using DataType = typename Vec::block_type;

  explicit CopyVectorDataHandleWithRank(const Vec &v) : sourcevec(&v) {}

  bool fixedSize() { return true; }
  std::size_t size(int) { return 1; }

  template <class Buffer>
  void gather(Buffer &buffer, int i)
  {
    buffer.write((*sourcevec)[i]);
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int, int from_rank)
  {
    DataType data;
    buffer.read(data);
    if (not copied_vecs.contains(from_rank)) {
      copied_vecs.emplace(from_rank, *sourcevec);
      copied_vecs[from_rank] = 0;
    }
    copied_vecs[from_rank][i] = data;
  }

  std::map<int, Vec> copied_vecs;

private:
  const Vec *sourcevec;
};

template <class Mat, class ParallelIndexSet>
class IdentifyBoundaryDataHandle {
public:
  IdentifyBoundaryDataHandle(const Mat &A, const ParallelIndexSet &paridxs) : boundary_mask(paridxs.size(), false), paridxs{paridxs}, A{A}, glis{paridxs} {}
  IdentifyBoundaryDataHandle(const IdentifyBoundaryDataHandle &) = delete;
  IdentifyBoundaryDataHandle(IdentifyBoundaryDataHandle &&) = delete;
  IdentifyBoundaryDataHandle &operator=(const IdentifyBoundaryDataHandle &) = delete;
  IdentifyBoundaryDataHandle &operator=(IdentifyBoundaryDataHandle &&) = delete;
  ~IdentifyBoundaryDataHandle() = default;

  using DataType = typename ParallelIndexSet::GlobalIndex;

  bool fixedSize() { return false; }

  std::size_t size(int i)
  {
    int count = 1; // send at least a dummy value
    for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
      if (cit.index() != static_cast<std::size_t>(i)) {
        ++count;
      }
    }
    return count;
  }

  template <class B>
  void gather(B &buffer, int i)
  {
    buffer.write(0);
    for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
      if (cit.index() != static_cast<std::size_t>(i)) {
        buffer.write(glis.pair(cit.index())->global());
      }
    }
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int size, int remoterank)
  {
    DataType gi;
    buffer.read(gi); // Read the dummy value

    boundaryMaskForRank[remoterank].resize(paridxs.size(), false);

    for (int k = 1; k < size; k++) {
      buffer.read(gi);
      if (not paridxs.exists(gi)) { // If we don't know the received global index, i is a boundary index
        boundaryMaskForRank[remoterank][i] = true;
        boundary_mask[i] = true;
      }
    }
  }

  const std::vector<bool> &get_boundary_mask() const
  {
    return boundary_mask;
    // std::vector<bool> boundaryMask(paridxs.size(), false);
    // for (const auto &[rank, mask] : boundaryMaskForRank) {
    //   for (std::size_t i = 0; i < boundaryMask.size(); ++i) {
    //     boundaryMask[i] = boundaryMask[i] || mask[i];
    //   }
    // }
    // return boundaryMask;
  }

  const std::map<int, std::vector<bool>> &getBoundaryMaskForRank() const { return boundaryMaskForRank; }

private:
  std::map<int, std::vector<bool>> boundaryMaskForRank;
  std::vector<bool> boundary_mask;
  const ParallelIndexSet &paridxs;
  const Mat &A;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;
};

struct RankTuple {
  using value_type = int;

  std::vector<std::set<int>> rankmap; // TODO: Check if a map or a vector is more efficient here
  int rank;
};

class RankDataHandle {
public:
  using DataType = int;

  static int gather(const RankTuple &rt, int) { return rt.rank; }
  static void scatter(RankTuple &rt, int otherrank, std::size_t i) { rt.rankmap[i].insert(otherrank); }
};

template <class Mat, class GlobalIndex>
class IndexsetExtensionMatrixGraphDataHandle {
public:
  using DataType = GlobalIndex;

  IndexsetExtensionMatrixGraphDataHandle(int rank, const Mat &A, std::vector<GlobalIndex> &ltg, std::unordered_set<GlobalIndex> &gis) : rank(rank), A(A), ltg(ltg), gis(gis) {}

  IndexsetExtensionMatrixGraphDataHandle(const IndexsetExtensionMatrixGraphDataHandle &) = delete;
  IndexsetExtensionMatrixGraphDataHandle(IndexsetExtensionMatrixGraphDataHandle &&) = delete;
  IndexsetExtensionMatrixGraphDataHandle &operator=(const IndexsetExtensionMatrixGraphDataHandle &) = delete;
  IndexsetExtensionMatrixGraphDataHandle &operator=(IndexsetExtensionMatrixGraphDataHandle &&) = delete;
  ~IndexsetExtensionMatrixGraphDataHandle() = default;

  bool fixedSize() { return false; }

  std::size_t size(std::size_t i)
  {
    std::size_t count = 1;
    if (i < A.N()) {
      for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
        if (cit.index() != i) {
          count++;
        }
      }
    }
    return count;
  }

  template <class Buffer>
  void gather(Buffer &b, std::size_t i)
  {
    b.write(0);
    if (i < A.N()) {
      for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
        if (cit.index() != i) {
          b.write(ltg[cit.index()]);
          for (const auto &p : rankmap[i]) {
            updated_rankmap[cit.index()].insert(p);
          }
        }
      }
    }
  }

  template <class Buffer>
  void scatter(Buffer &b, std::size_t, std::size_t size)
  {
    GlobalIndex gi;
    b.read(gi);
    for (std::size_t k = 1; k < size; ++k) {
      b.read(gi);
      if (not gis.contains(gi)) {
        ltg.push_back(gi);
        gis.insert(gi);
      }
    }
  }

  std::vector<std::set<int>> rankmap;
  std::vector<std::set<int>> updated_rankmap;

private:
  int rank;
  const Mat &A;
  std::vector<GlobalIndex> &ltg;
  std::unordered_set<GlobalIndex> &gis;
};

class UpdateRankInfoDataHandle {
public:
  explicit UpdateRankInfoDataHandle(std::vector<std::set<int>> &rankmap) : rankmap(rankmap) {}

  UpdateRankInfoDataHandle(const UpdateRankInfoDataHandle &) = delete;
  UpdateRankInfoDataHandle(UpdateRankInfoDataHandle &&) = delete;
  UpdateRankInfoDataHandle &operator=(const UpdateRankInfoDataHandle &) = delete;
  UpdateRankInfoDataHandle &operator=(UpdateRankInfoDataHandle &&) = delete;
  ~UpdateRankInfoDataHandle() = default;

  using DataType = int;

  bool fixedSize() { return false; }
  std::size_t size(std::size_t i) { return 1 + rankmap[i].size(); }

  template <class Buffer>
  void gather(Buffer &buffer, std::size_t i)
  {
    buffer.write(0);
    for (auto &&r : rankmap[i]) {
      buffer.write(r);
    }
  }

  template <class Buffer>
  void scatter(Buffer &buffer, std::size_t i, std::size_t size)
  {
    int curr_r = -1;
    buffer.read(curr_r);
    for (std::size_t r = 1; r < size; ++r) {
      buffer.read(curr_r);
      rankmap[i].insert(curr_r);
    }
  }

private:
  std::vector<std::set<int>> &rankmap;
};

/* A data handle to exchange global indices */
template <class Mat, class ParallelIndexSet>
class DataHandle {
public:
  DataHandle(const Mat &A, const ParallelIndexSet &paridxs, const std::map<int, std::set<int>> &connected_indices, const std::map<int, std::set<int>> &connected_ranks, int rank)
      : A(A), glis(paridxs), paridxs(paridxs), connected_indices(connected_indices), connected_ranks(connected_ranks), rank(rank)
  {
  }
  DataHandle(const DataHandle &) = delete;
  DataHandle(DataHandle &&) = delete;
  DataHandle &operator=(const DataHandle &) = delete;
  DataHandle &operator=(DataHandle &&) = delete;
  ~DataHandle() = default;

  using DataType = typename ParallelIndexSet::GlobalIndex;

  bool fixedSize() { return false; }
  std::size_t size(int i)
  {
    // assert(connected_indices.count(i) > 0);

    std::size_t count = 0;
    count += 1; // The number of global indices and connected ranks we send for this local index

    if (connected_indices.contains(i)) {
      for (const auto &idx : connected_indices.at(i)) {
        count += 1; // The actual global index

        count += 1; // The number of ranks that also know this index
        if (connected_ranks.contains(idx)) {
          count += connected_ranks.at(idx).size(); // The ranks
        }
      }
    }

    return count;
  }

  template <class Buffer>
  void gather(Buffer &buffer, int i)
  {
    if (connected_indices.contains(i)) {
      buffer.write(connected_indices.at(i).size());
      for (const auto &li : connected_indices.at(i)) {
        buffer.write(glis.pair(li)->global());

        if (connected_ranks.contains(li)) {
          buffer.write(connected_ranks.at(li).size());
          for (const auto &r : connected_ranks.at(li)) {
            buffer.write(r);
          }
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
  void scatter(Buffer &buffer, int, int)
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
  const Mat &A;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;
  const ParallelIndexSet &paridxs;
  const std::map<int, std::set<int>> &connected_indices;
  const std::map<int, std::set<int>> &connected_ranks;
  int rank;
};

template <class Mat, class ParallelIndexSet>
class CreateMatrixDataHandle {
public:
  CreateMatrixDataHandle(const Mat &A, const ParallelIndexSet &paridxs)
      : A(A), glis(paridxs), paridxs(paridxs), Aovlp(paridxs.size(), paridxs.size(), static_cast<double>(A.nonzeroes()) / A.N(), 0.6, Mat::implicit)
  {
    for (auto rIt = A.begin(); rIt != A.end(); ++rIt) {
      for (auto cIt = rIt->begin(); cIt != rIt->end(); ++cIt) {
        Aovlp.entry(rIt.index(), cIt.index()) = 0.0;
      }
    }
  }
  CreateMatrixDataHandle(const CreateMatrixDataHandle &) = delete;
  CreateMatrixDataHandle(CreateMatrixDataHandle &&) = delete;
  CreateMatrixDataHandle &operator=(const CreateMatrixDataHandle &) = delete;
  CreateMatrixDataHandle &operator=(CreateMatrixDataHandle &&) = delete;
  ~CreateMatrixDataHandle() = default;

  using DataType = typename ParallelIndexSet::GlobalIndex;

  bool fixedSize() { return false; }
  std::size_t size(int i)
  {
    std::size_t count = 1;
    if (static_cast<std::size_t>(i) < A.N()) { // Only if the index is part of the original matrix
      for (auto cIt = A[i].begin(); cIt != A[i].end(); ++cIt) {
        count++;
      }
    }
    return count;
  }

  template <class Buffer>
  void gather(Buffer &buffer, int i)
  {
    buffer.write(1); // send dummy data
    if (static_cast<std::size_t>(i) < A.N()) {
      for (auto cIt = A[i].begin(); cIt != A[i].end(); ++cIt) {
        buffer.write(glis.pair(cIt.index())->global());
      }
    }
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int size)
  {
    DataType gi;
    buffer.read(gi); // read dummy data
    for (int k = 0; k < size - 1; k++) {
      buffer.read(gi); // read global index
      if (paridxs.exists(gi)) {
        Aovlp.entry(i, paridxs[gi].local()) = 0.0;
      }
    }
  }

  Mat &&getOverlappingMatrix()
  {
    Aovlp.compress();
    return std::move(Aovlp);
  }

private:
  const Mat &A;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;
  const ParallelIndexSet &paridxs;

  Mat Aovlp;
};

template <class Mat, class ParallelIndexSet>
class AddMatrixDataHandle {
  using GlobalIndex = typename ParallelIndexSet::GlobalIndex;
  using MatrixEntry = typename Mat::block_type;

public:
  using DataType = std::pair<GlobalIndex, MatrixEntry>;

  AddMatrixDataHandle(const Mat &A, Mat &Aovlp, const ParallelIndexSet &paridxs) : Asource(A), Atarget(Aovlp), paridxs(paridxs), glis(paridxs)
  {
    for (auto rIt = A.begin(); rIt != A.end(); ++rIt) {
      for (auto cIt = rIt->begin(); cIt != rIt->end(); ++cIt) {
        Atarget[rIt.index()][cIt.index()] = Asource[rIt.index()][cIt.index()];
      }
    }
  }
  AddMatrixDataHandle(const AddMatrixDataHandle &) = delete;
  AddMatrixDataHandle(AddMatrixDataHandle &&) = delete;
  AddMatrixDataHandle &operator=(const AddMatrixDataHandle &) = delete;
  AddMatrixDataHandle &operator=(AddMatrixDataHandle &&) = delete;
  ~AddMatrixDataHandle() = default;

  bool fixedSize() { return false; }

  std::size_t size(int i)
  {
    std::size_t count = 1;
    if (static_cast<std::size_t>(i) < Asource.N()) {
      for (auto cIt = Asource[i].begin(); cIt != Asource[i].end(); ++cIt) {
        count++;
      }
    }
    return count;
  }

  template <class Buffer>
  void gather(Buffer &buffer, int i)
  {
    buffer.write(DataType(0, 0));
    if (static_cast<std::size_t>(i) < Asource.N()) {
      for (auto cIt = Asource[i].begin(); cIt != Asource[i].end(); ++cIt) {
        buffer.write(DataType(glis.pair(cIt.index())->global(), *cIt));
      }
    }
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int size)
  {
    DataType x;
    buffer.read(x);
    for (int k = 1; k < size; k++) {
      buffer.read(x);
      if (paridxs.exists(x.first)) {
        Atarget[i][paridxs[x.first].local()] += x.second;
      }
    }
  }

private:
  const Mat &Asource;
  Mat &Atarget;

  const ParallelIndexSet &paridxs;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;
};

template <class ParallelIndexSet>
class MarkIndicesForRank {
  using GlobalIndex = typename ParallelIndexSet::GlobalIndex;

public:
  using DataType = std::pair<int, bool>; // Our rank and whether the index is marked or not

  MarkIndicesForRank(const std::vector<std::size_t> &marked_indices, int rank, const ParallelIndexSet &paridxs) : mask(paridxs.size(), false), rank(rank), paridxs(paridxs), glis(paridxs)
  {
    for (auto i : marked_indices) {
      mask[i] = true;
    }
  }
  MarkIndicesForRank(const MarkIndicesForRank &) = delete;
  MarkIndicesForRank(MarkIndicesForRank &&) = delete;
  MarkIndicesForRank &operator=(const MarkIndicesForRank &) = delete;
  MarkIndicesForRank &operator=(MarkIndicesForRank &&) = delete;
  ~MarkIndicesForRank() = default;

  bool fixedSize() { return true; }
  std::size_t size(int) { return 1; }

  template <class Buffer>
  void gather(Buffer &buffer, int i)
  {
    buffer.write({rank, mask[i]});
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int)
  {
    DataType d;
    buffer.read(d);

    if (not mask_from_rank.contains(d.first)) {
      mask_from_rank[d.first].resize(paridxs.size(), false); // Initialise masks with false
    }

    mask_from_rank[d.first][i] = d.second;
  }

  std::map<int, std::vector<bool>> mask_from_rank;

private:
  std::vector<bool> mask;
  int rank;

  const ParallelIndexSet &paridxs;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;
};
