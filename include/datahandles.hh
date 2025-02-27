#pragma once

#include <dune/common/parallel/indexset.hh>

#include <cstddef>
#include <map>
#include <set>
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

template <class Mat, class ParallelIndexSet>
class IdentifyBoundaryDataHandle {
public:
  IdentifyBoundaryDataHandle(const Mat &A, const ParallelIndexSet &paridxs) : boundaryMask(paridxs.size(), false), paridxs{paridxs}, A{A}, glis{paridxs} {}
  IdentifyBoundaryDataHandle(const IdentifyBoundaryDataHandle &) = delete;
  IdentifyBoundaryDataHandle(IdentifyBoundaryDataHandle &&) = delete;
  IdentifyBoundaryDataHandle &operator=(const IdentifyBoundaryDataHandle &) = delete;
  IdentifyBoundaryDataHandle &operator=(IdentifyBoundaryDataHandle &&) = delete;
  ~IdentifyBoundaryDataHandle() = default;

  using DataType = typename ParallelIndexSet::GlobalIndex;

  bool fixedSize() { return false; }

  std::size_t size(int i)
  {
    int count = 0;
    for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
      if (cit.index() != i) {
        ++count;
      }
    }
    return count;
  }

  template <class B>
  void gather(B &buffer, int i)
  {
    for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
      if (cit.index() != i) {
        buffer.write(glis.pair(cit.index())->global());
      }
    }
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int size)
  {
    DataType gi;
    for (int k = 0; k < size; k++) {
      buffer.read(gi);
      if (not paridxs.exists(gi)) { // If we don't know the received global index, i is a boundary index
        boundaryMask[i] = true;
      }
    }
  }

  std::vector<bool> &&extractBoundaryMask() { return std::move(boundaryMask); }

private:
  std::vector<bool> boundaryMask;
  const ParallelIndexSet &paridxs;
  const Mat &A;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;
};

class RankDataHandle {
public:
  using DataType = int;

  RankDataHandle(int rank, std::map<int, std::set<int>> &neighbours_for_index) : rank(rank), neighbours_for_index(neighbours_for_index) {}

  RankDataHandle(const RankDataHandle &) = delete;
  RankDataHandle(RankDataHandle &&) = delete;
  RankDataHandle &operator=(const RankDataHandle &) = delete;
  RankDataHandle &operator=(RankDataHandle &&) = delete;
  ~RankDataHandle() = default;

  bool fixedSize() { return true; }
  std::size_t size(int) { return 1; }

  template <class Buffer>
  void gather(Buffer &buffer, int)
  {
    buffer.write(rank);
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int)
  {
    int otherrank = 0;
    buffer.read(otherrank);
    neighbours_for_index[i].insert(otherrank);
  }

private:
  int rank;
  std::map<int, std::set<int>> &neighbours_for_index;
};

// TODO: Merge this with the implementation in datahandles.hh
class SumDataHandle {
public:
  using DataType = int;

  explicit SumDataHandle(std::vector<int> &target) : source(target), target(target) {}
  SumDataHandle(const SumDataHandle &) = delete;
  SumDataHandle(SumDataHandle &&) = delete;
  SumDataHandle &operator=(const SumDataHandle &) = delete;
  SumDataHandle &operator=(SumDataHandle &&) = delete;
  ~SumDataHandle() = default;

  bool fixedSize() { return true; }
  std::size_t size(int) { return 1; }

  template <class Buffer>
  void gather(Buffer &buffer, int i)
  {
    buffer.write(source[i]);
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int)
  {
    int value{};
    buffer.read(value);
    target[i] += value;
  }

private:
  std::vector<int> source;
  std::vector<int> &target;
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
    count += 1;                                    // The number of global indices and connected ranks we send for this local index

    if (connected_indices.contains(i)) {
      for (const auto &idx : connected_indices.at(i)) {
        count += 1;                                // The actual global index

        count += 1;                                // The number of ranks that also know this index
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

    for (int i = 0; i < data; ++i) {
      DataType gi;
      buffer.read(gi);
      gis.insert(gi);

      // Read number of connected ranks
      DataType count;
      buffer.read(count);

      for (int j = 0; j < count; ++j) {
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
  CreateMatrixDataHandle(const Mat &A, const ParallelIndexSet &paridxs) : A(A), glis(paridxs), paridxs(paridxs), Aovlp(paridxs.size(), paridxs.size(), static_cast<double>(A.nonzeroes()) / A.N(), 0.5, Mat::implicit)
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
    if (i < A.N()) { // Only if the index is part of the original matrix
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
    if (i < A.N()) {
      for (auto cIt = A[i].begin(); cIt != A[i].end(); ++cIt) {
        buffer.write(glis.pair(cIt.index())->global());
      }
    }
  }

  template <class Buffer>
  void scatter(Buffer &buffer, int i, int size)
  {
    DataType gi;
    buffer.read(gi);   // read dummy data
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

  AddMatrixDataHandle(const Mat &A, Mat &Aovlp, const ParallelIndexSet &paridxs) : Asource(A), Atarget(Aovlp), paridxs(paridxs), glis(paridxs), dirichlet_rows(A.N(), true)
  {
    for (auto rIt = A.begin(); rIt != A.end(); ++rIt) {
      for (auto cIt = rIt->begin(); cIt != rIt->end(); ++cIt) {
        Aovlp[rIt.index()][cIt.index()] = *cIt;

        if (cIt.index() == rIt.index()) {
          dirichlet_rows[rIt.index()] = dirichlet_rows[rIt.index()] & (*cIt == 1.0);
        }
        else {
          dirichlet_rows[rIt.index()] = dirichlet_rows[rIt.index()] & (*cIt == 0.0);
        }
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
    if (i < Asource.N()) {
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
    if (i < Asource.N()) {
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
        if (i < Asource.N() && dirichlet_rows[i]) {
          continue; // Don't modify Dirichlet rows
        }

        Atarget[i][paridxs[x.first].local()] += x.second;
      }
    }
  }

private:
  const Mat &Asource;
  Mat &Atarget;

  const ParallelIndexSet &paridxs;
  Dune::GlobalLookupIndexSet<ParallelIndexSet> glis;

  std::vector<bool> dirichlet_rows;
};