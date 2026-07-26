#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_utils.hh"

#include <cstddef>
#include <dune/common/fmatrix.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/ddm/helpers.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <format>
#include <map>
#include <mpi.h>
#include <string>
#include <vector>

/** @file
 *
 *  Parallel tests for distributeMatrixFrom0: the union of the distributed owned rows must
 *  reproduce the matrix that rank 0 started with, and ranks left without rows must report
 *  themselves inactive.
 */

using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
using GlobalIndex = std::size_t;

namespace {

struct Entry {
  std::size_t row;
  std::size_t col;
  double val;
};

/// Owned rows of the local matrix, translated back to global indices and gathered on rank 0.
template <class Comm>
std::vector<Entry> gatherOwnedEntries(const Matrix& local, const Comm& comm, bool active, MPI_Comm mpiComm)
{
  std::vector<Entry> mine;

  if (active) {
    std::map<std::size_t, std::size_t> localToGlobal;
    std::vector<bool> owned(local.N(), false);
    for (const auto& idx : comm.indexSet()) {
      localToGlobal[idx.local().local()] = idx.global();
      if (idx.local().attribute() == Dune::OwnerOverlapCopyAttributeSet::owner) owned[idx.local().local()] = true;
    }

    for (auto ri = local.begin(); ri != local.end(); ++ri) {
      if (!owned[ri.index()]) continue;
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        const auto col = localToGlobal.find(ci.index());
        if (col == localToGlobal.end()) continue; // reported by the caller as a missing entry
        mine.push_back({localToGlobal.at(ri.index()), col->second, (*ci)[0][0]});
      }
    }
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(mpiComm, &rank);
  MPI_Comm_size(mpiComm, &size);

  const int my_count = static_cast<int>(mine.size() * sizeof(Entry));
  std::vector<int> counts(rank == 0 ? static_cast<std::size_t>(size) : 0);
  MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, mpiComm);

  std::vector<int> displs(rank == 0 ? static_cast<std::size_t>(size) : 0);
  std::size_t total_bytes = 0;
  if (rank == 0)
    for (int i = 0; i < size; ++i) {
      displs[i] = static_cast<int>(total_bytes);
      total_bytes += static_cast<std::size_t>(counts[i]);
    }

  std::vector<Entry> all(rank == 0 ? total_bytes / sizeof(Entry) : 0);
  MPI_Gatherv(mine.data(), my_count, MPI_BYTE, all.data(), counts.data(), displs.data(), MPI_BYTE, 0, mpiComm);
  return all;
}

/// Describes the first gathered entry that does not match the original, empty if all match.
std::string firstBadEntry(const Matrix& A, const std::vector<Entry>& gathered, std::vector<bool>& row_seen)
{
  for (const auto& e : gathered) {
    if (e.row >= A.N() or e.col >= A.M()) return std::format("entry ({},{}) is out of range", e.row, e.col);

    const auto ci = A[e.row].find(e.col);
    if (ci == A[e.row].end()) return std::format("entry ({},{}) is not in the original matrix", e.row, e.col);
    if ((*ci)[0][0] != e.val) return std::format("entry ({},{}): expected {}, got {}", e.row, e.col, (*ci)[0][0], e.val);

    row_seen[e.row] = true;
  }
  return {};
}

/// Distributes A over @p nparts and checks the owned rows reassemble into A on rank 0.
void checkRoundTrip(Dune::TestSuite& t, std::size_t nx, std::size_t ny, int nparts, int rank, const std::string& name)
{
  const Matrix A = rank == 0 ? ddmtest::buildLaplacian2d<Matrix>(nx, ny) : Matrix{};

  auto [comm, local, active] = distributeMatrixFrom0<Matrix, GlobalIndex>(A, MPI_COMM_WORLD, nparts);

  t.check(rank < nparts or !active, name + ": ranks beyond nparts are inactive") << "rank " << rank << " is active although only " << nparts << " parts were requested";
  t.check(active or local->N() == 0, name + ": inactive rank holds no rows") << "got " << local->N() << " rows";

  int local_active = active ? 1 : 0;
  int num_active = 0;
  MPI_Allreduce(&local_active, &num_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  t.check(num_active > 0 and num_active <= nparts, name + ": plausible number of active ranks") << num_active << " active, expected 1.." << nparts;

  const auto gathered = gatherOwnedEntries(*local, *comm, active, MPI_COMM_WORLD);

  if (rank != 0) return;

  const bool count_ok = gathered.size() == A.nonzeroes();
  t.check(count_ok, name + ": entry count") << "expected " << A.nonzeroes() << ", got " << gathered.size();
  if (!count_ok) return;

  std::vector<bool> row_seen(A.N(), false);
  const auto bad = firstBadEntry(A, gathered, row_seen);
  t.check(bad.empty(), name + ": entries match the original") << bad;
  if (!bad.empty()) return;

  const auto missing = std::find(row_seen.begin(), row_seen.end(), false);
  t.check(missing == row_seen.end(), name + ": every row is owned") << "row " << std::distance(row_seen.begin(), missing) << " is owned by no rank";
}

} // namespace

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  const int rank = helper.rank();
  const int size = helper.size();

  return ddmtest::runParallelTest("test_distribute_matrix", [&](Dune::TestSuite& t) {
    // Grid large enough that every rank gets rows.
    checkRoundTrip(t, 8, 8, size, rank, "all ranks active");

    // Fewer parts than ranks: the surplus ranks must come back inactive.
    if (size > 1) checkRoundTrip(t, 8, 8, size - 1, rank, "nparts < size");

    // nparts == 1 also selects the contiguous-block fallback instead of METIS.
    checkRoundTrip(t, 6, 6, 1, rank, "single part");
  });
}
