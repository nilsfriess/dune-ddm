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
#include <dune/istl/bvector.hh>
#include <format>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

/** @file
 *
 *  Tests for gatherMatrixFromRows and gatherMatrixFromRowsFlat. Everything is verified on rank 0,
 *  where the assembled matrix lives; the gathers themselves are collective and run on all ranks.
 */

namespace {

double at(const Dune::BCRSMatrix<double>& A, std::size_t i, std::size_t j) { return A[i][j]; }
double at(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>& A, std::size_t i, std::size_t j) { return A[i][j][0][0]; }

/// Describes the first entry differing from @p expected, empty if all match. Only the first is
/// reported so that a wrong matrix costs one line rather than one per entry.
template <class Mat, class Expected>
std::string firstBadEntry(const Mat& A, Expected expected)
{
  for (std::size_t i = 0; i < A.N(); ++i)
    for (std::size_t j = 0; j < A.M(); ++j) {
      const double want = expected(i, j);
      const double got = at(A, i, j);
      if (got != want) return std::format("entry ({},{}): expected {}, got {}", i, j, want, got);
    }
  return {};
}

/// Shape and contents of a dense matrix gathered on rank 0.
template <class Mat, class Expected>
void checkDense(Dune::TestSuite& t, const Mat& A, std::size_t rows, std::size_t cols, Expected expected, const std::string& name)
{
  const bool shape_ok = A.N() == rows and A.M() == cols;
  t.check(shape_ok, name + ": shape") << "expected " << rows << "x" << cols << ", got " << A.N() << "x" << A.M();
  if (!shape_ok) return;

  const auto bad = firstBadEntry(A, expected);
  t.check(bad.empty(), name + ": entries") << bad;
}

/// The clipping and zero-row cases all produce the same shape: rank i contributes a dense row 2i
/// holding the value i + 1, followed by an empty row 2i + 1.
template <class Mat>
void checkAlternatingRows(Dune::TestSuite& t, const Mat& A, std::size_t num_ranks, std::size_t cols, const std::string& name, bool check_values = true)
{
  const bool shape_ok = A.N() == 2 * num_ranks;
  t.check(shape_ok, name + ": row count") << "expected " << 2 * num_ranks << ", got " << A.N();
  if (!shape_ok) return;

  std::string bad;
  for (std::size_t i = 0; i < num_ranks and bad.empty(); ++i) {
    if (A.getrowsize(2 * i) != cols)
      bad = std::format("dense row {} has {} entries, expected {}", 2 * i, A.getrowsize(2 * i), cols);
    else if (A.getrowsize(2 * i + 1) != 0)
      bad = std::format("row {} should be empty, has {} entries", 2 * i + 1, A.getrowsize(2 * i + 1));
    else if (check_values)
      for (std::size_t c = 0; c < cols and bad.empty(); ++c) {
        const auto want = static_cast<double>(i + 1);
        const double got = at(A, 2 * i, c);
        if (got != want) bad = std::format("entry ({},{}): expected {}, got {}", 2 * i, c, want, got);
      }
  }
  t.check(bad.empty(), name + ": row pattern") << bad;
}

} // namespace

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  const auto rank = static_cast<std::size_t>(helper.rank());
  const auto num_ranks = static_cast<std::size_t>(helper.size());

  return ddmtest::runParallelTest("test_build_matrix", [&](Dune::TestSuite& t) {
    // Global row -> (owning rank, local row) for the 2-or-3-rows-per-rank layout used below.
    std::vector<std::pair<std::size_t, std::size_t>> layout;
    for (std::size_t i = 0; i < num_ranks; ++i)
      for (std::size_t r = 0; r < (i % 2 == 0 ? 2 : 3); ++r) layout.emplace_back(i, r);

    // Single row per rank. A negative clip tolerance keeps rank 0's exact zero.
    Dune::BlockVector<double> v(10);
    std::iota(v.begin(), v.end(), rank);

    auto A = gatherMatrixFromRows(v, MPI_COMM_WORLD, -1);

    if (rank == 0)
      checkDense(
          t, A, num_ranks, v.N(), [](std::size_t i, std::size_t j) { return static_cast<double>(i + j); }, "single row per rank");
    else
      t.check(A.N() == 0 and A.M() == 0, "matrix is empty off rank 0") << "got " << A.N() << "x" << A.M();

    // Multiple rows per rank. Filled with rank + 1 so nothing is dropped at clip tolerance 0.
    std::vector<Dune::BlockVector<double>> rows(rank % 2 == 0 ? 2 : 3, v);
    for (auto& row : rows) row = static_cast<double>(rank + 1);

    auto B = gatherMatrixFromRows(rows, MPI_COMM_WORLD, 0);

    if (rank == 0)
      checkDense(
          t, B, layout.size(), v.N(), [&](std::size_t i, std::size_t) { return static_cast<double>(layout[i].first + 1); }, "multiple rows per rank");

    // Clipping must not change the shape: each rank's second row is entirely below the tolerance
    // and must still occupy an (empty) row rather than shifting the following ranks up.
    std::vector<Dune::BlockVector<double>> clipped(2, v);
    clipped[0] = static_cast<double>(rank + 1);
    clipped[1] = 1e-12;

    auto C = gatherMatrixFromRows(clipped, MPI_COMM_WORLD, 1e-6);

    if (rank == 0) checkAlternatingRows(t, C, num_ranks, v.N(), "clipped rows");

    // gatherMatrixFromRowsFlat must agree with gatherMatrixFromRows. The shape is deliberately
    // non-square and the entries asymmetric in (row, col), so reading the flat input in the wrong
    // order does not go unnoticed.
    const std::size_t n_cols = 5;
    const std::size_t n_local = rank % 2 == 0 ? 2 : 3;
    const auto entry = [](std::size_t rk, std::size_t r, std::size_t c) { return static_cast<double>(100 * (rk + 1) + 10 * (r + 1) + (c + 1)); };

    std::vector<Dune::BlockVector<double>> nested(n_local, Dune::BlockVector<double>(n_cols));
    std::vector<double> flat(n_local * n_cols); // column major: flat[r + c * n_local]
    for (std::size_t r = 0; r < n_local; ++r)
      for (std::size_t c = 0; c < n_cols; ++c) {
        nested[r][c] = entry(rank, r, c);
        flat[r + c * n_local] = entry(rank, r, c);
      }

    auto D = gatherMatrixFromRows(nested, MPI_COMM_WORLD, 0);
    auto E = gatherMatrixFromRowsFlat(flat, n_cols, MPI_COMM_WORLD, 0);

    if (rank == 0) {
      const auto expected = [&](std::size_t i, std::size_t j) { return entry(layout[i].first, layout[i].second, j); };
      checkDense(t, D, layout.size(), n_cols, expected, "nested variant");
      checkDense(t, E, layout.size(), n_cols, expected, "flat variant");
      t.check(D.N() == E.N() and D.M() == E.M(), "flat and nested agree on the shape") << D.N() << "x" << D.M() << " vs " << E.N() << "x" << E.M();
    }

    // Same empty-row invariant as above, for the flat variant.
    std::vector<double> flat_clipped(2 * n_cols);
    for (std::size_t c = 0; c < n_cols; ++c) {
      flat_clipped[0 + c * 2] = static_cast<double>(rank + 1);
      flat_clipped[1 + c * 2] = 1e-12;
    }

    auto F = gatherMatrixFromRowsFlat(flat_clipped, n_cols, MPI_COMM_WORLD, 1e-6);

    if (rank == 0) checkAlternatingRows(t, F, num_ranks, n_cols, "clipped rows, flat variant");

    // Rows that are exactly zero are accepted by both variants and yield an empty row at clip
    // tolerance 0, the same as a fully clipped one.
    std::vector<Dune::BlockVector<double>> with_zero(2, Dune::BlockVector<double>(n_cols));
    with_zero[0] = static_cast<double>(rank + 1);
    with_zero[1] = 0;

    auto G = gatherMatrixFromRows(with_zero, MPI_COMM_WORLD, 0);

    if (rank == 0) checkAlternatingRows(t, G, num_ranks, n_cols, "zero rows");
  });
}
