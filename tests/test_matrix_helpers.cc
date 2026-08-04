#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_utils.hh"

#include <cstddef>
#include <dune/common/fmatrix.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/ddm/matrix_helpers.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <numeric>
#include <string>
#include <vector>

/** @file
 *
 *  Sequential tests for matrix_helpers.hh.
 *
 *  extract_submatrix() is checked against a naive reference that looks every entry up
 *  individually with BCRSMatrix::exists(), so the fast three-pass construction has to
 *  reproduce both the sparsity pattern and the values of A[rows, cols].
 *
 *  The debug-only preconditions (cols strictly increasing, indices in range) are not
 *  exercised here: they are assert()s, so violating them aborts the process rather than
 *  throwing something a test could catch.
 */

namespace {

/** @brief Checks extract_submatrix(A, rows, cols) against an entry-by-entry reference.
 *
 * Verifies the shape, the sparsity pattern in both directions (nothing stored that the
 * reference does not have, and the counts agree so nothing is missing), the values, and
 * that the column indices of every row come out sorted -- the latter is what
 * setIndicesNoSort() assumes and would otherwise silently violate.
 */
template <class Mat>
void checkExtraction(Dune::TestSuite& t, const Mat& A, const std::vector<std::size_t>& rows, const std::vector<std::size_t>& cols, const std::string& what)
{
  const Mat B = ddm::extract_submatrix(A, rows, cols);

  const bool shape_ok = B.N() == rows.size() and B.M() == cols.size();
  t.check(shape_ok, what + ": shape") << "expected " << rows.size() << "x" << cols.size() << ", got " << B.N() << "x" << B.M();
  if (not shape_ok) return;

  for (std::size_t i = 0; i < rows.size(); ++i) {
    // Everything B stores in row i must exist in A at the corresponding global position,
    // carry the same value, and appear in increasing column order.
    std::size_t stored = 0;
    std::size_t previous = 0;
    bool sorted = true;
    for (auto entry = B[i].begin(); entry != B[i].end(); ++entry) {
      const std::size_t j = entry.index();
      if (stored > 0 and j <= previous) sorted = false;
      previous = j;
      ++stored;

      if (not A.exists(rows[i], cols[j])) {
        t.check(false, what + ": spurious entry") << "B(" << i << "," << j << ") = A(" << rows[i] << "," << cols[j] << ") is not stored in A";
        continue;
      }
      const auto difference = (*entry - A[rows[i]][cols[j]]).infinity_norm();
      t.check(difference == 0.0, what + ": value") << "B(" << i << "," << j << ") differs from A(" << rows[i] << "," << cols[j] << ") by " << difference;
    }
    t.check(sorted, what + ": column indices sorted") << "row " << i << " of B is not strictly increasing";

    // ... and conversely nothing may be missing. Counting suffices given the check above.
    std::size_t expected = 0;
    for (std::size_t j = 0; j < cols.size(); ++j)
      if (A.exists(rows[i], cols[j])) ++expected;
    t.check(stored == expected, what + ": row size") << "row " << i << " has " << stored << " entries, expected " << expected;
  }
}

//! [begin, end)
std::vector<std::size_t> range(std::size_t begin, std::size_t end)
{
  std::vector<std::size_t> result(end - begin);
  std::iota(result.begin(), result.end(), begin);
  return result;
}

/** The 4x3 Neumann Laplacian gives a 12x12 matrix whose pattern is irregular enough
 * (corner/edge/interior rows have 3/4/5 entries) that a submatrix genuinely has to drop
 * entries in the middle of a row, not just at its ends.
 */
void checkOnLaplacian(Dune::TestSuite& t)
{
  using Mat = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  const auto A = ddmtest::buildLaplacian2d<Mat>(4, 3);

  const auto all = range(0, 12);

  checkExtraction(t, A, all, all, "identity extraction");
  checkExtraction(t, A, range(0, 6), range(0, 6), "leading square block");
  checkExtraction(t, A, range(6, 12), range(6, 12), "trailing square block");

  // Scattered columns: forces the global -> local map to be non-contiguous, so the local
  // indices are no longer just a shift of the global ones.
  checkExtraction(t, A, all, {0, 3, 4, 7, 8, 11}, "scattered columns");
  checkExtraction(t, A, {1, 2, 5, 9}, {0, 2, 4, 6, 8, 10}, "scattered rows and columns");

  // Rectangular, including the degenerate one-row and one-column cases.
  checkExtraction(t, A, range(0, 4), all, "rectangular: few rows");
  checkExtraction(t, A, all, range(0, 4), "rectangular: few columns");
  checkExtraction(t, A, {5}, all, "single row");
  checkExtraction(t, A, all, {5}, "single column");
  checkExtraction(t, A, {5}, {5}, "single entry");

  // The rows may be given in any order and may repeat; only cols is constrained.
  checkExtraction(t, A, {11, 0, 7, 3}, range(0, 12), "permuted rows");
  checkExtraction(t, A, {4, 4, 4}, range(0, 12), "repeated rows");
  checkExtraction(t, A, {9, 1, 1, 5, 9}, {2, 5, 6, 9}, "permuted and repeated rows");

  // A column selection that misses every off-diagonal neighbour of the selected rows
  // leaves only the diagonal, i.e. rows of length one.
  checkExtraction(t, A, {0, 2}, {0, 2}, "diagonal-only submatrix");

  // Rows that select nothing at all: the two blocks of a 2x2 splitting have empty
  // off-diagonal-only extractions here only if the sets are non-adjacent, so use the
  // stricter case of an empty column set.
  checkExtraction(t, A, all, {}, "no columns");
  checkExtraction(t, A, {}, all, "no rows");
  checkExtraction(t, A, {}, {}, "empty extraction");
}

/** Repeats a representative subset with 2x2 blocks. Each block is filled with four
 * distinct values so that a block copied transposed or offset would be caught.
 */
void checkOnBlockMatrix(Dune::TestSuite& t)
{
  using Block = Dune::FieldMatrix<double, 2, 2>;
  using Mat = Dune::BCRSMatrix<Block>;

  constexpr std::size_t n = 6;
  Mat A;
  A.setBuildMode(Mat::implicit);
  A.setImplicitBuildModeParameters(3, 0.3);
  A.setSize(n, n);

  const auto fill = [](std::size_t i, std::size_t j) {
    const double base = 100.0 * static_cast<double>(i) + static_cast<double>(j);
    return Block{{base + 0.1, base + 0.2}, {base + 0.3, base + 0.4}};
  };

  // Tridiagonal, so that dropping a column really punches holes into the rows.
  for (std::size_t i = 0; i < n; ++i) {
    if (i > 0) A.entry(i, i - 1) = fill(i, i - 1);
    A.entry(i, i) = fill(i, i);
    if (i + 1 < n) A.entry(i, i + 1) = fill(i, i + 1);
  }
  A.compress();

  const auto all = range(0, n);
  checkExtraction(t, A, all, all, "blocks: identity extraction");
  checkExtraction(t, A, all, {0, 2, 4}, "blocks: every other column");
  checkExtraction(t, A, {1, 3}, {0, 1, 2, 3}, "blocks: rectangular");
  checkExtraction(t, A, {5, 5, 0}, {1, 4, 5}, "blocks: repeated rows");
}

/** The always-on preconditions. Unlike the debug-only index checks these throw, because
 * they are O(1) and because getting them wrong means reading a matrix that is not there.
 */
void checkPreconditions(Dune::TestSuite& t)
{
  using Mat = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;

  const auto expect_throw = [&t](auto&& call, const std::string& name) {
    bool threw = false;
    try {
      call();
    }
    catch (const Dune::Exception&) {
      threw = true;
    }
    t.check(threw, name) << "expected a Dune::Exception, none was thrown";
  };

  // A default-constructed matrix has not allocated anything, so its rows must not be read.
  expect_throw(
      [] {
        const Mat unbuilt;
        return ddm::extract_submatrix(unbuilt, {0}, {0});
      },
      "unbuilt matrix throws");

  // A matrix still in its build phase is equally unsafe to index.
  expect_throw(
      [] {
        Mat half_built;
        half_built.setBuildMode(Mat::random);
        half_built.setSize(3, 3);
        return ddm::extract_submatrix(half_built, {0}, {0});
      },
      "half-built matrix throws");

  const auto A = ddmtest::buildLaplacian2d<Mat>(2, 2); // 4x4
  expect_throw([&A] { return ddm::extract_submatrix(A, range(0, 5), range(0, 4)); }, "too many rows throws");
  expect_throw([&A] { return ddm::extract_submatrix(A, range(0, 4), range(0, 5)); }, "too many columns throws");
}

} // namespace

int main(int argc, char** argv)
{
  Dune::MPIHelper::instance(argc, argv);

  return ddmtest::runParallelTest("test_matrix_helpers", [](Dune::TestSuite& t) {
    checkOnLaplacian(t);
    checkOnBlockMatrix(t);
    checkPreconditions(t);
  });
}
