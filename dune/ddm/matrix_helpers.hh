#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <dune/common/exceptions.hh>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

namespace ddm {
/* @brief Extract a submatrix A[rows, cols] from a Dune::BCRSMatrix
 *
 * \p cols must be strictly increasing (sorted and free of duplicates); that is what makes
 * the local column indices come out sorted and lets us use setIndicesNoSort() below.
 * \p rows has no such requirement: row i of the result is built independently from
 * A[rows[i]], so the rows may be given in any order and may even repeat.
 *
 * In debug builds the requirements on \p rows and \p cols are checked; with NDEBUG only
 * the O(1) preconditions are.
 */
template <class Matrix>
Matrix extract_submatrix(const Matrix& A, const std::vector<std::size_t>& rows, const std::vector<std::size_t>& cols)
{
  static_assert(std::is_integral_v<std::size_t>, "extract_submatrix needs an integral index type");

  if (A.buildStage() != Matrix::built) DUNE_THROW(Dune::Exception, "extract_submatrix needs a fully built matrix");
  if (rows.size() > A.N()) DUNE_THROW(Dune::Exception, "Asked for " << rows.size() << " rows, but the matrix only has " << A.N());
  if (cols.size() > A.M()) DUNE_THROW(Dune::Exception, "Asked for " << cols.size() << " columns, but the matrix only has " << A.M());

  // Out-of-range column indices would corrupt the heap in the global -> local map below,
  // and unsorted or duplicate ones would silently produce a broken BCRSMatrix, so these
  // are worth the O(n) scan whenever assertions are enabled.
  assert(std::is_sorted(cols.begin(), cols.end(), std::less_equal<std::size_t>{}) && "cols must be strictly increasing");
  assert(std::all_of(rows.begin(), rows.end(), [&](std::size_t r) { return static_cast<std::size_t>(r) < A.N(); }) && "row index out of range");
  assert(std::all_of(cols.begin(), cols.end(), [&](std::size_t c) { return static_cast<std::size_t>(c) < A.M(); }) && "column index out of range");

  Matrix B;

  // The column indices within a BCRS row are sorted, and rows is sorted as well, so
  // the global -> local map is monotone and the local column indices come out sorted
  // for free. That lets us build B in random mode and hand the indices over with
  // setIndicesNoSort(), which is a plain copy, and it lets the value copy below run in
  // lockstep over both rows instead of looking up every entry individually.
  constexpr auto invalid = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> cols_global_to_local(A.M(), invalid);
  for (std::size_t i = 0; i < cols.size(); ++i) cols_global_to_local[cols[i]] = i;

  B.setBuildMode(Matrix::random);
  B.setSize(rows.size(), cols.size());

  // Pass 1: row sizes
  std::size_t max_row_size = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::size_t row_size = 0;
    const auto& row = A[rows[i]];
    for (auto ci = row.begin(); ci != row.end(); ++ci)
      if (cols_global_to_local[ci.index()] != invalid) ++row_size;
    B.setrowsize(i, row_size);
    max_row_size = std::max(max_row_size, row_size);
  }
  B.endrowsizes();

  // Pass 2: column indices
  std::vector<std::size_t> col_indices;
  col_indices.reserve(max_row_size);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    col_indices.clear();
    const auto& row = A[rows[i]];
    for (auto ci = row.begin(); ci != row.end(); ++ci) {
      const auto j = cols_global_to_local[ci.index()];
      if (j != invalid) col_indices.push_back(j);
    }
    B.setIndicesNoSort(i, col_indices.begin(), col_indices.end());
  }
  B.endindices();

  // Pass 3: values
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& row = A[rows[i]];
    auto target = B[i].begin();
    for (auto ci = row.begin(); ci != row.end(); ++ci)
      if (cols_global_to_local[ci.index()] != invalid) {
        *target = *ci;
        ++target;
      }
    // Pass 1 counted exactly the entries pass 3 copies; if that ever diverges, the
    // writes above run past the end of the row.
    assert(target == B[i].end());
  }

  return B;
}

/** \brief Row-sum lumping: every row of Q is collapsed onto its diagonal entry.
 *
 * For a mass matrix this gives the diagonal matrix D_ii = int phi_i (times the weight of
 * the mass matrix), because the basis functions form a partition of unity. For the P1/P2
 * spaces used in the examples that is strictly positive, so D is invertible.
 */
template <class Matrix>
Matrix lump_row_sum(const Matrix& Q)
{
  Matrix D;
  D.setBuildMode(Matrix::random);
  D.setSize(Q.N(), Q.M());

  for (std::size_t i = 0; i < Q.N(); ++i) D.setrowsize(i, 1);
  D.endrowsizes();
  for (std::size_t i = 0; i < Q.N(); ++i) D.addindex(i, i);
  D.endindices();

  for (auto row = Q.begin(); row != Q.end(); ++row) {
    auto& target = D[row.index()][row.index()];
    target = 0;
    for (auto entry = row->begin(); entry != row->end(); ++entry) target += *entry;
  }

  return D;
}

/** \brief Inverts a diagonal matrix, block by block.
 *
 * Only the diagonal blocks are looked at, so this is the inverse of \p D if and only if
 * \p D really is (block-)diagonal -- which is what lump_row_sum() produces.
 */
template <class Matrix>
Matrix invert_diagonal(const Matrix& D)
{
  if (D.N() != D.M()) DUNE_THROW(Dune::Exception, "invert_diagonal needs a square matrix, got " << D.N() << "x" << D.M());

  Matrix R;
  R.setBuildMode(Matrix::random);
  R.setSize(D.N(), D.M());

  for (std::size_t i = 0; i < D.N(); ++i) R.setrowsize(i, 1);
  R.endrowsizes();
  for (std::size_t i = 0; i < D.N(); ++i) R.addindex(i, i);
  R.endindices();

  for (std::size_t i = 0; i < D.N(); ++i) {
    auto block = D[i][i];
    block.invert();
    R[i][i] = block;
  }

  return R;
}

/** \brief Returns A with alpha * P added into the submatrix A[rows, cols].
 *
 * This is the counterpart of extract_submatrix(): \p P is indexed in the local ordering,
 * i.e. its entry (i, j) is added to A[rows[i]][cols[j]]. The pattern of the result is the
 * union of the pattern of \p A and the (embedded) pattern of \p P, so this is the right
 * tool when the addition introduces couplings that \p A does not have -- which is exactly
 * why it cannot be done in place.
 *
 * Both \p rows and \p cols must be strictly increasing. That makes the local -> global
 * map monotone, so the merged column indices come out sorted and can be handed over with
 * setIndicesNoSort().
 */
template <class Matrix>
Matrix add_scaled_submatrix(const Matrix& A, const std::vector<std::size_t>& rows, const std::vector<std::size_t>& cols, typename Matrix::field_type alpha, const Matrix& P)
{
  if (A.buildStage() != Matrix::built || P.buildStage() != Matrix::built) DUNE_THROW(Dune::Exception, "add_scaled_submatrix needs fully built matrices");
  if (rows.size() != P.N()) DUNE_THROW(Dune::Exception, "Got " << rows.size() << " row indices, but the added matrix has " << P.N() << " rows");
  if (cols.size() != P.M()) DUNE_THROW(Dune::Exception, "Got " << cols.size() << " column indices, but the added matrix has " << P.M() << " columns");

  assert(std::is_sorted(rows.begin(), rows.end(), std::less_equal<std::size_t>{}) && "rows must be strictly increasing");
  assert(std::is_sorted(cols.begin(), cols.end(), std::less_equal<std::size_t>{}) && "cols must be strictly increasing");
  assert(std::all_of(rows.begin(), rows.end(), [&](std::size_t r) { return r < A.N(); }) && "row index out of range");
  assert(std::all_of(cols.begin(), cols.end(), [&](std::size_t c) { return c < A.M(); }) && "column index out of range");

  constexpr auto invalid = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> rows_global_to_local(A.N(), invalid);
  for (std::size_t i = 0; i < rows.size(); ++i) rows_global_to_local[rows[i]] = i;

  // Collects the union of the column indices of A[i] and of the row of P embedded there.
  // Both index sets are sorted, so this is a plain two-pointer merge.
  std::vector<std::size_t> merged;
  auto merge_row = [&](std::size_t i, std::size_t local) {
    merged.clear();
    auto a = A[i].begin();
    const auto a_end = A[i].end();

    if (local != invalid) {
      auto p = P[local].begin();
      const auto p_end = P[local].end();
      while (a != a_end && p != p_end) {
        const auto ja = a.index();
        const auto jp = cols[p.index()];
        if (ja < jp) {
          merged.push_back(ja);
          ++a;
        }
        else if (jp < ja) {
          merged.push_back(jp);
          ++p;
        }
        else {
          merged.push_back(ja);
          ++a;
          ++p;
        }
      }
      for (; p != p_end; ++p) merged.push_back(cols[p.index()]);
    }

    for (; a != a_end; ++a) merged.push_back(a.index());
  };

  Matrix B;
  B.setBuildMode(Matrix::random);
  B.setSize(A.N(), A.M());

  // Pass 1: row sizes
  for (std::size_t i = 0; i < A.N(); ++i) {
    merge_row(i, rows_global_to_local[i]);
    B.setrowsize(i, merged.size());
  }
  B.endrowsizes();

  // Pass 2: column indices
  for (std::size_t i = 0; i < A.N(); ++i) {
    merge_row(i, rows_global_to_local[i]);
    B.setIndicesNoSort(i, merged.begin(), merged.end());
  }
  B.endindices();

  // Pass 3: values. B[i] is a superset of both source rows and its indices are sorted, so
  // a single forward scan of the target row suffices for each of the two contributions.
  B = 0;
  for (std::size_t i = 0; i < A.N(); ++i) {
    auto target = B[i].begin();
    for (auto a = A[i].begin(); a != A[i].end(); ++a) {
      while (target.index() != a.index()) ++target;
      *target = *a;
    }

    const auto local = rows_global_to_local[i];
    if (local == invalid) continue;

    target = B[i].begin();
    for (auto p = P[local].begin(); p != P[local].end(); ++p) {
      const auto j = cols[p.index()];
      while (target.index() != j) ++target;
      target->axpy(alpha, *p);
    }
  }

  return B;
}

/** @brief Eliminate Dirichlet degrees of freedom from a matrix
 *
 *  For Dirichlet DOFs (where dirichlet_mask > 0), sets the diagonal entry to 1.0
 *  and all off-diagonal entries in that row to 0.0. Also zeros out the column
 *  entries corresponding to Dirichlet DOFs in other rows. This maintains the
 *  symmetric structure of the matrix while enforcing Dirichlet constraints.
 *
 *  @param A Matrix to modify (modified in place)
 *  @param dirichlet_mask Vector with non-zero values at Dirichlet DOF indices
 *  @param symmetrically Wheter to eliminate symmetrically or not
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
} // namespace ddm
