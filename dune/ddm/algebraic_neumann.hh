#pragma once

/** @file algebraic_neumann.hh
    @brief Approximate the Neumann matrix of an overlapping subdomain from the assembled matrix alone.

    The spectral coarse spaces in coarsespaces/coarse_spaces.hh all need the Neumann matrix
    \f$A^\mathrm{Neu}\f$ of the overlapping subdomain, i.e. the assembly of only those element
    contributions that come from elements inside the subdomain. That matrix is not algebraically
    available: assembling it requires access to the element level, which is what makes MS-GFEM
    (and GenEO) non-algebraic.

    This header provides approximations of \f$A^\mathrm{Neu}\f$ that use nothing but the assembled
    matrix and the overlapping communication object, so that the coarse spaces can be built for
    problems where no assembler is available (e.g. a matrix read from a file).
 */

#include "logger.hh"
#include "overlap_extension.hh"

#include <cstddef>
#include <mpi.h>
#include <vector>

/** @brief Replace @p A by its symmetric part \f$\tfrac{1}{2}(A + A^T)\f$, in place.
 *
 *  The spectral coarse spaces are built with symmetric eigensolvers, so a non-symmetric operator
 *  needs a symmetric surrogate. Dropping the skew part is not merely a way of forcing symmetry.
 *  For a discretisation of \f$-\nabla\cdot(K\nabla u) + b\cdot\nabla u\f$ with \f$\nabla\cdot b = 0\f$
 *  the skew part *is* the convection, so the symmetric part is exactly the operator the eigenproblem
 *  should see, and the split is exact rather than approximate.
 *
 *  Concretely, for the upwind DG discretisation used by the examples (SIPG diffusion + full
 *  upwinding), write the upwind flux as a central part plus a jump,
 *  \f$u^\mathrm{up} = \{u\} + \tfrac12\operatorname{sign}(b\cdot n)[u]\f$. The central part cancels
 *  against the volume term \f$-\int_E u\, b\cdot\nabla v\f$ and what survives is
 *
 *  \f[ c(u,v) + c(v,u) = \int_\Omega (\nabla\cdot b)\,uv + \sum_F \int_F |b\cdot n|\,[u][v] + \text{(boundary)}, \f]
 *
 *  i.e. the symmetric part is the SIPG diffusion plus the upwind numerical diffusion. That is the
 *  norm in which the scheme is coercive, and \f$x^T A x = x^T \operatorname{sym}(A) x\f$ holds
 *  identically: the symmetric part is positive definite exactly when the operator is. Note that this
 *  keeps strictly more information than assembling the elliptic part of the PDE separately, which
 *  drops the upwind term - the dominant symmetric contribution once the problem is convection
 *  dominated.
 *
 *  @warning The energy identity above is a statement about the *global* operator. On a subdomain the
 *           surviving boundary term \f$\tfrac12\int_{\partial\omega}(b\cdot n)u^2\f$ is negative
 *           wherever the flow enters \f$\omega\f$, so the local matrix can be indefinite at high
 *           Peclet number. That is a property of the local operator rather than of this
 *           construction; if the eigensolver trips over it, shift the eigenproblem or fall back to a
 *           diagonally dominant repair of the boundary rows.
 *
 *  @note Apply this *after* make_algebraic_neumann(). The row-sum correction there rests on
 *        \f$a(\mathbf 1, \varphi_i) = 0\f$, which holds for the rows of the *non-symmetric* matrix
 *        (the convection term annihilates constants row-wise as well, since
 *        \f$a(\mathbf 1, \varphi_i) = \int(\nabla\cdot b)\varphi_i\f$). The correction is diagonal,
 *        so the two operations commute in every other respect.
 *
 *  @note No communication is needed. Restriction commutes with symmetrisation,
 *        \f$\operatorname{sym}(R A R^T) = R \operatorname{sym}(A) R^T\f$ for a boolean \f$R\f$, and
 *        for \f$k, j\f$ both in the overlapping subdomain the local matrix already holds both
 *        \f$A_{kj}\f$ and \f$A_{jk}\f$.
 *
 *  @param A Matrix with a structurally symmetric sparsity pattern, as an assembled FEM matrix has.
 *           Entries are assumed to be scalar, in line with the rest of this header.
 */
template <class Mat>
void symmetrize(Mat& A)
{
  Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Algebraic Neumann", "symmetrise")};

  for (auto ri = A.begin(); ri != A.end(); ++ri) {
    const auto row = ri.index();
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      const auto col = ci.index();
      if (col == row) continue;

      // Both triangles are visited, so checking here catches a missing mirror entry either way round.
      if (not A.exists(col, row)) {
        logger::error_all("symmetrize: entry ({}, {}) has no counterpart ({}, {}); the sparsity pattern is not structurally symmetric", row, col, col, row);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      // Only the upper triangle writes, so A[col][row] is still the original value when it is read.
      if (col > row) {
        const auto mean = 0.5 * ((*ci)[0][0] + A[col][row][0][0]);
        (*ci)[0][0] = mean;
        A[col][row][0][0] = mean;
      }
    }
  }
}

/** @brief Approximate the Neumann matrix on an overlapping subdomain using only the assembled matrix.
 *
 *  Write \f$\Gamma\f$ for the outermost DOF layer of the overlapping subdomain (@p boundary) and
 *  \f$O\f$ for the DOFs outside it. The assembled restriction and the true Neumann matrix differ by
 *  the contribution of the elements outside the subdomain,
 *
 *  \f[ R A R^T = A^\mathrm{Neu} + E, \f]
 *
 *  and \f$E\f$ is supported entirely on the \f$\Gamma \times \Gamma\f$ block: a DOF interior to the
 *  subdomain has all of its element neighbours inside the subdomain, so its row of the assembled
 *  matrix already *is* the corresponding Neumann row. Only the outermost layer is wrong.
 *
 *  \f$E\f$ cannot be recovered from the assembled matrix. But if the exterior element assembly
 *  annihilates constants (exact for pure diffusion, approximate once a reaction term is present),
 *  then \f$E \mathbf{1}_\Gamma + A_{\Gamma O}\mathbf{1}_O = 0\f$, so the row sums of \f$E\f$ are known
 *  even though \f$E\f$ is not: \f$(E\mathbf{1})_k = -\sum_{j \in O} A_{kj}\f$. Lumping that onto the
 *  diagonal gives
 *
 *  \f[ A^\mathrm{Neu}_{kk} \approx A_{kk} + \sum_{j \in O} A_{kj}, \qquad k \in \Gamma, \f]
 *
 *  which makes the row sums of the local matrix vanish, i.e. it puts the constant vector back into the
 *  kernel of the local operator where it belongs. This is the correction PETSc applies for
 *  `-pc_hpddm_block_splitting` (see `PCHPDDMAlgebraicAuxiliaryMat_Private()` in `pchpddm.cxx`, which
 *  forms it as "full row sum minus diagonal-block row sum").
 *
 *  The exterior couplings are obtained by extending the overlap by one further layer, after which row
 *  \f$k\f$ is complete for every \f$k \in \Gamma\f$ and the exterior columns are exactly those with a
 *  local index beyond @p A.N().
 *
 *  Note that the couplings enter with their sign. Replacing them by \f$\sum_j |A_{kj}|\f$ gives the
 *  variant of Al Daas and Jolivet (arXiv:2201.02250), which is a provable SPSD lower bound rather than
 *  a correction that is exact on constants; the two coincide for M-matrices.
 *
 *  @warning @p A must be passed with MatrixRepresentation::Consistent. It is a restriction of an
 *           already assembled matrix, so every rank holding an entry holds its complete value. Using
 *           the additive default silently multiplies each shared exterior coupling by the number of
 *           ranks that hold it, which over-subtracts from the diagonal and can render the result
 *           indefinite.
 *
 *  @note Diagonal lumping fixes the action of \f$E\f$ on constants and nothing beyond it. In
 *        particular it does *not* guarantee \f$\hat{A}^\mathrm{Neu} \preceq A^\mathrm{Neu}\f$, so the
 *        approximation may overestimate the local energy and cause the eigensolver to discard modes
 *        that the exact problem would have kept.
 *
 *  @note Rows carrying a Dirichlet condition should not be corrected, since the constants-in-kernel
 *        argument does not apply to them. Pass a @p boundary mask with those DOFs cleared.
 *
 *  @param comm     Communication object on the overlapping index set (must match @p A in size).
 *  @param A        The assembled matrix restricted to the overlapping subdomain.
 *  @param boundary Mask marking the outermost DOF layer \f$\Gamma\f$ of the overlapping subdomain.
 *  @return A copy of @p A with the boundary-layer diagonal corrected.
 */
template <class Communication, class Mat>
[[nodiscard]] Mat make_algebraic_neumann(const Communication& comm, const Mat& A, const std::vector<bool>& boundary)
{
  Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Algebraic Neumann", "row-sum correction")};

  if (boundary.size() != A.N()) {
    logger::error_all("make_algebraic_neumann: boundary mask has size {} but matrix has size {}", boundary.size(), A.N());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // One extra overlap layer, so that row k of A_next is the complete row for every k in Gamma.
  // increase_overlap() appends the newly added DOFs at the end of the index set, so the exterior DOFs O
  // are exactly the columns with a local index >= A.N().
  auto [comm_next, A_next, boundary_next] = create_overlapping_matrix(comm, A, 1, MatrixRepresentation::Consistent);

  Mat A_neumann = A;
  std::size_t corrected = 0;
  for (std::size_t i = 0; i < boundary.size(); ++i) {
    if (boundary[i]) {
      typename Mat::field_type exterior_row_sum = 0;
      for (auto cit = (*A_next)[i].begin(); cit != (*A_next)[i].end(); ++cit)
        if (cit.index() >= A.N()) exterior_row_sum += (*cit)[0][0];
      A_neumann[i][i] += exterior_row_sum;
      ++corrected;
    }
  }
  logger::debug_all("Algebraic Neumann matrix: corrected {} of {} rows", corrected, A.N());

  return A_neumann;
}
