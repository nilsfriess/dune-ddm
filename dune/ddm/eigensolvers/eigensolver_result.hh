#pragma once

/** @file eigensolver_result.hh
    @brief Small value types describing the outcome of a generalised eigensolve.

    These live in their own header (rather than in eigensolvers.hh) so that both the
    high-level eigensolvers.hh and the low-level trl.hh can return them without running
    into include-ordering problems.
 */

#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>

#include <vector>

namespace ddm {
// Metadata describing a completed eigensolver run.
struct EigensolverResult {
  bool converged = false;
  unsigned int iterations = 0;
  unsigned int operator_applications = 0;
};

// Eigenpairs plus run metadata returned by the solve_gevp() family. The i-th entry of
// eigenvalues() corresponds to the i-th entry of eigenvectors(); both are ordered by
// ascending eigenvalue as far as the underlying solver guarantees it. Both vectors are
// empty when the solve did not converge (see info.converged).
template <class Scalar = double>
struct GevpSolution {
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;

  std::vector<Vector> eigenvectors;
  std::vector<Scalar> eigenvalues;
  EigensolverResult info{};
};
} // namespace ddm
