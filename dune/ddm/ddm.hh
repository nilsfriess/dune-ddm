#pragma once

/** @file
 *
 *  @brief Module-wide conventions for distributed vectors.
 *
 *  A vector distributed over an OwnerOverlapCopy index set can be stored in one of two ways. It is
 *  <b>consistent</b> if every rank holds the true global value at every index it owns or copies, so
 *  that duplicated entries agree. It is <b>additive</b> if the true value at an index is the sum of
 *  the contributions of all ranks holding it; the special case where the owner holds the whole value
 *  and every copy holds zero is what Dune::OwnerOverlapCopyCommunication::project() produces.
 *
 *  <b>Everything in dune-ddm exchanges consistent vectors.</b> Concretely:
 *
 *  - A linear operator maps a consistent argument to a consistent image. If the matrix is stored
 *    additively, that means summing the local products (AdditiveParallelMatrixOperator); if it is a
 *    consistent copy of the global rows on an overlapping subdomain, it means overwriting the copies
 *    from their owners (ConsistentParallelMatrixOperator).
 *  - A preconditioner maps a consistent defect to a consistent correction. Corrections from several
 *    preconditioners may therefore simply be added, which is what CombinedPreconditioner does.
 *  - A scalar product masks the sum to the owned entries, see ConsistentScalarProduct. That is exact
 *    for consistent vectors.
 *  - Every class reports Dune::SolverCategory::overlapping, which in this module is to be read as
 *    "vectors are consistent".
 *
 *  The invariant is what lets a Krylov method run without a preconditioner: the residual
 *  \f$r = b - Ax\f$ comes back in the representation the operator expects as input, so it can be
 *  used directly as a search direction. Dune::OverlappingSchwarzOperator deliberately breaks this --
 *  it projects its result onto the owners and leaves it to the preconditioner to restore consistency
 *  -- which is why this module provides its own overlapping operator instead.
 */
