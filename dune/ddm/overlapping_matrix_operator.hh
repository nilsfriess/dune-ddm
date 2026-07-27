#pragma once

#include "consistent_scalar_product.hh"
#include "ddm.hh"

#include <dune/istl/operators.hh>
#include <dune/istl/scalarproducts.hh>
#include <memory>

/** @brief Linear operator for a matrix that is a consistent copy of the global rows on an
 *         overlapping subdomain.
 *
 *  This is the situation after assembling on an overlapping grid view, or after extending a
 *  distributed matrix with create_overlapping_matrix(): every row belonging to an index this rank
 *  owns is complete, so the local product already carries the right value there. The rows of the
 *  copied indices at the outer rim of the subdomain are not complete, which is why apply() finishes
 *  by overwriting the copies with the values of their owners.
 *
 *  Unlike Dune::OverlappingSchwarzOperator, which projects its result onto the owners and leaves it
 *  to the preconditioner to restore consistency, this returns a consistent vector and so satisfies
 *  the invariant in ddm.hh. Both perform exactly one collective per application; the difference is
 *  only where it happens, and doing it here means an unpreconditioned Krylov method works and every
 *  preconditioner of a two-level scheme is spared its own reconstruction.
 *
 *  For a matrix stored additively across the ranks use AdditiveParallelMatrixOperator instead.
 */
template <class Mat, class X, class Y, class Communication>
class OverlappingMatrixOperator : public Dune::AssembledLinearOperator<Mat, X, Y> {
public:
  using domain_type = X;
  using range_type = Y;
  using matrix_type = Mat;
  using communication_type = Communication;
  using field_type = typename X::field_type;

  OverlappingMatrixOperator(std::shared_ptr<Mat> A, std::shared_ptr<Communication> comm)
      : A(std::move(A))
      , comm(std::move(comm))
  {
  }

  OverlappingMatrixOperator(const OverlappingMatrixOperator&) = delete;
  OverlappingMatrixOperator(OverlappingMatrixOperator&&) = delete;
  OverlappingMatrixOperator& operator=(const OverlappingMatrixOperator&) = delete;
  OverlappingMatrixOperator& operator=(OverlappingMatrixOperator&&) = delete;
  ~OverlappingMatrixOperator() = default;

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::overlapping; }

  void apply(const X& x, Y& y) const override
  {
    A->mv(x, y);
    comm->copyOwnerToAll(y, y);
  }

  void applyscaleadd(field_type alpha, const X& x, Y& y) const override
  {
    // y is consistent on entry, so only alpha*A*x has to be made consistent before adding it on.
    // Overwriting copies cannot be done in place on the sum, unlike the additive case.
    X ax(y);
    ax = 0;
    A->usmv(alpha, x, ax);
    comm->copyOwnerToAll(ax, ax);
    y += ax;
  }

  const Mat& getmat() const override { return *A; }

  const communication_type& getCommunication() const { return *comm; }
  std::shared_ptr<communication_type> getCommunicationPtr() const { return comm; }

private:
  std::shared_ptr<Mat> A;
  std::shared_ptr<communication_type> comm;
};

namespace Dune {
template <class M, class X, class Y, class C>
std::shared_ptr<ConsistentScalarProduct<X, C>> createScalarProduct(const std::shared_ptr<OverlappingMatrixOperator<M, X, Y, C>>& op)
{
  return std::make_shared<ConsistentScalarProduct<X, C>>(op->getCommunicationPtr());
}
} // namespace Dune
