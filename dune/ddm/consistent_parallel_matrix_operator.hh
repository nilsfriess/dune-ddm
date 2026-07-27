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
 *  distributed matrix with create_overlapping_matrix().
 *
 *  The local product is almost, but not quite, consistent already. Entry i of A*x is right whenever
 *  row i holds every global nonzero of that row, and a consistent x supplies the right value at
 *  every column it reaches. That covers all owned indices, and the copies well inside the overlap
 *  too -- but not the copies on the outermost layer, whose rows are truncated where the subdomain
 *  ends. Since the result is right exactly on the owned entries, one copyOwnerToAll() repairs the
 *  rest, and that is what apply() does.
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
class ConsistentParallelMatrixOperator : public Dune::AssembledLinearOperator<Mat, X, Y> {
public:
  using domain_type = X;
  using range_type = Y;
  using matrix_type = Mat;
  using communication_type = Communication;
  using field_type = typename X::field_type;

  ConsistentParallelMatrixOperator(std::shared_ptr<Mat> A, std::shared_ptr<Communication> comm)
      : A(std::move(A))
      , comm(std::move(comm))
  {
  }

  ConsistentParallelMatrixOperator(const ConsistentParallelMatrixOperator&) = delete;
  ConsistentParallelMatrixOperator(ConsistentParallelMatrixOperator&&) = delete;
  ConsistentParallelMatrixOperator& operator=(const ConsistentParallelMatrixOperator&) = delete;
  ConsistentParallelMatrixOperator& operator=(ConsistentParallelMatrixOperator&&) = delete;
  ~ConsistentParallelMatrixOperator() = default;

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::overlapping; }

  void apply(const X& x, Y& y) const override
  {
    A->mv(x, y);
    comm->copyOwnerToAll(y, y);
  }

  void applyscaleadd(field_type alpha, const X& x, Y& y) const override
  {
    // y and alpha*A*x are both right on the owned entries, hence so is their sum, and a single
    // reconstruction repairs the copies of the sum. No temporary is needed, unlike in
    // AdditiveParallelMatrixOperator, where the communication sums across the ranks and so must not
    // see the already consistent y.
    A->usmv(alpha, x, y);
    comm->copyOwnerToAll(y, y);
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
std::shared_ptr<ConsistentScalarProduct<X, C>> createScalarProduct(const std::shared_ptr<ConsistentParallelMatrixOperator<M, X, Y, C>>& op)
{
  return std::make_shared<ConsistentScalarProduct<X, C>>(op->getCommunicationPtr());
}
} // namespace Dune
