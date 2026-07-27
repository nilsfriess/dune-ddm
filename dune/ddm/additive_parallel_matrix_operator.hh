#pragma once

#include "consistent_scalar_product.hh"
#include "ddm.hh"
#include "helpers.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/communication.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/scalarproducts.hh>
#include <memory>

/** @brief Linear operator for a matrix stored additively across the ranks.
 *
 *  Additive means that no rank holds the full rows it shares with its neighbours; the global matrix
 *  is the sum of the local ones. apply() therefore multiplies locally and then sums the result over
 *  the ranks, which turns the additive product back into a consistent vector as ddm.hh requires.
 *
 *  This is the counterpart of Dune::MatrixAdapter for a distributed, additively stored matrix. Use
 *  it wherever the matrix comes out of a distributed assembly; for a matrix that is a consistent
 *  copy of the global rows on an overlapping subdomain, use ConsistentParallelMatrixOperator instead.
 */
template <class Mat, class X, class Y, class Communication>
class AdditiveParallelMatrixOperator : public Dune::AssembledLinearOperator<Mat, X, Y> {
public:
  using domain_type = X;
  using range_type = Y;
  using matrix_type = Mat;
  using communication_type = Communication;
  using field_type = typename X::field_type;

  AdditiveParallelMatrixOperator(std::shared_ptr<Mat> A, std::shared_ptr<Communication> comm)
      : A(std::move(A))
      , comm(std::move(comm))
  {
  }

  AdditiveParallelMatrixOperator(const AdditiveParallelMatrixOperator&) = delete;
  AdditiveParallelMatrixOperator(const AdditiveParallelMatrixOperator&&) = delete;
  AdditiveParallelMatrixOperator& operator=(const AdditiveParallelMatrixOperator&) = delete;
  AdditiveParallelMatrixOperator& operator=(const AdditiveParallelMatrixOperator&&) = delete;
  ~AdditiveParallelMatrixOperator() = default;

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::overlapping; }

  void apply(const X& x, Y& y) const override
  {
    // Logger::ScopedLog sl(apply_event);
    A->mv(x, y);
    comm->addOwnerCopyToOwnerCopy(y, y);
  }

  void applyscaleadd(field_type alpha, const X& x, Y& y) const override
  {
    // Logger::ScopedLog sl(applyscaleadd_event);
    // Since y is already consistent, we only communicate the result of alpha*A*x.
    auto y1 = y;
    y = 0;
    A->usmv(alpha, x, y);
    comm->addOwnerCopyToOwnerCopy(y, y);
    y += y1;
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
std::shared_ptr<ConsistentScalarProduct<X, C>> createScalarProduct(const std::shared_ptr<AdditiveParallelMatrixOperator<M, X, Y, C>>& op)
{
  return std::make_shared<ConsistentScalarProduct<X, C>>(op->getCommunicationPtr());
}
} // namespace Dune
