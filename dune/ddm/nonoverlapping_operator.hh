#pragma once

#include "helpers.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/communication.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/istl/operators.hh>
#include <memory>

template <class Mat, class X, class Y, class Communication>
class NonOverlappingOperator : public Dune::AssembledLinearOperator<Mat, X, Y> {
public:
  using domain_type = X;
  using range_type = Y;
  using matrix_type = Mat;
  using communication_type = Communication;
  using field_type = typename X::field_type;

  NonOverlappingOperator(std::shared_ptr<Mat> A, std::shared_ptr<Communication> comm)
      : A(std::move(A))
      , comm(std::move(comm))
  {
  }

  NonOverlappingOperator(const NonOverlappingOperator&) = delete;
  NonOverlappingOperator(const NonOverlappingOperator&&) = delete;
  NonOverlappingOperator& operator=(const NonOverlappingOperator&) = delete;
  NonOverlappingOperator& operator=(const NonOverlappingOperator&&) = delete;
  ~NonOverlappingOperator() = default;

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

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
template <class X, class C>
class NonOverlappingScalarProduct : public Dune::ScalarProduct<X> {
public:
  using base = Dune::ScalarProduct<X>;
  using communication_type = C;
  using field_type = typename base::field_type;
  using real_type = typename base::real_type;

  explicit NonOverlappingScalarProduct(std::shared_ptr<communication_type> comm)
      : comm(std::move(comm))
  {
  }

  field_type dot(const X& x, const X& y) const override
  {
    field_type res{};
    comm->dot(x, y, res);
    return res;
  }

  real_type norm(const X& x) const override { return comm->norm(x); }

  SolverCategory::Category category() const override { return SolverCategory::nonoverlapping; }

private:
  std::shared_ptr<communication_type> comm;
};

template <class M, class X, class Y, class C>
std::shared_ptr<NonOverlappingScalarProduct<X, C>> createScalarProduct(const std::shared_ptr<NonOverlappingOperator<M, X, Y, C>>& op)
{
  return std::make_shared<NonOverlappingScalarProduct<X, C>>(op->getCommunicationPtr());
}
} // namespace Dune
