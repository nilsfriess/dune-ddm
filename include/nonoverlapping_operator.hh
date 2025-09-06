#pragma once

#include "helpers.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/communication.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/istl/operators.hh>

class NonOverlappingCommunicator {
public:
  template <class RemoteIndices>
  explicit NonOverlappingCommunicator(const RemoteIndices &ri) : cc(ri.communicator())
  {
    mask.resize(ri.sourceIndexSet().size(), 1);
    for (const auto &idx : ri.sourceIndexSet()) {
      mask[idx.local()] = (unsigned)(idx.local().attribute() == Attribute::owner);
    }
  }

  const Dune::Communication<MPI_Comm> &communicator() const { return cc; }

  template <class X>
  void copyOwnerToAll(const X &, X &) const
  {
    DUNE_THROW(Dune::NotImplemented, "copyOwnerToAll");
  }

  template <class X>
  void addOwnerCopyToOwnerCopy(const X &, X &) const
  {
    DUNE_THROW(Dune::NotImplemented, "addOwnerCopyToOwnerCopy");
  }

  template <class X>
  void dot(const X &x, const X &y, double &res) const
  {
    res = 0;
    for (typename X::size_type i = 0; i < x.size(); i++) {
      res += x[i] * y[i] * mask[i];
    }
    res = cc.sum(res);
  }

  template <class X>
  double norm(const X &x) const
  {
    double res = 0;
    dot(x, x, res);
    return std::sqrt(res);
  }

private:
  Dune::Communication<MPI_Comm> cc;
  std::vector<unsigned> mask;
};

template <class Mat, class X, class Y = X>
class NonOverlappingOperator : public Dune::AssembledLinearOperator<Mat, X, Y> {
private:
  struct AddGatherScatter {
    using DataType = typename Dune::CommPolicy<X>::IndexedType;

    static DataType gather(const X &x, std::size_t i) { return x[i]; }
    static void scatter(X &x, DataType v, std::size_t i) { x[i] += v; }
  };

public:
  using domain_type = X;
  using range_type = Y;
  using matrix_type = Mat;
  using communication_type = NonOverlappingCommunicator;
  using field_type = typename X::field_type;

  template <class RemoteIndices>
  NonOverlappingOperator(std::shared_ptr<Mat> A, const RemoteIndices &ri) : A(std::move(A)), comm(ri)
  {
    const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
    all_all_interface.build(ri, allAttributes, allAttributes);
    communicator = std::make_shared<Dune::BufferedCommunicator>();
    communicator->build<X>(all_all_interface);
  }

  NonOverlappingOperator(const NonOverlappingOperator &) = delete;
  NonOverlappingOperator(const NonOverlappingOperator &&) = delete;
  NonOverlappingOperator &operator=(const NonOverlappingOperator &) = delete;
  NonOverlappingOperator &operator=(const NonOverlappingOperator &&) = delete;
  ~NonOverlappingOperator() = default;

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void apply(const X &x, Y &y) const override
  {
    // Logger::ScopedLog sl(apply_event);
    A->mv(x, y);
    communicator->forward<AddGatherScatter>(y);
  }

  void applyscaleadd(field_type alpha, const X &x, Y &y) const override
  {
    // Logger::ScopedLog sl(applyscaleadd_event);

    Y y1(y.N());
    y1 = 0;
    A->usmv(alpha, x, y1);
    communicator->forward<AddGatherScatter>(y1);
    y += y1;
  }

  const Mat &getmat() const override { return *A; }

  const communication_type &getCommunication() const { return comm; }

private:
  std::shared_ptr<Mat> A;
  Dune::Interface all_all_interface;
  std::shared_ptr<Dune::BufferedCommunicator> communicator;

  communication_type comm;

  // Logger::Event *apply_event;
  // Logger::Event *applyscaleadd_event;
};
