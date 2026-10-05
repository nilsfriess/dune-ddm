#pragma once

#include "ddm/communication.hh"
#include "ddm/vec/vec.hh"

#include <dune/istl/scalarproducts.hh>
#include <dune/istl/solvercategory.hh>
#include <memory>

namespace ddm {
/** @brief Scalar product for consistently stored distributed vectors.
 *
 *  Sums over the owned entries only and reduces over all ranks, which counts every global index
 *  exactly once. That is exact for the consistent vectors dune-ddm exchanges everywhere.
 */
template <class T>
class ConsistentScalarProduct : public Dune::ScalarProduct<Vec<T>> {
public:
  explicit ConsistentScalarProduct(std::shared_ptr<Communication> comm)
      : comm_(std::move(comm))
  {
  }

  T dot(const Vec<T>& x, const Vec<T>& y) const override
  {
    T res{};
    comm_->dot(x, y, res);
    return res;
  }

  T norm(const Vec<T>& x) const override { return comm_->norm(x); }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::overlapping; }

private:
  std::shared_ptr<Communication> comm_;
};
} // namespace ddm
