#pragma once

#include <dune/istl/scalarproducts.hh>
#include <memory>

namespace Dune {

/** @brief Scalar product for consistently stored distributed vectors.
 *
 *  Sums over the owned entries only and reduces over all ranks, which counts every global index
 *  exactly once. That is exact for the consistent vectors dune-ddm exchanges everywhere, see
 *  ddm.hh; it is also exact if one or both arguments happen to be stored owner-only, because the
 *  two representations agree on the owned entries.
 *
 *  Note that this deliberately does not depend on how the matrix of the accompanying operator is
 *  stored: additive and overlapping operators share this scalar product.
 */
template <class X, class C>
class ConsistentScalarProduct : public Dune::ScalarProduct<X> {
public:
  using base = Dune::ScalarProduct<X>;
  using communication_type = C;
  using field_type = typename base::field_type;
  using real_type = typename base::real_type;

  explicit ConsistentScalarProduct(std::shared_ptr<communication_type> comm)
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

  SolverCategory::Category category() const override { return SolverCategory::overlapping; }

private:
  std::shared_ptr<communication_type> comm;
};

} // namespace Dune
