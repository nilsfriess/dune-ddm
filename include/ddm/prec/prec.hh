#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/mat/mat.hh"
#include "ddm/registry.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solvercategory.hh>
#include <memory>
#include <string>
#include <utility>

namespace ddm {
template <class T>
class Prec : public Dune::Preconditioner<Vec<T>, Vec<T>> {
public:
  using value_type = T;

  virtual ~Prec() = default;
  Prec(const Prec&) = delete;
  Prec& operator=(const Prec&) = delete;

  void pre(Vec<T>&, Vec<T>&) override {}
  void post(Vec<T>&) override {}

  // Sequential by default, parallel implementations override this
  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::sequential; }

  // z = P^{-1} r
  void apply(Vec<T>& z, const Vec<T>& r) final override
  {
    DDM_CHECK(&r != &z, "prec: apply() requires distinct vectors r and z");
    DDM_CHECK(r.size() == A_->rows() && z.size() == A_->rows(), "prec: apply() size mismatch, matrix is {}x{}, but r has size {} and z has size {}", A_->rows(), A_->cols(), r.size(), z.size());
    DDM_CHECK(r.backend() == A_->backend() && z.backend() == A_->backend(), "prec: apply() backend mismatch, matrix is {}, r is {}, z is {}", to_string(A_->backend()), to_string(r.backend()),
              to_string(z.backend()));
    do_apply(z, r);
  }

  // Recomputes the preconditioner after the values (but not the pattern) of the matrix changed
  void update()
  {
    DDM_CHECK(A_->is_assembled(), "prec: update() called with unassembled matrix");
    do_update();
  }

  const std::shared_ptr<const Mat<T>>& mat() const { return A_; }

protected:
  explicit Prec(std::shared_ptr<const Mat<T>> A)
      : A_(std::move(A))
  {
    DDM_CHECK(A_ != nullptr, "prec: matrix is nullptr");
    DDM_CHECK(A_->rows() == A_->cols(), "prec: matrix must be square, but is {}x{}", A_->rows(), A_->cols());
  }

private:
  virtual void do_apply(Vec<T>& z, const Vec<T>& r) = 0;
  virtual void do_update() = 0;

  std::shared_ptr<const Mat<T>> A_;
};

template <class T>
using PrecRegistry = Registry<std::shared_ptr<Prec<T>>, std::shared_ptr<const Mat<T>>>;

// Registers a prec that works for any backend
template <class T>
void register_prec(std::string name, typename PrecRegistry<T>::Factory factory)
{
  PrecRegistry<T>::instance().add(std::move(name), std::move(factory));
}

// Registers a prec that is only used for matrices of the given backend. It takes precedence over one with the same
// name that is registered for any backend.
template <class T>
void register_prec(std::string name, BackendId backend, typename PrecRegistry<T>::Factory factory)
{
  PrecRegistry<T>::instance().add(std::move(name), backend, std::move(factory));
}

template <class T>
std::shared_ptr<Prec<T>> create_prec(const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A)
{
  initialize();
  DDM_CHECK(A != nullptr, "prec: matrix is nullptr");
  const auto backend = A->backend(); // before A is moved into the argument list
  return PrecRegistry<T>::instance().create_for_backend("prec", config, "none", backend, std::move(A));
}
} // namespace ddm
