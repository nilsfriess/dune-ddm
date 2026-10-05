#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/registry.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvercategory.hh>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace ddm {
template <class T>
class Mat : public Dune::LinearOperator<Vec<T>, Vec<T>> {
public:
  using field_type = T;

  virtual ~Mat() = default;
  Mat(const Mat&) = delete;
  Mat& operator=(const Mat&) = delete;

  virtual Index rows() const = 0;
  virtual Index cols() const = 0;
  virtual BackendId backend() const = 0;

  // Sequential by default, parallel implementations override this
  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::sequential; }

  // Adds the dense block `vals` (row-major, rows.size() x cols.size()) at the given global indices.
  // Every (rows[r], cols[c]) must be part of the pattern.
  void add_values(std::span<const Index> rows, std::span<const Index> cols, std::span<const T> vals)
  {
    DDM_CHECK(vals.size() == rows.size() * cols.size(), "mat: add_values() expects {} values, got {}", rows.size() * cols.size(), vals.size());
    for (auto i : rows) DDM_CHECK(0 <= i && i < this->rows(), "mat: add_values() row {} out of range for {} rows", i, this->rows());
    for (auto j : cols) DDM_CHECK(0 <= j && j < this->cols(), "mat: add_values() column {} out of range for {} columns", j, this->cols());
    assembled_ = false;
    do_add_values(rows, cols, vals);
  }

  // Must be called after the last add_values() and before the matrix is used
  void assemble()
  {
    do_assemble();
    assembled_ = true;
  }

  bool is_assembled() const { return assembled_; }

  // Sets all entries in the given rows to zero and the diagonal entries of these rows to diag, e.g. to impose
  // Dirichlet conditions. The diagonal entries must be part of the pattern. Only the rows are changed, so a symmetric
  // matrix generally becomes nonsymmetric.
  void zero_rows(std::span<const Index> rows, T diag = T{1})
  {
    DDM_CHECK(assembled_, "mat: zero_rows() called before assemble()");
    DDM_CHECK(this->rows() == cols(), "mat: zero_rows() requires a square matrix, but matrix is {}x{}", this->rows(), cols());
    for (auto i : rows) DDM_CHECK(0 <= i && i < this->rows(), "mat: zero_rows() row {} out of range for {} rows", i, this->rows());
    do_zero_rows(rows, diag);
  }

  // y = A x
  void apply(const Vec<T>& x, Vec<T>& y) const final
  {
    DDM_CHECK(assembled_, "mat: apply() called before assemble()");
    DDM_CHECK(&x != &y, "mat: apply() requires distinct vectors x and y");
    DDM_CHECK(x.size() == cols() && y.size() == rows(), "mat: apply() size mismatch, matrix is {}x{}, but x has size {} and y has size {}", rows(), cols(), x.size(), y.size());
    DDM_CHECK(x.backend() == backend() && y.backend() == backend(), "mat: apply() backend mismatch, matrix is {}, x is {}, y is {}", to_string(backend()), to_string(x.backend()),
              to_string(y.backend()));
    do_apply(x, y);
  }

  // y = y + alpha * A x
  void applyscaleadd(T alpha, const Vec<T>& x, Vec<T>& y) const final
  {
    DDM_CHECK(assembled_, "mat: applyscaleadd() called before assemble()");
    DDM_CHECK(&x != &y, "mat: applyscaleadd() requires distinct vectors x and y");
    DDM_CHECK(x.size() == cols() && y.size() == rows(), "mat: applyscaleadd() size mismatch, matrix is {}x{}, but x has size {} and y has size {}", rows(), cols(), x.size(), y.size());
    DDM_CHECK(x.backend() == backend() && y.backend() == backend(), "mat: applyscaleadd() backend mismatch, matrix is {}, x is {}, y is {}", to_string(backend()), to_string(x.backend()),
              to_string(y.backend()));
    do_applyscaleadd(alpha, x, y);
  }

  // Returns a zero-initialized vector x from the domain of the matrix, i.e. one that can be used in apply(x, y)
  virtual Vec<T> create_domain_vector() const = 0;

  // Returns a zero-initialized vector y from the range of the matrix, i.e. one that can be used in apply(x, y)
  virtual Vec<T> create_range_vector() const = 0;

  // Returns zero-initialized vectors {x, y} that are compatible with apply(x, y)
  std::pair<Vec<T>, Vec<T>> create_vectors() const { return {create_domain_vector(), create_range_vector()}; }

  // Fill the given vector with the matrix's diagonal. Throws if the matrix is not square
  void get_diag(Vec<T>& diag) const
  {
    DDM_CHECK(assembled_, "mat: get_diag() called before assemble()");
    DDM_CHECK(rows() == cols(), "mat: get_diag() only for square matrices");
    DDM_CHECK(rows() == diag.size(), "mat: get_diag() size mismatch, matrix is {}x{}, vector is {}", rows(), cols(), diag.size());
    DDM_CHECK(backend() == diag.backend(), "mat: get_diag() backend mismatch, matrix is {}, vector is {}", to_string(backend()), to_string(diag.backend()));
    do_get_diag(diag);
  }

protected:
  Mat() = default;

private:
  virtual void do_add_values(std::span<const Index> rows, std::span<const Index> cols, std::span<const T> vals) = 0;
  virtual void do_assemble() = 0;
  virtual void do_zero_rows(std::span<const Index> rows, T diag) = 0;
  virtual void do_apply(const Vec<T>& x, Vec<T>& y) const = 0;
  virtual void do_applyscaleadd(T alpha, const Vec<T>& x, Vec<T>& y) const = 0;
  virtual void do_get_diag(Vec<T>& diag) const = 0;

  bool assembled_ = false;
};

template <class T>
using MatRegistry = Registry<std::shared_ptr<Mat<T>>, const Pattern&>;

template <class T>
void register_mat(std::string name, typename MatRegistry<T>::Factory factory)
{
  MatRegistry<T>::instance().add(std::move(name), std::move(factory));
}

// The pattern must be finalized
template <class T>
std::shared_ptr<Mat<T>> create_mat(const Dune::ParameterTree& config, const Pattern& pattern)
{
  DDM_CHECK(pattern.is_finalized(), "mat: create_mat() requires a finalized pattern");
  initialize();
  return MatRegistry<T>::instance().create("mat", config, "istl", pattern);
}
} // namespace ddm
