#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/registry.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ddm {
// The entries of a matrix in CSR form in host memory: the entries of row i are at positions
// [row_ptr[i], row_ptr[i + 1]) of cols and values.
template <class T>
struct HostCsr {
  std::vector<Index> row_ptr;
  std::vector<Index> cols;
  std::vector<T> values;
};

// Interface implemented by the matrix backends: a sequential matrix in local numbering, without any knowledge of
// other ranks. Users of the library work with Mat<T>, which holds a LocalMat<T> and adds the parallel semantics.
//
// The public functions check their arguments and then call the corresponding do_*() function, so implementations
// can rely on valid arguments.
template <class T>
class LocalMat {
public:
  using field_type = T;

  virtual ~LocalMat() = default;
  LocalMat(const LocalMat&) = delete;
  LocalMat& operator=(const LocalMat&) = delete;

  virtual Index rows() const = 0;
  virtual Index cols() const = 0;
  virtual BackendId backend() const = 0;

  // The pattern the matrix was created with
  const Pattern& pattern() const { return pattern_; }

  // Adds the dense block `vals` (row-major, rows.size() x cols.size()) at the given indices.
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
  void mv(const Vec<T>& x, Vec<T>& y) const
  {
    check_mv_args(x, y, "mv()");
    do_mv(x, y);
  }

  // y = y + alpha * A x
  void usmv(T alpha, const Vec<T>& x, Vec<T>& y) const
  {
    check_mv_args(x, y, "usmv()");
    do_usmv(alpha, x, y);
  }

  // Returns a zero-initialized vector x from the domain of the matrix, i.e. one that can be used in mv(x, y)
  virtual Vec<T> create_domain_vector() const = 0;

  // Returns a zero-initialized vector y from the range of the matrix, i.e. one that can be used in mv(x, y)
  virtual Vec<T> create_range_vector() const = 0;

  // Fill the given vector with the matrix's diagonal. Throws if the matrix is not square
  void get_diag(Vec<T>& diag) const
  {
    DDM_CHECK(assembled_, "mat: get_diag() called before assemble()");
    DDM_CHECK(rows() == cols(), "mat: get_diag() only for square matrices");
    DDM_CHECK(rows() == diag.size(), "mat: get_diag() size mismatch, matrix is {}x{}, vector is {}", rows(), cols(), diag.size());
    DDM_CHECK(backend() == diag.backend(), "mat: get_diag() backend mismatch, matrix is {}, vector is {}", to_string(backend()), to_string(diag.backend()));
    do_get_diag(diag);
  }

  // Returns the entries of the matrix. Backends that do not support this throw
  HostCsr<T> host_csr() const
  {
    DDM_CHECK(assembled_, "mat: host_csr() called before assemble()");
    return do_host_csr();
  }

  // Print info about this matrix
  void info() const { do_info(); }

protected:
  explicit LocalMat(const Pattern& pattern)
      : pattern_(pattern)
  {
  }

private:
  void check_mv_args(const Vec<T>& x, const Vec<T>& y, std::string_view op) const
  {
    DDM_CHECK(assembled_, "mat: {} called before assemble()", op);
    DDM_CHECK(&x != &y, "mat: {} requires distinct vectors x and y", op);
    DDM_CHECK(x.size() == cols() && y.size() == rows(), "mat: {} size mismatch, matrix is {}x{}, but x has size {} and y has size {}", op, rows(), cols(), x.size(), y.size());
    DDM_CHECK(x.backend() == backend() && y.backend() == backend(), "mat: {} backend mismatch, matrix is {}, x is {}, y is {}", op, to_string(backend()), to_string(x.backend()),
              to_string(y.backend()));
  }

  virtual void do_add_values(std::span<const Index> rows, std::span<const Index> cols, std::span<const T> vals) = 0;
  virtual void do_assemble() = 0;
  virtual void do_zero_rows(std::span<const Index> rows, T diag) = 0;
  virtual void do_mv(const Vec<T>& x, Vec<T>& y) const = 0;
  virtual void do_usmv(T alpha, const Vec<T>& x, Vec<T>& y) const = 0;
  virtual void do_get_diag(Vec<T>& diag) const = 0;
  virtual void do_info() const = 0;

  virtual HostCsr<T> do_host_csr() const
  {
    DDM_CHECK(false, "mat: host_csr() is not supported by backend {}", to_string(backend()));
    return {};
  }

  Pattern pattern_;
  bool assembled_ = false;
};

template <class T>
using LocalMatRegistry = Registry<std::shared_ptr<LocalMat<T>>, const Pattern&>;

template <class T>
void register_local_mat(std::string name, typename LocalMatRegistry<T>::Factory factory)
{
  LocalMatRegistry<T>::instance().add(std::move(name), std::move(factory));
}

// Creates the local matrix of the type given by config["type"]. The pattern must be finalized
template <class T>
std::shared_ptr<LocalMat<T>> create_local_mat(const Dune::ParameterTree& config, const Pattern& pattern)
{
  DDM_CHECK(pattern.is_finalized(), "mat: create_local_mat() requires a finalized pattern");
  initialize();
  return LocalMatRegistry<T>::instance().create("mat", config, "istl", pattern);
}
} // namespace ddm
