#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/multivec/multivec.hh"
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
  LocalMat& operator=(const LocalMat&) = delete;

  virtual Index rows() const = 0;
  virtual Index cols() const = 0;
  virtual BackendId backend() const = 0;

  // The pattern the matrix was created with
  const Pattern& pattern() const
  {
    return pattern_;
  }

  // Adds the dense block `vals` (column-major, rows.size() x cols.size()) at the given indices.
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

  bool is_assembled() const
  {
    return assembled_;
  }

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

  // Y = A X, column by column
  void spmm(const MultiVec<T>& X, MultiVec<T>& Y) const
  {
    DDM_CHECK(assembled_, "mat: spmm() called before assemble()");
    DDM_CHECK(!X.overlaps(Y), "mat: spmm() requires distinct multivectors X and Y");
    X.check_usable("spmm()");
    Y.check_usable("spmm()");
    DDM_CHECK(X.rows() == cols() && Y.rows() == rows() && X.cols() == Y.cols(), "mat: spmm() size mismatch, matrix is {}x{}, but X is {}x{} and Y is {}x{}", rows(), cols(), X.rows(), X.cols(),
              Y.rows(), Y.cols());
    DDM_CHECK(X.backend() == backend() && Y.backend() == backend(), "mat: spmm() backend mismatch, matrix is {}, X is {}, Y is {}", to_string(backend()), to_string(X.backend()),
              to_string(Y.backend()));
    do_spmm(X, Y);
  }

  // Returns a zero-initialized vector x from the domain of the matrix, i.e. one that can be used in mv(x, y)
  virtual Vec<T> create_domain_vector() const = 0;

  // Sets all entries in the given rows and columns to zero and the diagonal entries of these rows to diag, e.g. to
  // impose Dirichlet conditions symmetrically. The diagonal entries must be part of the pattern.
  void zero_rows_and_cols(std::span<const Index> idx, T diag = T{1})
  {
    DDM_CHECK(assembled_, "mat: zero_rows_and_cols() called before assemble()");
    DDM_CHECK(this->rows() == cols(), "mat: zero_rows_and_cols() requires a square matrix, but matrix is {}x{}", this->rows(), cols());
    for (auto i : idx) DDM_CHECK(0 <= i && i < this->rows(), "mat: zero_rows_and_cols() index {} out of range for {} rows", i, this->rows());
    do_zero_rows_and_cols(idx, diag);
  }

  // Returns a zero-initialized vector y from the range of the matrix, i.e. one that can be used in mv(x, y)
  virtual Vec<T> create_range_vector() const = 0;

  // Returns a zero-initialized multivector with m columns from the domain of the matrix
  std::unique_ptr<MultiVec<T>> create_domain_multivector(Index m) const
  {
    DDM_CHECK(m >= 0, "mat: create_domain_multivector() with negative number of columns {}", m);
    return do_create_domain_multivector(m);
  }

  // Returns a zero matrix of the same type (and thus backend) as *this with the given pattern, which must be finalized
  std::shared_ptr<LocalMat<T>> create_like(const Pattern& pattern) const
  {
    DDM_CHECK(pattern.is_finalized(), "mat: create_like() requires a finalized pattern");
    return do_create_like(config_, pattern);
  }

  // Returns a copy of the matrix (same type, pattern and values)
  std::shared_ptr<LocalMat<T>> clone() const
  {
    return do_clone();
  }

  // Returns a zero-initialized multivector with m columns from the range of the matrix
  std::unique_ptr<MultiVec<T>> create_range_multivector(Index m) const
  {
    DDM_CHECK(m >= 0, "mat: create_range_multivector() with negative number of columns {}", m);
    return do_create_range_multivector(m);
  }

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
  void info() const
  {
    do_info();
  }

  // Scale the matrix rows and columns by the diagonal matrix diag(d)
  void scale(const Vec<T>& d)
  {
    DDM_CHECK(backend() == d.backend(), "mat: scale() backend mismatch, matrix is {}, vector is {}", to_string(backend()), to_string(d.backend()));
    DDM_CHECK(rows() == d.size() and cols() == d.size(), "mat: scale() size mismatch, matrix is {}x{}, d is {}", rows(), cols(), d.size());
    do_scale(d);
  }

protected:
  explicit LocalMat(const Dune::ParameterTree& config, const Pattern& pattern)
      : config_(config)
      , pattern_(pattern)
  {
  }

  // Only for the implementations of do_clone(), from outside matrices are copied with clone()
  LocalMat(const LocalMat&) = default;

private:
  void check_mv_args(const Vec<T>& x, const Vec<T>& y, std::string_view op) const
  {
    DDM_CHECK(assembled_, "mat: {} called before assemble()", op);
    DDM_CHECK(!x.multivec().overlaps(y.multivec()), "mat: {} requires distinct vectors x and y", op);
    DDM_CHECK(x.size() == cols() && y.size() == rows(), "mat: {} size mismatch, matrix is {}x{}, but x has size {} and y has size {}", op, rows(), cols(), x.size(), y.size());
    DDM_CHECK(x.backend() == backend() && y.backend() == backend(), "mat: {} backend mismatch, matrix is {}, x is {}, y is {}", op, to_string(backend()), to_string(x.backend()),
              to_string(y.backend()));
  }

  virtual void do_add_values(std::span<const Index> rows, std::span<const Index> cols, std::span<const T> vals) = 0;
  virtual void do_assemble() = 0;
  virtual void do_zero_rows(std::span<const Index> rows, T diag) = 0;
  virtual void do_zero_rows_and_cols(std::span<const Index> idx, T diag) = 0;
  virtual void do_mv(const Vec<T>& x, Vec<T>& y) const = 0;
  virtual void do_usmv(T alpha, const Vec<T>& x, Vec<T>& y) const = 0;
  virtual void do_spmm(const MultiVec<T>& X, MultiVec<T>& Y) const = 0;
  virtual void do_get_diag(Vec<T>& diag) const = 0;
  virtual std::unique_ptr<MultiVec<T>> do_create_domain_multivector(Index m) const = 0;
  virtual std::unique_ptr<MultiVec<T>> do_create_range_multivector(Index m) const = 0;
  virtual std::shared_ptr<LocalMat<T>> do_create_like(const Dune::ParameterTree& config, const Pattern& pattern) const = 0;
  virtual std::shared_ptr<LocalMat<T>> do_clone() const = 0;
  virtual void do_info() const = 0;
  virtual void do_scale(const Vec<T>& d) = 0;

  virtual HostCsr<T> do_host_csr() const
  {
    DDM_CHECK(false, "mat: host_csr() is not supported by backend {}", to_string(backend()));
    return {};
  }

  Dune::ParameterTree config_;
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

// Creates a local matrix for a matrix derived from `like` (e.g. a subdomain matrix): of the type given by
// config["type"] if it is set, otherwise of the same type as `like`, so that it has the same backend
// Returns A + alpha * B as a new matrix of the same type as A, on the union of both patterns. Both matrices must be
// assembled, and their backend must support host_csr()
template <class T>
std::shared_ptr<LocalMat<T>> add(const LocalMat<T>& A, T alpha, const LocalMat<T>& B)
{
  DDM_CHECK(A.rows() == B.rows() && A.cols() == B.cols(), "mat: add() size mismatch, A is {}x{}, B is {}x{}", A.rows(), A.cols(), B.rows(), B.cols());
  const auto a = A.host_csr();
  const auto b = B.host_csr();

  Pattern pattern(A.rows(), A.cols());
  for (const auto* csr : {&a, &b})
    for (Index i = 0; i < A.rows(); ++i)
      for (auto k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; ++k) pattern.add(i, csr->cols[k]);
  pattern.finalize();

  auto C = A.create_like(pattern);
  std::vector<T> scaled;
  for (Index i = 0; i < A.rows(); ++i) {
    const auto row = std::span(&i, 1);
    C->add_values(row, std::span(a.cols).subspan(a.row_ptr[i], a.row_ptr[i + 1] - a.row_ptr[i]), std::span(a.values).subspan(a.row_ptr[i], a.row_ptr[i + 1] - a.row_ptr[i]));

    scaled.assign(b.values.begin() + b.row_ptr[i], b.values.begin() + b.row_ptr[i + 1]);
    for (auto& v : scaled) v *= alpha;
    C->add_values(row, std::span(b.cols).subspan(b.row_ptr[i], b.row_ptr[i + 1] - b.row_ptr[i]), scaled);
  }
  C->assemble();
  return C;
}

template <class T>
std::shared_ptr<LocalMat<T>> create_local_mat_like(const Dune::ParameterTree& config, const Pattern& pattern, const LocalMat<T>& like)
{
  if (config.hasKey("type")) return create_local_mat<T>(config, pattern);
  return like.create_like(pattern);
}
} // namespace ddm
