#pragma once

#include "ddm/backend_id.hh"
#include "ddm/check.hh"
#include "ddm/communication.hh"
#include "ddm/index.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/vec/vec.hh"

#include <cstddef>
#include <dune/common/parametertree.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvercategory.hh>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ddm {
// A matrix that is stored additively across the ranks (like PETSc's MATIS): every rank holds a LocalMat<T> in its
// local numbering, and the global matrix is the sum of the local ones. The Communication relates the local indices of
// the ranks to each other. Without a Communication, the matrix is sequential and just the local matrix.
//
// As a linear operator, it expects consistent vectors (every rank holding an index has the same value for it) and
// returns consistent vectors: apply() multiplies with the local matrix and sums the result over all holders.
//
// The matrix is assembled locally: add_values() and zero_rows() take local indices and only change the local matrix.
template <class T>
class Mat final : public Dune::LinearOperator<Vec<T>, Vec<T>> {
public:
  using field_type = T;

  // Collective if comm is not nullptr
  Mat(std::shared_ptr<LocalMat<T>> local, std::shared_ptr<Communication> comm)
      : local_(std::move(local))
      , comm_(std::move(comm))
  {
    DDM_CHECK(local_ != nullptr, "mat: local matrix is nullptr");
    if (comm_) {
      DDM_CHECK(local_->rows() == local_->cols(), "mat: a parallel matrix must be square, but the local matrix is {}x{}", local_->rows(), local_->cols());
      compute_multiplicity();
    }
  }

  Mat(const Mat&) = delete;
  Mat& operator=(const Mat&) = delete;

  // The sizes are the local sizes
  Index rows() const { return local_->rows(); }
  Index cols() const { return local_->cols(); }
  BackendId backend() const { return local_->backend(); }

  Dune::SolverCategory::Category category() const override { return comm_ ? Dune::SolverCategory::overlapping : Dune::SolverCategory::sequential; }

  // See LocalMat::add_values(), the indices are local
  void add_values(std::span<const Index> rows, std::span<const Index> cols, std::span<const T> vals) { local_->add_values(rows, cols, vals); }

  // Must be called after the last add_values() and before the matrix is used
  void assemble() { local_->assemble(); }

  bool is_assembled() const { return local_->is_assembled(); }

  // Sets the given rows of the global matrix to zero and their diagonal entries to diag. Every rank holding one of the
  // rows must pass it: each of them sets its local diagonal entry to diag / (number of holders), so that the sum is
  // diag and the local matrices stay invertible.
  void zero_rows(std::span<const Index> rows, T diag = T{1})
  {
    if (!comm_) {
      local_->zero_rows(rows, diag);
      return;
    }

    for (auto i : rows) {
      DDM_CHECK(0 <= i && i < this->rows(), "mat: zero_rows() row {} out of range for {} rows", i, this->rows());
      local_->zero_rows(std::span(&i, 1), diag / multiplicity_[static_cast<std::size_t>(i)]);
    }
  }

  // y = A x. Collective
  void apply(const Vec<T>& x, Vec<T>& y) const override
  {
    local_->mv(x, y);
    if (comm_) comm_->reduce(y);
  }

  // y = y + alpha * A x. Collective
  void applyscaleadd(T alpha, const Vec<T>& x, Vec<T>& y) const override
  {
    if (!comm_) {
      local_->usmv(alpha, x, y);
      return;
    }

    // y is consistent already, so only alpha * A x must be summed up
    if (!tmp_) tmp_.emplace(local_->create_range_vector());
    tmp_->zero();
    local_->usmv(alpha, x, *tmp_);
    comm_->reduce(*tmp_);
    y += *tmp_;
  }

  // Returns a zero-initialized vector x from the domain of the matrix, i.e. one that can be used in apply(x, y)
  Vec<T> create_domain_vector() const { return local_->create_domain_vector(); }

  // Returns a zero-initialized vector y from the range of the matrix, i.e. one that can be used in apply(x, y)
  Vec<T> create_range_vector() const { return local_->create_range_vector(); }

  // Returns zero-initialized vectors {x, y} that are compatible with apply(x, y)
  std::pair<Vec<T>, Vec<T>> create_vectors() const { return {create_domain_vector(), create_range_vector()}; }

  // Fills the given vector with the diagonal of the global matrix (consistent). Collective
  void get_diag(Vec<T>& diag) const
  {
    local_->get_diag(diag);
    if (comm_) comm_->reduce(diag);
  }

  LocalMat<T>& local() { return *local_; }
  const LocalMat<T>& local() const { return *local_; }

  // nullptr if the matrix is sequential
  const std::shared_ptr<Communication>& communication() const { return comm_; }

  // Returns true if this is a sequential matrix (i.e. comm is nullptr)
  bool sequential() const { return comm_ == nullptr; }

private:
  // The number of ranks holding each local index
  void compute_multiplicity()
  {
    auto ones = local_->create_range_vector();
    ones = T{1};
    comm_->reduce(ones);

    auto view = ones.host_view(read);
    multiplicity_.assign(view.begin(), view.end());
  }

  std::shared_ptr<LocalMat<T>> local_;
  std::shared_ptr<Communication> comm_;

  std::vector<T> multiplicity_;
  mutable std::optional<Vec<T>> tmp_; ///< temporary for applyscaleadd()
};

// Creates a matrix whose local matrix has the type given by config["type"]. The pattern must be finalized. comm may
// be nullptr for a sequential matrix. Collective if comm is not nullptr
template <class T>
std::shared_ptr<Mat<T>> create_mat(const Dune::ParameterTree& config, const Pattern& pattern, std::shared_ptr<Communication> comm = nullptr)
{
  return std::make_shared<Mat<T>>(create_local_mat<T>(config, pattern), std::move(comm));
}
} // namespace ddm
