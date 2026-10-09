#pragma once

#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/index.hh"
#include "ddm/mat/mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/overlap.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"

#include <dune/common/parametertree.hh>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ddm {
// The space H of discrete A-harmonic functions on the overlapping subdomain: (A u)_i = 0 for the interior dofs,
// u_i = 0 for the Dirichlet dofs, the boundary dofs are free.
template <class T>
class HarmonicSpace {
public:
  // A:         the operator that defines harmonicity (the overlapping matrix of the actual operator)
  // boundary:  the outermost layer without the Dirichlet dofs, sorted
  // dirichlet: the Dirichlet dofs, sorted
  // A_dir:     solver for A with identity rows on boundary \cup dirichlet, nullptr if not available
  HarmonicSpace(std::shared_ptr<const Mat<T>> A, std::vector<Index> boundary, std::vector<Index> dirichlet, std::shared_ptr<Solver<T>> A_dir = nullptr)
      : A_(std::move(A))
      , boundary_(std::move(boundary))
      , dirichlet_(std::move(dirichlet))
      , A_dir_(std::move(A_dir))
  {
    // Interior dofs are those that are neither boundary nor Dirichlet dofs; the following algorithm works
    // because boundary and dirichlet are sorted by assumption
    interior_.reserve(A_->rows());
    std::size_t i_boundary = 0;
    std::size_t i_dirichlet = 0;
    for (Index i = 0; i < A_->rows(); ++i) {
      if (i_boundary < boundary_.size() && boundary_[i_boundary] == i) {
        i_boundary++;
        continue;
      }

      if (i_dirichlet < dirichlet_.size() && dirichlet_[i_dirichlet] == i) {
        i_dirichlet++;
        continue;
      }

      // The current i is neither in the boundary list nor in the dirichlet list, so it's an interior dof
      interior_.push_back(i);
    }
    const auto sum = boundary_.size() + dirichlet_.size() + interior_.size();
    DDM_ASSERT(sum == std::size_t(A_->rows()), "HarmonicSpace: the sum of the number of partitioned dofs {} does not equal the total number of dofs in the subdomain {}", sum, A_->rows());
  }

  const Mat<T>& constraint_matrix() const { return *A_; }
  std::span<const Index> boundary() const { return boundary_; }
  std::span<const Index> dirichlet() const { return dirichlet_; }
  std::span<const Index> interior() const { return interior_; }
  bool can_project() const { return A_dir_ != nullptr; }

  // y = harmonic function with the boundary values of x (projection onto H). Requires A_dir
  void project(const Vec<T>& x, Vec<T>& y) const
  {
    DDM_CHECK(can_project(), "HarmonicSpace: project() requires a solver for the matrix, pass A_dir to the constructor");
    if (!rhs_.has_value()) rhs_.emplace(A_->create_range_vector());

    {
      rhs_->zero();
      auto x_host = x.host_view(read);
      auto rhs_host = rhs_->host_view(read_write);
      for (auto idx : boundary_) rhs_host[idx] = x_host[idx];
    }
    A_dir_->solve(*rhs_, y);
  }

private:
  std::shared_ptr<const Mat<T>> A_;
  std::vector<Index> boundary_;
  std::vector<Index> dirichlet_;
  std::vector<Index> interior_;
  std::shared_ptr<Solver<T>> A_dir_;
  mutable std::optional<Vec<T>> rhs_; ///< x with everything except the boundary values set to zero
};

namespace detail {

/* Enforces x in H with Lagrange multipliers mu, one per interior dof, i.e. solves the saddle point eigenproblem
 *
 *   [ A  C^T ] [x ]          [ B  0 ] [x ]
 *   [ C  0   ] [mu] = lambda [ 0  0 ] [mu]
 *
 * where C consists of the interior rows of the constraint matrix. The Dirichlet dofs are not unknowns (x is zero
 * there), so x only has the interior and boundary dofs, numbered interior first, then boundary. The multipliers
 * follow after them.
 */
template <class T>
std::shared_ptr<MultiVec<T>> solve_harmonic_gevp_saddle_point(const Dune::ParameterTree& config, std::shared_ptr<const LocalMat<T>> A, std::shared_ptr<const LocalMat<T>> B, const HarmonicSpace<T>& H)
{
  const Index n = A->rows();
  const Index n_interior = std::ssize(H.interior());
  const Index n_x = n_interior + std::ssize(H.boundary());
  const Index n_big = n_x + n_interior; // (n_interior + n_boundary) + n_interior  (upper block + lower block)

  // Position of every dof in the saddle point system, -1 for the Dirichlet dofs
  std::vector<Index> map(n, -1);
  for (Index k = 0; k < n_interior; ++k) map[H.interior()[k]] = k;
  for (Index k = 0; k < n_x - n_interior; ++k) map[H.boundary()[k]] = n_interior + k;

  const auto a = A->host_csr();
  const auto b = B->host_csr();
  const auto c = H.constraint_matrix().local().host_csr();

  // Calls f(row, col, value) for all entries of csr in rows and columns that are not Dirichlet dofs, in the numbering
  // of the saddle point system. If interior_rows_only, only the interior rows are visited.
  const auto for_each_entry = [&](const HostCsr<T>& csr, bool interior_rows_only, auto&& f) {
    for (Index i = 0; i < n; ++i) {
      const auto r = map[i];
      if (r < 0 || (interior_rows_only && r >= n_interior)) continue;
      for (auto k = csr.row_ptr[i]; k < csr.row_ptr[i + 1]; ++k)
        if (const auto col = map[csr.cols[k]]; col >= 0) f(r, col, csr.values[k]);
    }
  };

  // Left-hand side: A in the upper left block, C below it and C^T to its right
  Pattern lhs_pattern(n_big, n_big);
  for_each_entry(a, false, [&](Index r, Index col, T) { lhs_pattern.add(r, col); });
  for_each_entry(c, true, [&](Index r, Index col, T) {
    lhs_pattern.add(n_x + r, col);
    lhs_pattern.add(col, n_x + r);
  });
  lhs_pattern.finalize();

  auto lhs = A->create_like(lhs_pattern);
  for_each_entry(a, false, [&](Index r, Index col, T v) { lhs->add_values(std::span(&r, 1), std::span(&col, 1), std::span(&v, 1)); });
  for_each_entry(c, true, [&](Index r, Index col, T v) {
    const Index mu = n_x + r;
    lhs->add_values(std::span(&mu, 1), std::span(&col, 1), std::span(&v, 1));
    lhs->add_values(std::span(&col, 1), std::span(&mu, 1), std::span(&v, 1));
  });
  lhs->assemble();

  // Right-hand side: B in the upper left block, zero for the multipliers
  Pattern rhs_pattern(n_big, n_big);
  for_each_entry(b, false, [&](Index r, Index col, T) { rhs_pattern.add(r, col); });
  rhs_pattern.finalize();

  auto rhs = B->create_like(rhs_pattern);
  for_each_entry(b, false, [&](Index r, Index col, T v) { rhs->add_values(std::span(&r, 1), std::span(&col, 1), std::span(&v, 1)); });
  rhs->assemble();

  auto eigensolver = create_eigensolver<T>(config.sub("eigensolver"), lhs, rhs);
  eigensolver->solve();

  // Drop the multipliers and go back to the original numbering, with zeros at the Dirichlet dofs
  const auto big = eigensolver->vectors();
  auto vectors = std::shared_ptr<MultiVec<T>>(A->create_domain_multivector(big->cols()));
  {
    const auto big_view = std::as_const(*big).host_view(read);
    auto view = vectors->host_view(write);

    const std::size_t m_big = n_big;
    const std::size_t cols = big->cols();
    for (std::size_t col = 0; col < cols; ++col)
      for (std::size_t i = 0; i < std::size_t(n); ++i) {
        const auto r = map[i];
        view.data()[col * n + i] = r < 0 ? T{0} : big_view.data()[col * m_big + r];
      }
  }
  return vectors;
}

template <class T>
std::shared_ptr<MultiVec<T>> solve_harmonic_gevp_projected(const Dune::ParameterTree& config, std::shared_ptr<const LocalMat<T>> A, std::shared_ptr<const LocalMat<T>> B, const HarmonicSpace<T>& H)
{
  DDM_CHECK(H.can_project(), "solve_harmonic_gevp_projected(): the HarmonicSpace object must be able to project into the space, pass a suitable solver to its constructor");
  auto eigensolver = create_eigensolver<T>(config.sub("eigensolver"), A, B);
  eigensolver->set_projection([&](const Vec<T>& x, Vec<T>& y) { H.project(x, y); });
  eigensolver->solve();
  return eigensolver->vectors();
}

} // namespace detail

/** Solves A x = lambda B x for x in H, computing the eigenpairs with the smallest eigenvalues.
 *
 *  Config:
 *  - method:      saddle_point | projected | basis (default saddle_point); projected and basis need H.can_extend()
 *  - eigensolver: config of the eigensolver (see create_eigensolver())
 */
template <class T>
std::shared_ptr<MultiVec<T>> solve_harmonic_gevp(const Dune::ParameterTree& config, std::shared_ptr<const LocalMat<T>> A, std::shared_ptr<const LocalMat<T>> B, const HarmonicSpace<T>& H)
{
  // TODO: Check sizes match

  auto method = config.get("method", "saddle_point");
  if (method == "saddle_point") return detail::solve_harmonic_gevp_saddle_point(config, A, B, H);
  else if (method == "projected") return detail::solve_harmonic_gevp_projected(config, A, B, H);
  else DDM_CHECK(false, "solve_harmonic_gevp() unknown method `{}`, known: {}, {}", method, "saddle_point", "projected");
  return nullptr;
}
} // namespace ddm
