#pragma once

#if DUNE_DDM_HAVE_STRUMPACK

#include <dune/istl/bvector.hh>
#include <dune/istl/foreach.hh>
#include <dune/istl/solver.hh>

#include <StrumpackOptions.hpp>
#include <StrumpackSparseSolver.hpp>

#include <mpi.h>

namespace Dune {
template <typename M, typename V = BlockVector<FieldVector<double, 1>>>
class STRUMPACK : public InverseOperator<V, V> {
public:
  using T = typename M::field_type;

  using X = BlockVector<FieldVector<double, 1>>;
  using Y = BlockVector<FieldVector<double, 1>>;

  explicit STRUMPACK(const M &A) : solver(false, true)
  {
    auto &options = solver.options();
    options.set_Krylov_solver(strumpack::KrylovSolver::DIRECT);
    options.set_reordering_method(strumpack::ReorderingStrategy::METIS);
    options.set_verbose(false);

    int nnz = 0;
    const auto [fr, fc] = flatMatrixForEach(A, [&](auto &&, auto &&, auto &&) { ++nnz; });
    int N = static_cast<int>(fr);

    std::vector<int> row_ptr(N + 1, 0);
    std::vector<int> col_ind(nnz, 0);
    std::vector<double> values(nnz, 0);

    // Count number of entries per row and accumulate
    flatMatrixForEach(A, [&](auto &&, auto &&row, auto &&) { row_ptr[row + 1] += 1; });
    for (int i = 0; i < N; ++i) {
      row_ptr[i + 1] += row_ptr[i];
    }

    assert(row_ptr[N] == nnz);

    // Now fill the other two arrays
    std::vector<int> row_pos(N, 0); // position counter in each row
    flatMatrixForEach(A, [&](auto &&entry, auto &&row, auto &&col) {
      auto row_start = row_ptr[row];

      col_ind[row_start + row_pos[row]] = col;
      values[row_start + row_pos[row]] = entry;

      row_pos[row] += 1;
    });

    solver.set_csr_matrix(N, row_ptr.data(), col_ind.data(), values.data());
    solver.factor();
  }

  SolverCategory::Category category() const override { return SolverCategory::Category::sequential; }

  void apply(T *x, const T *b) { solver.solve(b, x); }

  void apply(X &x, Y &b, InverseOperatorResult &res) override
  {
    std::vector<T> x_flat(x.dim());
    std::vector<T> b_flat(b.dim());

    flatVectorForEach(x, [&](auto &&entry, auto &&offset) { x_flat[offset] = entry; });
    flatVectorForEach(b, [&](auto &&entry, auto &&offset) { b_flat[offset] = entry; });

    solver.solve(b_flat.data(), x_flat.data());

    flatVectorForEach(x, [&](auto &&entry, auto &&offset) { entry = x_flat[offset]; });

    res.iterations = 1;
    res.converged = true;
  }
  void apply(X &x, Y &b, [[maybe_unused]] double reduction, InverseOperatorResult &res) override { apply(x, b, res); }

private:
  strumpack::SparseSolver<double> solver;
};
} // namespace Dune

#endif
