#pragma once

#include "ddm/check.hh"
#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/index.hh"
#include "ddm/logger.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/multivec/multivec.hh"
#include "ddm/vec/vec.hh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra-semi"
#include <Eigen/Dense>
#pragma GCC diagnostic pop

#include <algorithm>
#include <cmath>
#include <dune/common/parametertree.hh>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace ddm {
/** Thick-restart block Lanczos for A x = lambda B x. It computes the same eigenpairs as SpectraEigensolver (up to the
 *  tolerance), but works on blocks of vectors with the multivector operations of the backend.
 *
 *  It runs Lanczos for the shift-invert operator S = P (A - sigma B)^{-1} P^T B, where P and P^T are left out if they
 *  are not set (see Eigensolver::set_projection(), and Eigensolver::set_shift_invert() for a replacement of
 *  (A - sigma B)^{-1}). S is self-adjoint in the B inner product <X, Y> = X^H B Y. The eigenvalues theta of S with the
 *  largest magnitude belong to the eigenvalues lambda = sigma + 1 / theta closest to the shift.
 *
 *  This is the variant without locking of thick-restart block Lanczos (Wu and Simon 2000 for block size 1, Zhou and Saad
 *  2008, the same as SLEPc's Krylov-Schur for a self-adjoint operator). It keeps the decomposition
 *
 *      S V = V T + v E,    [V, v] B-orthonormal,
 *
 *  of the basis V (m columns), the next block v (s columns, the block size), the Hermitian projected matrix
 *  T = <V, S V> (m x m) and the coupling E = <v, S V> (s x m). A Ritz pair (theta, V y) of T then has the residual
 *  S V y - theta V y = v E y, with B-norm ||E y||, so convergence is checked without touching the basis. Each iteration
 *  1. extends the basis by a block Lanczos step. It orthogonalizes w = S v against [V, v] and B-orthonormalizes it,
 *     S v = V E^H + v alpha + w R, and continues with
 *         V <- [V, v],  T <- [T, E^H; E, alpha],  E <- [0, R],  v <- w.
 *     The orthogonalization runs against the whole basis and twice (full reorthogonalization), so that the basis stays
 *     orthonormal in floating point;
 *  2. once the basis has more than ncv - s columns, computes the Ritz pairs of T and stops once the wanted ones (largest
 *     |theta|) have converged, i.e. ||E y|| <= tolerance |theta|. With a threshold (see below), this is checked after
 *     every step once the basis can hold the wanted pairs: the dense eigenproblem of T is cheap compared with applying S
 *     to a block (3 ms for m = 200, 19 ms for m = 400).
 *  3. otherwise, if the basis is full, restarts with the k leading converged Ritz vectors and a fraction keep of the
 *     others (thick restart). With Y these Ritz vectors of T, S (V Y) = (V Y) diag(theta) + v (E Y) is again
 *     a decomposition of the form above:
 *         V <- V Y,  T <- diag(theta),  E <- E Y.
 *
 *  Without a threshold, the wanted Ritz pairs are the nev leading ones. With a threshold tau (see Eigensolver), they are
 *  those with |theta| >= theta_tau = 1 / |tau - sigma| (lambda below tau), at least nev and at most nev_max. Since V is
 *  a B-orthonormal basis and S is self-adjoint in the B inner product, the j-th largest Ritz value is at most the j-th
 *  largest eigenvalue of S (Cauchy interlacing, applied at both ends of the spectrum), so the number w of Ritz values
 *  above theta_tau is a lower bound on the number of eigenvalues there, converged or not. The solver waits for the w + 1
 *  leading Ritz pairs: the first one below theta_tau must have converged as well, otherwise a Ritz value that is still
 *  rising towards an eigenvalue above theta_tau would be cut off. It returns the w leading ones (or nev, or nev_max). The
 *  decomposition above holds for every m, so if 2 (w + 1) exceeds ncv, ncv grows (by at least half) by copying [V, v]
 *  into a larger basis, and the iteration continues without losing work. Like with a fixed nev, and more likely since it
 *  stops as early as possible (see 2.), a Krylov method can miss eigenvalues whose eigenvectors are (nearly) orthogonal to
 *  the starting block, e.g. more copies of a multiple eigenvalue than the block size.
 *
 *  Config (see Eigensolver for the others):
 *  - blocksize:    number of columns s of a block (default 4)
 *  - keep:         fraction of the unconverged Ritz vectors that are kept at a restart (default 0.5)
 *  - solver, prec: config of the solver for A - sigma B, see create_subproblem_solver() (default: exact solve)
 *  ncv is raised to nev + blocksize if it is smaller, so that a restart leaves room for at least one more block. With a
 *  threshold, nev_max is lowered to the size of the problem minus twice the block size if it is larger.
 */
template <class T>
class LanczosEigensolver final : public Eigensolver<T> {
  using Real = typename Vec<T>::real_type;
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>; // small dense matrices on the host
  using RealVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

public:
  static constexpr int default_blocksize = 4;

  LanczosEigensolver(const Dune::ParameterTree& config, std::shared_ptr<const LocalMat<T>> A, std::shared_ptr<const LocalMat<T>> B)
      : Eigensolver<T>(config, std::move(A), std::move(B))
      , config_(config)
      , blocksize_(config.get("blocksize", default_blocksize))
      , keep_(config.get("keep", 0.5))
  {
    DDM_CHECK(blocksize_ >= 1, "lanczos eigensolver: blocksize must be at least 1, got {}", blocksize_);
    DDM_CHECK(0 < keep_ && keep_ < 1, "lanczos eigensolver: keep must be between 0 and 1, got {}", keep_);
  }

  const std::vector<T>& values() const override
  {
    return values_;
  }
  std::shared_ptr<MultiVec<T>> vectors() const override
  {
    return vectors_;
  }

private:
  void do_solve() override
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Eigensolver", "solve")};
    auto* operator_event = Logger::get().registerOrGetEvent("Lanczos", "operator");
    auto* orthonormalize_event = Logger::get().registerOrGetEvent("Lanczos", "orthonormalize");
    auto* ritz_event = Logger::get().registerOrGetEvent("Lanczos", "ritz");
    auto* restart_event = Logger::get().registerOrGetEvent("Lanczos", "restart");

    const Index n = this->A()->rows();
    const Index s = blocksize_;
    const Index nev = this->nev();
    Index ncv = std::max<Index>(this->ncv(), nev + s); // when in `threshold` mode, this will grow during the solve
    DDM_CHECK(ncv + s <= n, "lanczos eigensolver: ncv + blocksize = {} must not exceed the size of the problem {}", ncv + s, n);

    // With a threshold, the wanted Ritz values are those with |theta| >= theta_tau. A restart must leave room for one more
    // block, i.e. ncv >= nev_max + s, and ncv + s must not exceed n (nev <= nev_max still holds by the check above).
    const auto tau = this->threshold();
    const Index nev_max = tau ? std::min<Index>(this->nev_max(), n - 2 * s) : nev;
    const Real theta_tau = tau ? 1 / std::abs(*tau - this->shift()) : 0;

    shift_invert_ = this->make_shift_invert(config_, this->shift());
    BX_ = this->A()->create_domain_multivector(s);
    tmp_ = this->A()->create_domain_multivector(s);
    rng_.seed(0);

    // V is in the columns [0, m) of storage, v in [m, m + s)
    auto storage = this->A()->create_domain_multivector(ncv + s);
    Index m = 0;
    Matrix T_proj(0, 0);
    Matrix E(s, 0);

    // The starting block, see random_block()
    {
      auto v = storage->columns(0, s);
      random_block(*v);
      orthonormalize(*storage, 0, s);
    }

    Index n_op = 0;   // number of columns the operator was applied to
    Index want = nev; // number of leading Ritz pairs that must converge, grows with the Ritz values above the threshold
    int restart = 0;
    for (;;) {
      // 1. Extend the basis by a block Lanczos step
      {
        {
          Logger::ScopedLog sl_op{operator_event};
          apply_operator(*storage->columns(m, s), *storage->columns(m + s, s)); // w = S v
          n_op += s;
        }
        Logger::ScopedLog sl_orth{orthonormalize_event};
        const auto [C, R] = orthonormalize(*storage, m + s, s); // against [V, v]
        const Matrix alpha = C.bottomRows(s);                   // <v, S v>, the rows above it are E^H up to rounding

        Matrix T_next = Matrix::Zero(m + s, m + s);
        T_next.topLeftCorner(m, m) = T_proj;
        T_next.topRightCorner(m, s) = E.adjoint();
        T_next.bottomLeftCorner(s, m) = E;
        T_next.bottomRightCorner(s, s) = (alpha + alpha.adjoint()) / Real(2); // Hermitian up to rounding
        T_proj = std::move(T_next);
        E = Matrix::Zero(s, m + s);
        E.rightCols(s) = R;
        m += s;
      }

      // Without a threshold, check only once the basis is full. With one, after every step: the needed ncv is not known,
      // so a generous one should not cost iterations. The wanted Ritz pairs cannot have converged while there are fewer
      // columns than wanted ones (the basis is never full then, since ncv >= want + s).
      if (m + s <= ncv && (!tau || m < want)) continue;

      // 2. Ritz pairs, and the number k of leading wanted ones that have converged. With a threshold, the number of Ritz
      //    values above theta_tau (lambda below tau) is a lower bound on the number of eigenvalues there, and the first
      //    Ritz pair below theta_tau must converge as well.
      const auto [theta, Y] = [&] {
        Logger::ScopedLog sl_ritz{ritz_event};
        return ritz_pairs(T_proj);
      }();
      Index above = 0;
      if (tau) {
        while (above < m && std::abs(theta(above)) >= theta_tau) ++above;
        want = std::clamp(above + 1, nev, nev_max);
      }
      Index k = 0;
      while (k < std::min(want, m) && (E * Y.col(k)).norm() <= this->tolerance() * std::abs(theta(k))) ++k;
      if (k == want) {
        const Index count = tau ? std::clamp(above, nev, nev_max) : nev;
        logger::debug_all("Lanczos converged after {} restarts and {} operator applications (block size {}, ncv {}, {} eigenpairs)", restart, n_op, s, ncv, count);
        if (tau) log_threshold(theta, count, above, nev_max, *tau);
        set_result(*storage->columns(0, m), theta, Y, count);
        break;
      }

      // 3. With a threshold, make room for twice the wanted pairs, like the default ncv = 2 nev, and for at least one
      //    block after them. Copying [V, v] into a larger basis keeps the decomposition, the iteration goes on where it
      //    was. Since want <= nev_max <= n - 2 s, the cap n - s still leaves room for a block.
      if (tau && 2 * want > ncv && ncv + s < n) {
        const Index grown = std::min(n - s, std::max({2 * want, want + s, ncv + ncv / 2}));
        logger::debug_all("Lanczos: {} Ritz values below the threshold, growing ncv from {} to {}", above, ncv, grown);
        auto larger = this->A()->create_domain_multivector(grown + s);
        larger->columns(0, m + s)->copy_from(*storage->columns(0, m + s));
        storage = std::move(larger);
        ncv = grown;
      }
      if (m + s <= ncv) continue;

      // 4. Thick restart with the k converged Ritz vectors and a fraction of the others, leaving room for one more block
      DDM_CHECK(restart < this->maxit(), "lanczos eigensolver: only {} of {} eigenpairs converged after {} restarts", k, want, restart);
      Logger::ScopedLog sl_restart{restart_event};
      const Index kept = std::min(k + std::max<Index>(1, static_cast<Index>(keep_ * static_cast<double>(m - k))), ncv - s);
      storage->columns(0, m)->mult_in_place(*to_backend(Y.leftCols(kept), *storage));
      tmp_->copy_from(*storage->columns(m, s)); // the old and the new place of v may overlap
      storage->columns(kept, s)->copy_from(*tmp_);
      T_proj = theta.head(kept).template cast<T>().asDiagonal();
      E = E * Y.leftCols(kept);
      m = kept;
      ++restart;
    }

    // Release the factorization of A - sigma B and the buffers
    shift_invert_ = {};
    BX_.reset();
    tmp_.reset();
  }

  // W = P (A - sigma B)^{-1} P^T B X, where P and P^T are left out if they are not set
  void apply_operator(const MultiVec<T>& X, MultiVec<T>& W)
  {
    this->B()->spmm(X, *BX_);
    if (this->projection_transposed()) {
      this->projection_transposed()(*BX_, *tmp_);
      BX_->copy_from(*tmp_);
    }
    if (this->projection()) {
      shift_invert_(*BX_, *tmp_);
      this->projection()(*tmp_, W);
    }
    else shift_invert_(*BX_, W);
  }

  // Orthogonalizes the columns W = X[:, j:j+s) against the B-orthonormal columns V = X[:, 0:j) and B-orthonormalizes
  // them. Returns C and R such that W before and after the call satisfy W_before = V C + W_after R.
  //
  // This is block classical Gram-Schmidt with reorthogonalization (BCGS2): project out V and orthonormalize the block
  // (see svqb()), twice. The first pass leaves errors that grow with the condition number of the block, the second one
  // removes them. Directions of W that are numerically in the span of V (e.g. once the Krylov space contains an
  // invariant subspace) get zero rows in R. They are replaced by new random directions, which are orthonormalized in
  // turn against V and the other directions of W. The first replacement is S times a random block (see random_block()).
  // If that is dependent as well, V contains the range of S to working precision: S Z is dominated by the directions of
  // the large eigenvalues, which V already has, and the rest is at the level of the rounding errors. This happens if the
  // eigenvalues of S decay so quickly that fewer than ncv + blocksize of them are above the rounding errors, e.g. for
  // the harmonic spaces of subdomains with few boundary dofs. The later replacements are plain random blocks then, like
  // in ARPACK and Spectra. They are B-orthogonal to the range of S, so their Ritz values are zero up to rounding, and the
  // wanted Ritz vectors don't contain them.
  std::pair<Matrix, Matrix> orthonormalize(MultiVec<T>& X, Index j, Index s, int attempt = 0)
  {
    const auto V = X.columns(0, j);
    auto W = X.columns(j, s);

    // A direction is dependent if its B-norm after the projection is at the level of the rounding errors of the
    // projection, i.e. below tol times the largest B-norm of a column before it. Directions that are only small (e.g.
    // because the eigenvalues of S decay quickly) are kept, the second pass orthonormalizes them to working precision.
    constexpr Real tol = 1000 * std::numeric_limits<Real>::epsilon();
    std::vector<T> norms2(s);
    W->column_dots(*B_times(*W), norms2);
    Real scale = 0;
    for (auto x : norms2) scale = std::max(scale, std::real(x));

    const Matrix C1 = project_out(*V, *W);
    const auto [R1, good] = svqb(*W, tol * tol * scale);

    // Second pass on the directions that are kept, which are orthonormal now. There is none if the whole block is
    // dependent (svqb() must not see an empty block).
    Matrix C = C1;
    Matrix R = Matrix::Zero(s, s);
    if (good > 0) {
      auto W_good = X.columns(j, good);
      const Matrix C2 = project_out(*V, *W_good);
      const auto [R2, good2] = svqb(*W_good, tol * tol);
      DDM_CHECK(good2 == good, "lanczos eigensolver: the second orthonormalization of a block found dependent directions");

      // W_before = V C1 + W_1 R1 with only the good columns of W_1 contributing, and W_1 = V C2 + W_2 R2 on those columns
      C += C2 * R1.topRows(good);
      R.topRows(good) = R2 * R1.topRows(good);
    }

    if (good < s) {
      DDM_CHECK(attempt < 3, "lanczos eigensolver: cannot extend the Krylov space, ncv + blocksize is probably larger than the rank of B");
      auto Z = X.create_multivector(X.rows(), blocksize_); // apply_operator() needs a whole block
      if (attempt == 0) {
        logger::debug_all("Lanczos: replacing {} dependent directions of a block by S times random vectors", s - good);
        random_block(*Z);
      }
      else {
        logger::debug_all("Lanczos: replacing {} dependent directions of a block by random vectors, the range of S is exhausted", s - good);
        random_fill(*Z);
      }
      X.columns(j + good, s - good)->copy_from(*Z->columns(0, s - good));
      orthonormalize(X, j + good, s - good, attempt + 1); // C and R are not needed, the rows of R of these columns are zero
    }
    return {C, R};
  }

  // W <- W - V <V, W>, returns <V, W>
  Matrix project_out(const MultiVec<T>& V, MultiVec<T>& W)
  {
    const Matrix C = dot(V, *B_times(W));
    V.mult(*to_backend(C, V), W, T(-1), T(1));
    return C;
  }

  // B-orthonormalizes the columns of W in place with the eigendecomposition of their Gram matrix (SVQB, Stathopoulos and
  // Wu 2002): with <W, W> = U Lambda U^H, W <- W U Lambda^{-1/2}, so W_before = W_after R with R = Lambda^{1/2} U^H. The
  // eigenvalues are in decreasing order. Those up to tiny belong to directions in which W is numerically rank deficient:
  // their columns of W and rows of R are set to zero. Returns R and the number of the other ("good") columns, which come
  // first.
  std::pair<Matrix, Index> svqb(MultiVec<T>& W, Real tiny)
  {
    const Index s = W.cols();
    const Eigen::SelfAdjointEigenSolver<Matrix> eig(dot(W, *B_times(W))); // only reads the lower triangle
    DDM_CHECK(eig.info() == Eigen::Success, "lanczos eigensolver: eigendecomposition of a Gram matrix failed");

    Matrix M = Matrix::Zero(s, s); // U Lambda^{-1/2}
    Matrix R = Matrix::Zero(s, s); // Lambda^{1/2} U^H
    Index good = 0;
    for (Index i = 0; i < s; ++i) {
      const Real lambda = eig.eigenvalues()(s - 1 - i); // Eigen sorts them in increasing order
      if (lambda <= tiny) break;
      M.col(i) = eig.eigenvectors().col(s - 1 - i) / std::sqrt(lambda);
      R.row(i) = eig.eigenvectors().col(s - 1 - i).adjoint() * std::sqrt(lambda);
      ++good;
    }
    W.mult_in_place(*to_backend(M, W));
    return {R, good};
  }

  // B W, in a buffer that is valid until the next call
  std::unique_ptr<MultiVec<T>> B_times(const MultiVec<T>& W)
  {
    auto BW = BX_->columns(0, W.cols());
    this->B()->spmm(W, *BW);
    return BW;
  }

  // The eigenpairs of the Hermitian matrix T, ordered by decreasing magnitude of the eigenvalues
  static std::pair<RealVector, Matrix> ritz_pairs(const Matrix& T_proj)
  {
    const Eigen::SelfAdjointEigenSolver<Matrix> eig(T_proj);
    DDM_CHECK(eig.info() == Eigen::Success, "lanczos eigensolver: eigendecomposition of the projected matrix failed");

    const Index m = T_proj.rows();
    std::vector<Index> order(m);
    std::iota(order.begin(), order.end(), Index(0));
    std::ranges::stable_sort(order, [&](Index a, Index b) { return std::abs(eig.eigenvalues()(a)) > std::abs(eig.eigenvalues()(b)); });

    RealVector theta(m);
    Matrix Y(m, m);
    for (Index j = 0; j < m; ++j) {
      theta(j) = eig.eigenvalues()(order[j]);
      Y.col(j) = eig.eigenvectors().col(order[j]);
    }
    return {theta, Y};
  }

  // Logs the number of eigenvalues below the threshold tau and the first one above it, which converged as well (the
  // quantity the bounds of GenEO and MS-GFEM are stated in). Warns if nev_max cut off eigenvalues below tau.
  void log_threshold(const RealVector& theta, Index count, Index above, Index nev_max, Real tau) const
  {
    const auto lambda = [&](Index j) {
      return this->shift() + 1 / theta(j);
    };
    if (above >= nev_max) logger::warn_all("lanczos eigensolver: stopped at nev_max = {} eigenpairs, but the largest one {} is still below the threshold {}", nev_max, lambda(nev_max - 1), tau);
    else logger::debug_all("Lanczos: {} eigenvalues below the threshold {} (returning {}), the next one is {}", above, tau, count, lambda(above));
  }

  // The eigenpairs from the count leading Ritz pairs: lambda = sigma + 1 / theta, sorted from small to large like
  // Spectra's, with the Ritz vectors V y
  void set_result(const MultiVec<T>& V, const RealVector& theta, const Matrix& Y, Index count)
  {
    const auto lambda = [&](Index j) {
      return this->shift() + 1 / theta(j);
    };
    std::vector<Index> order(count);
    std::iota(order.begin(), order.end(), Index(0));
    std::ranges::sort(order, [&](Index a, Index b) { return lambda(a) < lambda(b); });

    values_.clear();
    Matrix Y_sorted(Y.rows(), count);
    for (Index j = 0; j < count; ++j) {
      values_.push_back(T(lambda(order[j])));
      Y_sorted.col(j) = Y.col(order[j]);
    }
    vectors_ = this->A()->create_domain_multivector(count);
    V.mult(*to_backend(Y_sorted, V), *vectors_);
  }

  // X = S Z for a random block Z (X has blocksize columns). Starting with a block in the range of S keeps the whole Krylov
  // space there. Otherwise, if B is only semidefinite (e.g. zero at the Dirichlet dofs), the components of the random
  // block in the kernel of B, which the B inner product does not see, end up in the eigenvectors, and the iteration can
  // amplify them without bound. With a projection, this also puts X into the range of P.
  void random_block(MultiVec<T>& X)
  {
    auto Z = X.create_multivector(X.rows(), X.cols());
    random_fill(*Z);
    apply_operator(*Z, X);
  }

  // Fills X with random numbers, uniformly distributed in [-1, 1] (real and imaginary parts)
  void random_fill(MultiVec<T>& X)
  {
    std::uniform_real_distribution<Real> dist(-1, 1);
    auto view = X.host_view(write);
    for (auto& z : view)
      if constexpr (std::is_same_v<T, Real>) z = dist(rng_);
      else z = T(dist(rng_), dist(rng_));
  }

  // X^H Y as a host matrix
  static Matrix dot(const MultiVec<T>& X, const MultiVec<T>& Y)
  {
    auto C = X.create_multivector(X.cols(), Y.cols());
    X.dot(Y, *C);
    const auto view = std::as_const(*C).host_view(read);
    return Eigen::Map<const Matrix>(view.data(), X.cols(), Y.cols());
  }

  // The host matrix M as a multivector on the backend of like, e.g. as the coefficients for MultiVec::mult()
  static std::unique_ptr<MultiVec<T>> to_backend(const Matrix& M, const MultiVec<T>& like)
  {
    auto X = like.create_multivector(M.rows(), M.cols());
    {
      auto view = X->host_view(write);
      Eigen::Map<Matrix>(view.data(), M.rows(), M.cols()) = M;
    }
    return X;
  }

  Dune::ParameterTree config_;
  int blocksize_;
  double keep_;

  // Set up for the duration of a solve
  typename Eigensolver<T>::ShiftInvert shift_invert_; ///< (A - sigma B)^{-1} or its replacement
  std::unique_ptr<MultiVec<T>> BX_;                   ///< B times a block
  std::unique_ptr<MultiVec<T>> tmp_;                  ///< a block
  std::mt19937 rng_;

  std::vector<T> values_;
  std::shared_ptr<MultiVec<T>> vectors_;
};
} // namespace ddm
