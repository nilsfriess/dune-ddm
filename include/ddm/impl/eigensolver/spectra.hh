#pragma once

#ifdef DUNE_DDM_HAVE_SPECTRA

#include "ddm/check.hh"
#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/logger.hh"
#include "ddm/mat/local_mat.hh"
#include "ddm/mat/mat.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wextra-semi"
#include <Spectra/HermEigsBase.h>
#include <Spectra/MatOp/internal/SymGEigsShiftInvertOp.h>
#include <Spectra/Util/CompInfo.h>
#include <Spectra/Util/SimpleRandom.h>
#pragma GCC diagnostic pop

#include <algorithm>
#include <complex>
#include <dune/common/parametertree.hh>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace ddm {
namespace detail::spectra {
// The shift-invert operator (A - sigma B)^{-1}, the OpType of Spectra's generalised shift-invert solver, followed by the
// projection P if one is set and preceded by its transpose P^T if that is set as well (see
// Eigensolver::set_projection()). Spectra calls set_shift() once before the iteration, which sets up (A - sigma B)^{-1}
// or its replacement (see Eigensolver::make_shift_invert()).
template <class T>
class ShiftInvertOp {
public:
  using Scalar = T;

  ShiftInvertOp(std::shared_ptr<const LocalMat<T>> A,
                typename Eigensolver<T>::ShiftInvertFactory make_shift_invert,
                typename Eigensolver<T>::Projection projection,
                typename Eigensolver<T>::Projection projection_transposed)
      : A_(std::move(A))
      , make_shift_invert_(std::move(make_shift_invert))
      , projection_(std::move(projection))
      , projection_transposed_(std::move(projection_transposed))
      , b_(A_->create_range_vector())
      , pt_(A_->create_range_vector())
      , x_(A_->create_domain_vector())
      , p_(A_->create_domain_vector())
  {
  }

  Eigen::Index rows() const
  {
    return A_->rows();
  }
  Eigen::Index cols() const
  {
    return A_->cols();
  }

  void set_shift(T sigma)
  {
    solve_ = make_shift_invert_(std::real(sigma));
  }

  // y = P (A - sigma B)^{-1} P^T x, where P and P^T are left out if they are not set and (A - sigma B)^{-1} is
  // replaced if a replacement was set
  void perform_op(const T* x_in, T* y_out) const
  {
    {
      auto b = b_.host_view(write);
      std::copy_n(x_in, b.size(), b.data());
    }
    if (projection_transposed_) projection_transposed_(b_.multivec(), pt_.multivec());
    solve_((projection_transposed_ ? pt_ : b_).multivec(), x_.multivec());
    if (projection_) projection_(x_.multivec(), p_.multivec());

    const auto y = std::as_const(projection_ ? p_ : x_).host_view(read);
    std::copy_n(y.data(), y.size(), y_out);
  }

private:
  std::shared_ptr<const LocalMat<T>> A_;
  typename Eigensolver<T>::ShiftInvertFactory make_shift_invert_;
  typename Eigensolver<T>::Projection projection_;
  typename Eigensolver<T>::Projection projection_transposed_;
  typename Eigensolver<T>::ShiftInvert solve_; ///< (A - sigma B)^{-1} or its replacement, set up by set_shift()

  mutable Vec<T> b_;
  mutable Vec<T> pt_; ///< P^T applied to b_
  mutable Vec<T> x_;
  mutable Vec<T> p_; ///< the projection of x_
};

// The product y = B x, the BOpType of Spectra's generalised solver
template <class T>
class MatOp {
public:
  using Scalar = T;

  explicit MatOp(std::shared_ptr<const LocalMat<T>> B)
      : B_(std::move(B))
      , x_(B_->create_domain_vector())
      , y_(B_->create_range_vector())
  {
  }

  void perform_op(const T* x_in, T* y_out) const
  {
    {
      auto x = x_.host_view(write);
      std::copy_n(x_in, x.size(), x.data());
    }
    B_->mv(x_, y_);
    const auto y = std::as_const(y_).host_view(read);
    std::copy_n(y.data(), y.size(), y_out);
  }

private:
  std::shared_ptr<const LocalMat<T>> B_;

  mutable Vec<T> x_;
  mutable Vec<T> y_;
};

// Spectra's SymGEigsShiftSolver in shift-invert mode, but with a real shift, so that it also works for complex Hermitian
// problems. Spectra's version stores the shift as OpType::Scalar, which does not compile for complex Scalar: the Ritz
// values are real, and transforming them back with a complex shift gives complex values.
// TODO: Use Spectra::SymGEigsShiftSolver again once Spectra supports complex Scalar there
template <class OpType, class BOpType>
class HermGEigsShiftSolver final : public Spectra::HermEigsBase<Spectra::SymGEigsShiftInvertOp<OpType, BOpType>, BOpType> {
  using Scalar = typename OpType::Scalar;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  using ModeMatOp = Spectra::SymGEigsShiftInvertOp<OpType, BOpType>;
  using Base = Spectra::HermEigsBase<ModeMatOp, BOpType>;
  using Base::m_nev;
  using Base::m_ritz_val;

  const RealScalar sigma_;

  static ModeMatOp set_shift_and_move(ModeMatOp&& op, RealScalar sigma)
  {
    op.set_shift(Scalar(sigma));
    return std::move(op);
  }

  // The iteration computes the eigenvalues nu = 1 / (lambda - sigma) of the shift-invert operator
  void sort_ritzpair(Spectra::SortRule sort_rule) override
  {
    m_ritz_val.head(m_nev).array() = RealScalar(1) / m_ritz_val.head(m_nev).array() + sigma_;
    Base::sort_ritzpair(sort_rule);
  }

public:
  HermGEigsShiftSolver(OpType& op, BOpType& Bop, Eigen::Index nev, Eigen::Index ncv, RealScalar sigma)
      : Base(set_shift_and_move(ModeMatOp(op, Bop), sigma), Bop, nev, ncv)
      , sigma_(sigma)
  {
  }
};

inline std::string_view to_string(Spectra::CompInfo info)
{
  switch (info) {
    case Spectra::CompInfo::Successful: return "successful";
    case Spectra::CompInfo::NotComputed: return "not computed";
    case Spectra::CompInfo::NotConverging: return "not converging";
    case Spectra::CompInfo::NumericalIssue: return "numerical issue";
  }
  return "unknown";
}
} // namespace detail::spectra

/** Eigensolver based on Spectra's shift-invert Lanczos for generalised symmetric or Hermitian problems (see
 *  detail::spectra::HermGEigsShiftSolver). The shifted matrix A - sigma B is solved with create_subproblem_solver(),
 *  configured by the subtrees "solver" and "prec" (default: exact solve). Runs on the host, so the matrices must support
 *  host_csr(). See Eigensolver for the other parameters.
 */
template <class T>
class SpectraEigensolver final : public Eigensolver<T> {
public:
  SpectraEigensolver(const Dune::ParameterTree& config, std::shared_ptr<const LocalMat<T>> A, std::shared_ptr<const LocalMat<T>> B)
      : Eigensolver<T>(config, std::move(A), std::move(B))
      , config_(config)
  {
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
    const int n = this->A()->rows();
    const auto nev = this->nev();
    const auto ncv = std::min(this->ncv(), n); // Spectra requires ncv <= n
    DDM_CHECK(!this->threshold(), "spectra eigensolver: threshold is not supported, use the lanczos eigensolver");
    DDM_CHECK(nev < n, "spectra eigensolver: nev must be smaller than the size of the problem, got nev = {} for size {}", nev, n);

    detail::spectra::ShiftInvertOp<T> op(this->A(), [this](typename Vec<T>::real_type sigma) { return this->make_shift_invert(config_, sigma); }, this->projection(), this->projection_transposed());
    detail::spectra::MatOp<T> Bop(this->B());
    detail::spectra::HermGEigsShiftSolver<detail::spectra::ShiftInvertOp<T>, detail::spectra::MatOp<T>> geigs(op, Bop, nev, ncv, this->shift());
    if (this->projection()) {
      // The starting vector must lie in the range of the projection as well, otherwise so does the Krylov space. This
      // is the random vector of Spectra's init(), projected.
      auto v0 = this->A()->create_domain_vector();
      auto p0 = this->A()->create_domain_vector();
      {
        const auto random = Spectra::SimpleRandom<T>(0).random_vec(n);
        auto view = v0.host_view(write);
        std::copy_n(random.data(), n, view.data());
      }
      this->projection()(v0.multivec(), p0.multivec());
      const auto view = std::as_const(p0).host_view(read);
      geigs.init(view.data());
    }
    else geigs.init();

    // The largest eigenvalues of the shift-invert operator are the ones of the original problem closest to the shift,
    // sorted from small to large
    const auto nconv = geigs.compute(Spectra::SortRule::LargestMagn, this->maxit(), this->tolerance(), Spectra::SortRule::SmallestAlge);
    DDM_CHECK(geigs.info() == Spectra::CompInfo::Successful, "spectra eigensolver: computation failed ({}), {} of {} eigenpairs converged", detail::spectra::to_string(geigs.info()), nconv, nev);

    const auto evalues = geigs.eigenvalues();
    values_.assign(evalues.data(), evalues.data() + evalues.size());
    logger::debug_all("Spectra computed {} eigenvalues: smallest {}, largest {}", values_.size(), values_.front(), values_.back());

    // Eigen matrices are column-major, like the data of a multivector
    const auto evecs = geigs.eigenvectors();
    vectors_ = this->A()->create_domain_multivector(evecs.cols());
    auto view = vectors_->host_view(write);
    std::copy_n(evecs.data(), evecs.size(), view.data());
  }

  Dune::ParameterTree config_;

  std::vector<T> values_;
  std::shared_ptr<MultiVec<T>> vectors_;
};
} // namespace ddm

#endif
