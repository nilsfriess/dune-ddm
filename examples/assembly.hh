#pragma once

#include "ddm/check.hh"
#include "ddm/communication.hh"
#include "ddm/index.hh"
#include "ddm/mat/mat.hh"
#include "ddm/mat/pattern.hh"
#include "ddm/vec/vec.hh"

#include <algorithm>
#include <cstddef>
#include <dune/common/parametertree.hh>
#include <dune/grid/common/gridenums.hh>
#include <dune/grid/common/partitionset.hh> // used, but not included by localassemblerenginebase.hh
#include <dune/pdelab/constraints/common/constraintstransformation.hh>
#include <dune/pdelab/gridfunctionspace/localvector.hh>
#include <dune/pdelab/gridoperator/common/localassemblerenginebase.hh>
#include <dune/pdelab/gridoperator/common/localmatrix.hh>
#include <dune/pdelab/localoperator/callswitch.hh>
#include <memory>
#include <type_traits>
#include <vector>

namespace ddm::examples {
namespace detail {
/** Engine for PDELab's DefaultAssembler, the grid loop of a GridOperator: go.assembler().assemble(engine). Instead of
 *  into a PDELab backend, it assembles the sparsity pattern into a ddm::Pattern, the Jacobian into a ddm::Mat or the
 *  residual into the host view of a ddm::Vec, depending on Target. As PDELab's own engines, it calls the local operator
 *  through LocalAssemblerCallSwitch, i.e. only the methods enabled by its do* flags.
 *
 *  The ddm objects are additive, so every element and every face is assembled on exactly one rank:
 *  - An element on the rank where it is interior, ghost and overlap elements are skipped.
 *  - A face from the element with the smaller global id. The engine lets the assembler visit every face from both
 *    sides, because PDELab's one-sided rule compares rank-local indices and can pick a different side on each rank. A
 *    local operator with doSkeletonTwoSided gets both sides.
 *  The Jacobian gets one add_values() call per element (volume and boundary terms) and one per face (the four blocks of
 *  the skeleton terms), so these blocks can be recorded (for use with GenEO or MS-GFEM). Constraints are not applied.
 *
 *  The dofs are the flat container indices of the function space, i.e. it must not use blocking.
 */
template <class GO, class Target>
class Engine : public Dune::PDELab::LocalAssemblerEngineBase {
  using LOP = typename GO::Traits::LocalAssembler::LocalOperator;
  using X = typename GO::Traits::Domain;
  using DF = typename GO::Traits::DomainField;
  using RF = typename GO::Traits::RangeField;
  using JF = typename GO::Traits::JacobianField;
  using IdSet = typename GO::Traits::TrialGridFunctionSpace::Traits::GridView::Grid::GlobalIdSet;
  static_assert(std::is_same_v<typename GO::Traits::TrialGridFunctionSpace, typename GO::Traits::TestGridFunctionSpace>, "only Galerkin methods are supported");

  static constexpr bool pattern = std::is_same_v<Target, ddm::Pattern>;
  static constexpr bool jacobian = std::is_same_v<Target, ddm::Mat<JF>>;
  static constexpr bool residual = std::is_same_v<Target, ddm::HostView<RF>>;
  static_assert(pattern || jacobian || residual, "the target must be a Pattern, a Mat or the HostView of a Vec");

  template <bool flag>
  using Call = Dune::PDELab::LocalAssemblerCallSwitch<LOP, flag>;

public:
  // x is the point of linearisation (not used for the pattern)
  Engine(const GO& go, const X& x, Target& target)
      : lop_(go.localAssembler().localOperator())
      , ids_(go.trialGridFunctionSpace().gridView().grid().globalIdSet())
      , x_(x)
      , target_(target)
  {
  }

  bool requireSkeleton() const
  {
    return requireUVSkeleton() || requireVSkeleton();
  }
  bool requireSkeletonTwoSided() const
  {
    return true;
  }
  bool requireUVSkeleton() const
  {
    return LOP::doAlphaSkeleton;
  }
  bool requireVSkeleton() const
  {
    return residual && LOP::doLambdaSkeleton;
  }
  bool requireUVBoundary() const
  {
    return !pattern && LOP::doAlphaBoundary;
  }
  bool requireVBoundary() const
  {
    return residual && LOP::doLambdaBoundary;
  }
  bool requireUVVolumePostSkeleton() const
  {
    return !pattern && LOP::doAlphaVolumePostSkeleton;
  }
  bool requireVVolumePostSkeleton() const
  {
    return residual && LOP::doLambdaVolumePostSkeleton;
  }

  template <class CU, class CV>
  bool needsConstraintsCaching(const CU&, const CV&) const
  {
    return false;
  }

  template <class EG>
  bool skipEntity(const EG& eg) const
  {
    return eg.entity().partitionType() != Dune::InteriorEntity || Call<LOP::doSkipEntity>::skip_entity(lop_, eg);
  }

  template <class IG>
  bool skipIntersection(const IG& ig) const
  {
    return Call<LOP::doSkipIntersection>::skip_intersection(lop_, ig);
  }

  // The element: dofs, coefficients and local matrix/vector

  template <class EG, class LFSVC>
  void onBindLFSV(const EG&, const LFSVC& lfsv_cache)
  {
    bind(lfsv_cache, dofs_s_);
    if constexpr (pattern)
      for (auto i : dofs_s_)
        for (auto j : dofs_s_) target_.add(i, j);
    if constexpr (residual) r_s_.assign(dofs_s_.size(), 0.0);
  }

  template <class EG, class LFSUC, class LFSVC>
  void onBindLFSUV(const EG&, const LFSUC&, const LFSVC&)
  {
    if constexpr (jacobian) a_ss_.assign(dofs_s_.size(), dofs_s_.size(), 0.0);
  }

  template <class LFSUC>
  void loadCoefficientsLFSUInside(const LFSUC& lfsu_cache)
  {
    load(lfsu_cache, x_s_);
  }

  template <class EG, class LFSUC, class LFSVC>
  void onUnbindLFSUV(const EG&, const LFSUC&, const LFSVC&)
  {
    if constexpr (jacobian) target_.add_values(dofs_s_, dofs_s_, a_ss_.base()); // both column-major
  }

  template <class EG, class LFSVC>
  void onUnbindLFSV(const EG&, const LFSVC&)
  {
    if constexpr (residual) scatter(dofs_s_, r_s_);
  }

  // The neighbour across a face, and the face matrix of the four blocks for face_dofs_ (the dofs of the element, then
  // those of the neighbour)

  template <class IG, class LFSVC>
  void onBindLFSVOutside(const IG& ig, const LFSVC&, const LFSVC& lfsv_n_cache)
  {
    bind(lfsv_n_cache, dofs_n_);
    assemble_face_ = LOP::doSkeletonTwoSided || ids_.id(ig.inside()) < ids_.id(ig.outside());
    face_dofs_ = dofs_s_;
    face_dofs_.insert(face_dofs_.end(), dofs_n_.begin(), dofs_n_.end());
    if constexpr (pattern)
      if (assemble_face_)
        for (auto i : face_dofs_)
          for (auto j : face_dofs_) target_.add(i, j);
    if constexpr (residual) r_n_.assign(dofs_n_.size(), 0.0);
  }

  template <class IG, class LFSUC, class LFSVC>
  void onBindLFSUVOutside(const IG&, const LFSUC&, const LFSVC&, const LFSUC&, const LFSVC&)
  {
    if constexpr (jacobian) {
      const auto m_s = dofs_s_.size();
      const auto m_n = dofs_n_.size();
      face_ss_.assign(m_s, m_s, 0.0);
      face_sn_.assign(m_s, m_n, 0.0);
      face_ns_.assign(m_n, m_s, 0.0);
      face_nn_.assign(m_n, m_n, 0.0);
    }
  }

  template <class LFSUC>
  void loadCoefficientsLFSUOutside(const LFSUC& lfsu_n_cache)
  {
    load(lfsu_n_cache, x_n_);
  }

  template <class IG, class LFSUC, class LFSVC>
  void onUnbindLFSUVOutside(const IG&, const LFSUC&, const LFSVC&, const LFSUC&, const LFSVC&)
  {
    if constexpr (jacobian) {
      if (!assemble_face_) return;
      const auto m_s = dofs_s_.size();
      const auto m = face_dofs_.size();
      face_mat_.resize(m * m);
      for (std::size_t j = 0; j < m; ++j)
        for (std::size_t i = 0; i < m; ++i)
          face_mat_[j * m + i] = i < m_s ? (j < m_s ? face_ss_.getEntry(i, j) : face_sn_.getEntry(i, j - m_s)) : (j < m_s ? face_ns_.getEntry(i - m_s, j) : face_nn_.getEntry(i - m_s, j - m_s));
      target_.add_values(face_dofs_, face_dofs_, face_mat_);
    }
  }

  template <class IG, class LFSVC>
  void onUnbindLFSVOutside(const IG&, const LFSVC&, const LFSVC&)
  {
    if constexpr (residual) scatter(dofs_n_, r_n_);
  }

  // The terms of the local operator

  template <class EG, class LFSUC, class LFSVC>
  void assembleUVVolume(const EG& eg, const LFSUC& lfsu_cache, const LFSVC& lfsv_cache)
  {
    const auto& lfsu = lfsu_cache.localFunctionSpace();
    const auto& lfsv = lfsv_cache.localFunctionSpace();
    if constexpr (jacobian) Call<LOP::doAlphaVolume>::jacobian_volume(lop_, eg, lfsu, x_s_, lfsv, a_ss_view_);
    if constexpr (residual) Call<LOP::doAlphaVolume>::alpha_volume(lop_, eg, lfsu, x_s_, lfsv, r_s_view_);
  }

  template <class EG, class LFSVC>
  void assembleVVolume(const EG& eg, const LFSVC& lfsv_cache)
  {
    if constexpr (residual) Call<LOP::doLambdaVolume>::lambda_volume(lop_, eg, lfsv_cache.localFunctionSpace(), r_s_view_);
  }

  template <class IG, class LFSUC, class LFSVC>
  void assembleUVSkeleton(const IG& ig, const LFSUC& lfsu_s_cache, const LFSVC& lfsv_s_cache, const LFSUC& lfsu_n_cache, const LFSVC& lfsv_n_cache)
  {
    if (!assemble_face_) return;
    const auto& lfsu_s = lfsu_s_cache.localFunctionSpace();
    const auto& lfsv_s = lfsv_s_cache.localFunctionSpace();
    const auto& lfsu_n = lfsu_n_cache.localFunctionSpace();
    const auto& lfsv_n = lfsv_n_cache.localFunctionSpace();
    if constexpr (jacobian) Call<LOP::doAlphaSkeleton>::jacobian_skeleton(lop_, ig, lfsu_s, x_s_, lfsv_s, lfsu_n, x_n_, lfsv_n, face_ss_view_, face_sn_view_, face_ns_view_, face_nn_view_);
    if constexpr (residual) Call<LOP::doAlphaSkeleton>::alpha_skeleton(lop_, ig, lfsu_s, x_s_, lfsv_s, lfsu_n, x_n_, lfsv_n, r_s_view_, r_n_view_);
  }

  template <class IG, class LFSVC>
  void assembleVSkeleton(const IG& ig, const LFSVC& lfsv_s_cache, const LFSVC& lfsv_n_cache)
  {
    if (!assemble_face_) return;
    if constexpr (residual) Call<LOP::doLambdaSkeleton>::lambda_skeleton(lop_, ig, lfsv_s_cache.localFunctionSpace(), lfsv_n_cache.localFunctionSpace(), r_s_view_, r_n_view_);
  }

  template <class IG, class LFSUC, class LFSVC>
  void assembleUVBoundary(const IG& ig, const LFSUC& lfsu_cache, const LFSVC& lfsv_cache)
  {
    const auto& lfsu = lfsu_cache.localFunctionSpace();
    const auto& lfsv = lfsv_cache.localFunctionSpace();
    if constexpr (jacobian) Call<LOP::doAlphaBoundary>::jacobian_boundary(lop_, ig, lfsu, x_s_, lfsv, a_ss_view_);
    if constexpr (residual) Call<LOP::doAlphaBoundary>::alpha_boundary(lop_, ig, lfsu, x_s_, lfsv, r_s_view_);
  }

  template <class IG, class LFSVC>
  void assembleVBoundary(const IG& ig, const LFSVC& lfsv_cache)
  {
    if constexpr (residual) Call<LOP::doLambdaBoundary>::lambda_boundary(lop_, ig, lfsv_cache.localFunctionSpace(), r_s_view_);
  }

  template <class EG, class LFSUC, class LFSVC>
  void assembleUVVolumePostSkeleton(const EG& eg, const LFSUC& lfsu_cache, const LFSVC& lfsv_cache)
  {
    const auto& lfsu = lfsu_cache.localFunctionSpace();
    const auto& lfsv = lfsv_cache.localFunctionSpace();
    if constexpr (jacobian) Call<LOP::doAlphaVolumePostSkeleton>::jacobian_volume_post_skeleton(lop_, eg, lfsu, x_s_, lfsv, a_ss_view_);
    if constexpr (residual) Call<LOP::doAlphaVolumePostSkeleton>::alpha_volume_post_skeleton(lop_, eg, lfsu, x_s_, lfsv, r_s_view_);
  }

  template <class EG, class LFSVC>
  void assembleVVolumePostSkeleton(const EG& eg, const LFSVC& lfsv_cache)
  {
    if constexpr (residual) Call<LOP::doLambdaVolumePostSkeleton>::lambda_volume_post_skeleton(lop_, eg, lfsv_cache.localFunctionSpace(), r_s_view_);
  }

private:
  using LocalX = Dune::PDELab::LocalVector<DF, Dune::PDELab::TrialSpaceTag>;
  using LocalR = Dune::PDELab::LocalVector<RF, Dune::PDELab::TestSpaceTag>;
  using LocalA = Dune::PDELab::LocalMatrix<JF>;

  template <class Cache>
  static void bind(const Cache& cache, std::vector<Index>& dofs)
  {
    dofs.resize(cache.size());
    for (std::size_t i = 0; i < cache.size(); ++i) {
      DDM_ASSERT(cache.containerIndex(i).size() == 1, "assembly: the function space must not use blocking");
      dofs[i] = static_cast<Index>(cache.containerIndex(i)[0]);
    }
  }

  template <class Cache>
  void load(const Cache& cache, LocalX& x_local) const
  {
    x_local.resize(cache.size());
    if constexpr (!pattern)
      for (std::size_t i = 0; i < cache.size(); ++i) x_local.base()[i] = x_[cache.containerIndex(i)];
  }

  void scatter(const std::vector<Index>& dofs, const LocalR& r_local)
  {
    for (std::size_t i = 0; i < dofs.size(); ++i) target_[static_cast<std::size_t>(dofs[i])] += r_local.base()[i];
  }

  const LOP& lop_;
  const IdSet& ids_;
  const X& x_;
  Target& target_;

  std::vector<Index> dofs_s_; // the element
  std::vector<Index> dofs_n_; // its neighbour across a face
  std::vector<Index> face_dofs_;
  bool assemble_face_ = false;

  LocalX x_s_;
  LocalX x_n_;
  LocalR r_s_;
  LocalR r_n_;
  typename LocalR::WeightedAccumulationView r_s_view_{r_s_, 1.0};
  typename LocalR::WeightedAccumulationView r_n_view_{r_n_, 1.0};
  LocalA a_ss_;
  LocalA face_ss_;
  LocalA face_sn_;
  LocalA face_ns_;
  LocalA face_nn_;
  typename LocalA::WeightedAccumulationView a_ss_view_{a_ss_, 1.0};
  typename LocalA::WeightedAccumulationView face_ss_view_{face_ss_, 1.0};
  typename LocalA::WeightedAccumulationView face_sn_view_{face_sn_, 1.0};
  typename LocalA::WeightedAccumulationView face_ns_view_{face_ns_, 1.0};
  typename LocalA::WeightedAccumulationView face_nn_view_{face_nn_, 1.0};
  std::vector<JF> face_mat_;
};

// The constrained dofs of a constraints container of a GridOperator, none for EmptyTransformation
template <class C>
std::vector<Index> constrained_dofs(const C& cc)
{
  std::vector<Index> dofs;
  if constexpr (!std::is_same_v<C, Dune::PDELab::EmptyTransformation>)
    for (const auto& [ci, row] : cc) dofs.push_back(static_cast<Index>(ci[0]));
  return dofs;
}
} // namespace detail

/** Assembles the Jacobian of the PDELab GridOperator go at x into a ddm::Mat with the Communication comm (see
 *  make_communication()). Only the grid loop of go is used, PDELab assembles no matrix. Every element and every face is
 *  assembled on one rank (see detail::Engine), and the element and face matrices are recorded (for GenEO/MsGFEM)
 *  unless config["mat.record_values"] says otherwise.
 *
 *  All dofs constrained in go are treated as Dirichlet dofs: their rows are replaced by unit rows (see
 *  Mat::zero_rows()), the recorded matrices are not changed. Every rank holding a constrained dof must have it in the
 *  constraints. The Jacobian then belongs to the correction form of PDELab: with the residual r of
 *  assemble_residual() and A d = r, the solution is x - d, and d is zero at the Dirichlet dofs.
 *
 *  Collective.
 */
template <class GO>
std::shared_ptr<ddm::Mat<typename GO::Traits::JacobianField>> assemble_jacobian(const Dune::ParameterTree& config,
                                                                                const GO& go,
                                                                                const typename GO::Traits::Domain& x,
                                                                                std::shared_ptr<ddm::Communication> comm)
{
  using JF = typename GO::Traits::JacobianField;
  const auto& gfs = go.trialGridFunctionSpace();
  constexpr int dim = std::remove_cvref_t<decltype(gfs)>::Traits::GridView::dimension;

  // Every element couples with itself and, for skeleton terms, with its neighbours
  const auto n = static_cast<ddm::Index>(comm->roots().size());
  ddm::Pattern pattern(n, n, static_cast<ddm::Index>(gfs.maxLocalSize() * (2 * dim + 1)));
  {
    detail::Engine engine(go, x, pattern);
    go.assembler().assemble(engine);
  }
  pattern.finalize();

  auto mat_config = config.sub("mat");
  if (!mat_config.hasKey("record_values")) mat_config["record_values"] = "true";
  auto A = ddm::create_mat<JF>(mat_config, pattern, std::move(comm));
  {
    detail::Engine engine(go, x, *A);
    go.assembler().assemble(engine);
  }
  A->assemble();
  A->zero_rows(detail::constrained_dofs(go.localAssembler().trialConstraints()));
  return A;
}

/** Assembles the residual of the PDELab GridOperator go at x, as a consistent vector in the range of A (the matrix of
 *  assemble_jacobian()), zero at the constrained dofs. x must be consistent, including the ghost dofs of a DG space.
 *
 *  Collective.
 */
template <class GO>
ddm::Vec<typename GO::Traits::RangeField> assemble_residual(const GO& go, const typename GO::Traits::Domain& x, const ddm::Mat<typename GO::Traits::JacobianField>& A)
{
  static_assert(std::is_same_v<typename GO::Traits::RangeField, typename GO::Traits::JacobianField>, "the residual is a vector in the range of the Jacobian");

  auto r = A.create_range_vector();
  {
    auto view = r.host_view(ddm::write);
    std::fill(view.begin(), view.end(), 0);
    detail::Engine engine(go, x, view);
    go.assembler().assemble(engine);
    for (auto i : detail::constrained_dofs(go.localAssembler().testConstraints())) view[static_cast<std::size_t>(i)] = 0;
  }
  A.communication()->reduce(r);
  return r;
}

// Copies between a PDELab vector and a ddm::Vec of the same function space
template <class T, class X>
void copy(const ddm::Vec<T>& from, X& to)
{
  const auto view = from.host_view(ddm::read);
  std::copy(view.begin(), view.end(), to.begin());
}

template <class X, class T>
void copy(const X& from, ddm::Vec<T>& to)
{
  auto view = to.host_view(ddm::write);
  std::copy(from.begin(), from.end(), view.begin());
}
} // namespace ddm::examples
