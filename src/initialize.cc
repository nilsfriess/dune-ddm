#include "ddm/eigensolver/eigensolver.hh"
#include "ddm/impl/eigensolver/spectra.hh"
#include "ddm/impl/mat/istlmat.hh"
#include "ddm/impl/prec/ilu_istl.hh"
#include "ddm/impl/prec/jacobiprec.hh"
#include "ddm/impl/prec/noneprec.hh"
#include "ddm/impl/prec/schwarz.hh"
#include "ddm/impl/prec/strumpack.hh"
#include "ddm/impl/prec/two_level_schwarz.hh"
#include "ddm/impl/prec/umfpack.hh"
#include "ddm/impl/solver/direct.hh"
#include "ddm/impl/solver/gmres.hh"
#include "ddm/impl/vec/istlvec.hh"
#include "ddm/logger.hh"
#include "ddm/registry.hh"

#include <complex>
#include <dune/common/parametertree.hh>
#include <memory>
#include <mutex>
#include <type_traits>

namespace ddm {
namespace {
template <class T>
void register_all()
{
  // Register vectors
  register_vec<T>("istl", [](const Dune::ParameterTree&, Index n) { return std::make_unique<IstlVec<T>>(n); });

  // Register matrices
  register_local_mat<T>("istl", [](const Dune::ParameterTree& config, const Pattern& pattern) { return std::make_shared<IstlMat<T>>(config, pattern); });

  // Register preconditioners
  register_prec<T>("none", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<NonePrec<T>>(config, std::move(A)); });
  register_prec<T>("jacobi", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<JacobiPrec<T>>(config, std::move(A)); });
  register_prec<T>("ilu", BackendId::istl, [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<IstlILUPrec<T>>(config, std::move(A)); });
  register_prec<T>("umfpack", BackendId::istl, [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<UMFPACKPrec<T>>(config, std::move(A)); });
#ifdef DUNE_DDM_HAVE_STRUMPACK
  register_prec<T>("strumpack", BackendId::istl, [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<StrumpackPrec<T>>(config, std::move(A)); });
#endif
  // The LU decomposition of each backend, the default for exact solves of subproblems (see create_subproblem_solver()).
  // UMFPACK has no float, so float uses STRUMPACK if it is available
  register_prec<T>("lu", BackendId::istl, [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) -> std::shared_ptr<Prec<T>> {
#ifdef DUNE_DDM_HAVE_STRUMPACK
    if constexpr (std::is_same_v<T, float>) return std::make_shared<StrumpackPrec<T>>(config, std::move(A));
#endif
    return std::make_shared<UMFPACKPrec<T>>(config, std::move(A));
  });
  register_prec<T>("schwarz", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<SchwarzPrec<T>>(config, std::move(A)); });
  register_prec<T>("tlschwarz", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<TwoLevelSchwarzPrec<T>>(config, std::move(A)); });

  // Register solvers
  register_solver<T>(
      "gmres", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, std::shared_ptr<Prec<T>> P) { return std::make_shared<GMResSolver<T>>(config, std::move(A), std::move(P)); });
  register_solver<T>(
      "direct", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, std::shared_ptr<Prec<T>> P) { return std::make_shared<DirectSolver<T>>(config, std::move(A), std::move(P)); });

  // Register eigensolvers
#ifdef DUNE_DDM_HAVE_SPECTRA
  register_eigensolver<T>("spectra", [](const Dune::ParameterTree& config, std::shared_ptr<const LocalMat<T>> A, std::shared_ptr<const LocalMat<T>> B) {
    return std::make_shared<SpectraEigensolver<T>>(config, std::move(A), std::move(B));
  });
#endif
}

// The logger report needs the same events on all ranks, in the same order. Objects that only exist on some ranks (e.g.
// the coarse matrix and its solver on rank 0) would register their events only there, so all events of the library
// are registered here, on every rank. The objects then only look them up.
void register_events()
{
  auto& logger = Logger::get();
  for (const auto* event : {"apply"}) logger.registerOrGetEvent("Mat", event);
  for (const auto* event : {"solve"}) logger.registerOrGetEvent("GMRES", event);
  for (const auto* event : {"factorize", "solve"}) logger.registerOrGetEvent("UMFPACK", event);
#ifdef DUNE_DDM_HAVE_STRUMPACK
  for (const auto* event : {"factorize", "solve"}) logger.registerOrGetEvent("STRUMPACK", event);
#endif
  for (const auto* event : {"apply"}) logger.registerOrGetEvent("ILU", event);
  for (const auto* event : {"coarse space", "apply"}) logger.registerOrGetEvent("Two-level Schwarz", event);
  for (const auto* event : {"setup", "apply", "coarse solve"}) logger.registerOrGetEvent("Coarse correction", event);
  for (const auto* event : {"exchange", "matrix"}) logger.registerOrGetEvent("Neumann", event);
  for (const auto* event : {"solve"}) logger.registerOrGetEvent("Eigensolver", event);
  for (const auto* event : {"project"}) logger.registerOrGetEvent("Harmonic space", event);
}
} // namespace

void initialize()
{
  static std::once_flag flag;
  std::call_once(flag, [] {
    register_events();
    register_all<float>();
    register_all<double>();
    register_all<std::complex<double>>();
  });
}
} // namespace ddm
