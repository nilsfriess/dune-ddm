#include "ddm/impl/mat/istlmat.hh"
#include "ddm/impl/prec/ilu_istl.hh"
#include "ddm/impl/prec/jacobiprec.hh"
#include "ddm/impl/prec/noneprec.hh"
#include "ddm/impl/prec/schwarz.hh"
#include "ddm/impl/prec/two_level_schwarz.hh"
#include "ddm/impl/prec/umfpack.hh"
#include "ddm/impl/solver/direct.hh"
#include "ddm/impl/solver/gmres.hh"
#include "ddm/impl/vec/istlvec.hh"
#include "ddm/registry.hh"

#include <dune/common/parametertree.hh>
#include <memory>
#include <mutex>

namespace ddm {
namespace {
template <class T>
void register_all()
{
  // Register vectors
  register_vec<T>("istl", [](const Dune::ParameterTree&, Index n) { return std::make_unique<IstlVec<T>>(n); });

  // Register matrices
  register_local_mat<T>("istl", [](const Dune::ParameterTree&, const Pattern& pattern) { return std::make_shared<IstlMat<T>>(pattern); });

  // Register preconditioners
  register_prec<T>("none", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<NonePrec<T>>(config, std::move(A)); });
  register_prec<T>("jacobi", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<JacobiPrec<T>>(config, std::move(A)); });
  register_prec<T>("ilu", BackendId::istl, [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<IstlILUPrec<T>>(config, std::move(A)); });
  register_prec<T>("umfpack", BackendId::istl, [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<UMFPACKPrec<T>>(config, std::move(A)); });
  // The LU decomposition of each backend, the default for exact solves of subproblems (see create_subproblem_solver())
  register_prec<T>("lu", BackendId::istl, [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<UMFPACKPrec<T>>(config, std::move(A)); });
  register_prec<T>("schwarz", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<SchwarzPrec<T>>(config, std::move(A)); });
  register_prec<T>("tlschwarz", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A) { return std::make_shared<TwoLevelSchwarzPrec<T>>(config, std::move(A)); });

  // Register solvers
  register_solver<T>(
      "gmres", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, std::shared_ptr<Prec<T>> P) { return std::make_shared<GMResSolver<T>>(config, std::move(A), std::move(P)); });
  register_solver<T>(
      "direct", [](const Dune::ParameterTree& config, std::shared_ptr<const Mat<T>> A, std::shared_ptr<Prec<T>> P) { return std::make_shared<DirectSolver<T>>(config, std::move(A), std::move(P)); });
}
} // namespace

void initialize()
{
  static std::once_flag flag;
  std::call_once(flag, [] {
    register_all<float>();
    register_all<double>();
  });
}
} // namespace ddm
