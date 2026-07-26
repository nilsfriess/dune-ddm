#pragma once

#include "eigensolver_params.hh"
#include "eigensolver_result.hh"
#include "spectra.hh"
#include "trl.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/ddm/logger.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/solver.hh>
#include <string>
#include <vector>

namespace ddm {

template <class Scalar = double>
[[nodiscard]] GevpSolution<Scalar> solve_gevp(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A, const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>> B, const Dune::ParameterTree& ptree)
{
  auto* eigensolver_event = Logger::get().registerOrGetEvent("Eigensolver", "solve");
  Logger::ScopedLog sl(eigensolver_event);

  EigensolverParams params(ptree);

  auto A_view = Dune::stackobject_to_shared_ptr(A);
  auto B_view = Dune::stackobject_to_shared_ptr(B);

  if (params.type == EigensolverParams::Type::Spectra) {
#ifdef DUNE_DDM_HAVE_SPECTRA
    SpectraEigensolver<Scalar> eigensolver(A_view, B_view, params, ptree);
    auto info = eigensolver.solve();
    return {.eigenvectors = eigensolver.eigenvectors(), .eigenvalues = eigensolver.eigenvalues(), .info = info};
#else
    DUNE_THROW(Dune::NotImplemented, "Spectra eigensolver requested but DUNE_DDM_HAVE_SPECTRA is not defined");
#endif
  }
  if (params.type == EigensolverParams::Type::trl) return trl_gevp(A, B, params);
  else DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
}

template <class Scalar = double>
[[nodiscard]] GevpSolution<Scalar> solve_gevp(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A, const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>> B,
                                              Dune::InverseOperator<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>, Dune::BlockVector<Dune::FieldVector<Scalar, 1>>>* constraint_solver,
                                              const std::vector<bool>& subdomain_boundary_mask, const Dune::ParameterTree& ptree)
{
  auto* eigensolver_event = Logger::get().registerOrGetEvent("Eigensolver", "solve (constraint)");
  Logger::ScopedLog sl(eigensolver_event);

  EigensolverParams params(ptree);

  if (params.type == EigensolverParams::Type::trl) return trl_gevp(A, B, constraint_solver, subdomain_boundary_mask, params);
  else DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
}

} // namespace ddm
