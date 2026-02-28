#pragma once

#include "eigensolver_params.hh"
#include "spectra.hh"
#include "trl.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/solver.hh>
#include <vector>

template <class Scalar = double>
std::vector<Dune::BlockVector<Dune::FieldVector<Scalar, 1>>> solve_gevp(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A, const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>> B,
                                                                        const Dune::ParameterTree& ptree)
{
  auto* eigensolver_event = Logger::get().registerOrGetEvent("Eigensolver", "solve");
  Logger::ScopedLog sl(eigensolver_event);

  EigensolverParams params(ptree);

  if (params.type == EigensolverParams::Type::Spectra) return spectra_gevp(A, B, params);
  if (params.type == EigensolverParams::Type::trl) return trl_gevp(A, B, params);
  else DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
}

template <class Scalar = double>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>>
solve_gevp(const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>& A, const Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>> B,
           Dune::InverseOperator<Dune::BlockVector<Dune::FieldVector<double, 1>>, Dune::BlockVector<Dune::FieldVector<double, 1>>>* constraint_solver, const std::vector<bool>& subdomain_boundary_mask,
           const Dune::ParameterTree& ptree)
{
  auto* eigensolver_event = Logger::get().registerOrGetEvent("Eigensolver", "solve (constraint)");
  Logger::ScopedLog sl(eigensolver_event);

  EigensolverParams params(ptree);

  if (params.type == EigensolverParams::Type::trl) return trl_gevp(A, B, constraint_solver, subdomain_boundary_mask, params);
  else DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
}
