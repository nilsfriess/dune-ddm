#pragma once

#include "dune/common/exceptions.hh"
#include "eigensolver_params.hh"
#include "spectra.hh"
#include "subspace_iteration.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/solver.hh>
#include <vector>

template <class Mat>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> solve_gevp(const Mat& A, Mat& B, const Dune::ParameterTree& ptree)
{
  EigensolverParams params(ptree);

  if (params.type == EigensolverParams::Type::SubspaceIteration) return subspace_iteration(A, B, params);
  else if (params.type == EigensolverParams::Type::Spectra) return spectra_gevp(A, B, params);
  else DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
}
