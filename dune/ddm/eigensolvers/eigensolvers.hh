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

  if (params.type == EigensolverParams::Type::SubspaceIteration) { //
    return subspace_iteration(A, B, params);
  }
  else if (params.type == EigensolverParams::Type::Spectra) {
    return spectra_gevp(A, B, params);
  }
  // else if (params.type == EigensolverParams::Type::RAES) {
  //   std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> evecs(params.nev);
  //   std::vector<double> evals(params.nev);
  //   GeneralizedInverse(A, B, params.shift, 0.0, params.tolerance, params.maxit, params.nev, evals, evecs, 1);
  //   std::cout << "Computed eigenvalues using Rayleigh quotients:\n";
  //   for (std::size_t i = 0; i < params.nev; ++i) std::cout << evals[i] << "  ";
  //   std::cout << "\n";
  //   return evecs;
  // }
  else {
    DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
  }
}

// std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>>
// solveGEVP(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A, const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &B, Eigensolver eigensolver, const Dune::ParameterTree &ptree,
//           Dune::Preconditioner<Dune::BlockVector<Dune::FieldVector<double, 1>>, Dune::BlockVector<Dune::FieldVector<double, 1>>> *prec = nullptr);
