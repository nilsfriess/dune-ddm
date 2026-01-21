#pragma once

#include "dune/common/exceptions.hh"
#include "eigensolver_params.hh"
#include "spectra.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/solver.hh>
#include <type_traits>
#include <vector>

template <class Mat1, class Mat2>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> solve_gevp(std::shared_ptr<Mat1> A, std::shared_ptr<Mat2> B, const Dune::ParameterTree& ptree)
{
  static_assert(std::is_convertible_v<Mat1, Mat2> or std::is_convertible_v<Mat1, Mat2>, "The two matrix types must be compatible");
  using Mat = std::remove_cvref_t<Mat1>;

  EigensolverParams params(ptree);

  if (params.type == EigensolverParams::Type::Spectra) return spectra_gevp(*A, *B, params);
  else DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
}

template <class Mat1, class Mat2, class Callback>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> solve_gevp(std::shared_ptr<Mat1> A, std::shared_ptr<Mat2> B, Callback&& callback, const Dune::ParameterTree& ptree)
{
  (void)callback;

  static_assert(std::is_convertible_v<Mat1, Mat2> or std::is_convertible_v<Mat1, Mat2>, "The two matrix types must be compatible");
  using Mat = std::remove_cvref_t<Mat1>;

  EigensolverParams params(ptree);

  if (params.type == EigensolverParams::Type::Spectra) return spectra_gevp(*A, *B, params);
  else DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
}
