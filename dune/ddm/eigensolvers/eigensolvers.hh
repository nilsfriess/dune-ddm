#pragma once

#include "../test/matrix_symmetry_helper.hh"
#include "dune/common/exceptions.hh"
#include "dune/ddm/eigensolvers/block_krylov_schur.hh"
#include "dune/ddm/eigensolvers/umfpack.hh"
#include "eigensolver_params.hh"
#include "shift_invert_eigenproblem.hh"
#include "spectra.hh"
#include "srrit.hh"
#include "subspace_iteration.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/solver.hh>
#include <type_traits>
#include <vector>

template <class Mat1, class Mat2>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> solve_gevp(std::shared_ptr<Mat1> A_, std::shared_ptr<Mat2> B_, const Dune::ParameterTree& ptree)
{
  static_assert(std::is_convertible_v<Mat1, Mat2> or std::is_convertible_v<Mat1, Mat2>, "The two matrix types must be compatible");
  using Mat = std::remove_cvref_t<Mat1>;

  auto A = std::make_shared<Mat>(*A_);
  auto B = std::make_shared<Mat>(*B_);

  matrix_symmetry_helper::ensure_matrix_symmetry(A, B);

  EigensolverParams params(ptree);

  constexpr std::size_t blocksize = 4;
  auto evp = std::make_shared<ShiftInvertEigenproblem<Mat, blocksize>>(*A, B, params.shift);

  if (params.type == EigensolverParams::Type::SubspaceIteration) return subspace_iteration(*evp, params);
  else if (params.type == EigensolverParams::Type::Spectra) return spectra_gevp(*A, *B, params);
  else if (params.type == EigensolverParams::Type::SRRIT) return srrit(evp, params);
  else if (params.type == EigensolverParams::Type::KrylovSchur) return block_krylov_schur(evp, params);
  else DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
}

template <class Mat1, class Mat2, class Callback>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> solve_gevp(std::shared_ptr<Mat1> A_, std::shared_ptr<Mat2> B_, Callback&& callback, const Dune::ParameterTree& ptree)
{
  static_assert(std::is_convertible_v<Mat1, Mat2> or std::is_convertible_v<Mat1, Mat2>, "The two matrix types must be compatible");
  using Mat = std::remove_cvref_t<Mat1>;

  auto A = std::make_shared<Mat>(*A_);
  auto B = std::make_shared<Mat>(*B_);

  matrix_symmetry_helper::ensure_matrix_symmetry(A, B);

  EigensolverParams params(ptree);

  constexpr std::size_t blocksize = 4;
  auto evp = std::make_shared<ShiftInvertEigenproblem<Mat, blocksize, Callback>>(*A, B, params.shift, std::forward<Callback>(callback));

  if (params.type == EigensolverParams::Type::SubspaceIteration) return subspace_iteration(*evp, params);
  else if (params.type == EigensolverParams::Type::Spectra) return spectra_gevp(*A, *B, params);
  else if (params.type == EigensolverParams::Type::SRRIT) return srrit(evp, params);
  else if (params.type == EigensolverParams::Type::KrylovSchur) return block_krylov_schur(evp, params);
  else DUNE_THROW(Dune::NotImplemented, "Eigensolver not implemented");
}
