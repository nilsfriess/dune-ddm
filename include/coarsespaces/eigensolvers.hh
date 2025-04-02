#pragma once

#include <dune/common/parametertree.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <dune/istl/solver.hh>
#include <vector>

enum class Eigensolver { Spectra, BLOPEX };

// Helper to convert floats to string because std::to_string outputs with low precision
template <typename T>
std::string to_string_with_precision(const T a_value)
{
  std::ostringstream out;
  out << a_value;
  return std::move(out).str();
}

// template <class Prec>
// void applyPreconditionerFake(void *T_, void *r_, void *Tr_)
// {
//   auto *T = static_cast<Prec *>(T_);
//   auto *r = static_cast<mv_TempMultiVector *>(r_);
//   auto *Tr = static_cast<mv_TempMultiVector *>(Tr_);

//   for (std::size_t i = 0; i < r->numVectors; ++i) {
//     if (r->mask && r->mask[i] == 0) {
//       continue;
//     }

//     auto *v = static_cast<Vector *>(r->vector[i]);
//     auto *res = static_cast<Vector *>(Tr->vector[i]);

//     Dune::InverseOperatorResult ior;
//     T->apply(*res, *v, ior);
//   }
// }

// template <class Prec>
// void applyPreconditioner(void *T_, void *r_, void *Tr_)
// {
//   auto *T = static_cast<Prec *>(T_);
//   auto *r = static_cast<Multivec *>(r_);
//   auto *Tr = static_cast<Multivec *>(Tr_);

//   Dune::BlockVector<Dune::FieldVector<double, 1>> v(r->N);
//   Dune::BlockVector<Dune::FieldVector<double, 1>> d(r->N);

//   Dune::InverseOperatorResult ior;
//   for (std::size_t i = 0; i < r->active_indices.size(); ++i) {
//     auto *rstart = r->entries.data() + r->active_indices[i] * r->N;
//     auto *Trstart = Tr->entries.data() + Tr->active_indices[i] * Tr->N;

//     std::copy(rstart, rstart + r->N, d.begin());
//     v = 0;

//     Dune::InverseOperatorResult res;
//     T->apply(v, d, res);

//     std::copy(v.begin(), v.end(), Trstart);
//   }
// }

std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> solveGEVP(const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &A, const Dune::BCRSMatrix<Dune::FieldMatrix<double, 1>> &B,
                                                                       Eigensolver eigensolver, const Dune::ParameterTree &ptree);
