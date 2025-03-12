#pragma once

#include <Eigen/Eigen>
#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymGEigsShiftSolver.h>

#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/istl/foreach.hh>

#include <spdlog/spdlog.h>

#include <vector>

#include "datahandles.hh"
#include "helpers.hh"

template <class Vec, class Mat, class RemoteIndices>
std::vector<Vec> buildGenEOCoarseSpace(const RemoteIndices &ovlp_ids, const Mat &Aovlp, const std::vector<TripleWithRank> &remote_ncorr_triples, const Vec &dirichlet_mask_novlp, const Vec &pou,
                                       long nev = 10)
{
  // For the GenEO coarse space, we first need to assemble the Neumann matrix on the overlapping subdomain.
  // This can be done by copying the Dirichlet matrix and applying the Neumann correction that we created
  // earlier.
  auto A = Aovlp;
  
  int cnt = 0;
  const auto &ovlp_paridxs = *ovlp_ids.second;
  for (const auto &triple : remote_ncorr_triples) {
    // The triples use global indices, so we first have to convert them to local indices
    // on the overlapping subdomain. Also, we might have received some indices that are
    // outside of our overlapping subdomain, so we first have to check that.
    if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
      auto lrow = ovlp_paridxs[triple.row].local();
      auto lcol = ovlp_paridxs[triple.col].local();

      // assert(lrow >= paridxs.size() && lcol >= paridxs.size() && "Should only change entries outside of the original domain");

      A[lrow][lcol] -= triple.val;
      cnt++;
    }
    else {
      spdlog::trace("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
    }
  }

  Vec dirichlet_mask_ovlp(A.N());
  dirichlet_mask_ovlp = 0;
  for (std::size_t i = 0; i < dirichlet_mask_novlp.N(); ++i) {
    dirichlet_mask_ovlp[i] = dirichlet_mask_novlp[i];
  }

  const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  Dune::Interface interface;
  interface.build(*ovlp_ids.first, allAttributes, allAttributes);
  Dune::VariableSizeCommunicator communicator(interface);
  AddVectorDataHandle<Vec> advdh;
  advdh.setVec(dirichlet_mask_ovlp);
  communicator.forward(advdh);

  for (std::size_t i = 0; i < A.N(); ++i) {
    if (dirichlet_mask_ovlp[i] > 0) {
      for (auto ci = A[i].begin(); ci != A[i].end(); ++ci) {
        *ci = (ci.index() == i) ? 1.0 : 0.0;
      }
    }
    else {
      for (auto ci = A[i].begin(); ci != A[i].end(); ++ci) {
        if (dirichlet_mask_ovlp[ci.index()] > 0) {
          *ci = 0.0;
        }
      }
    }
  }

  // Now we have the Neumann matrix on the overlapping subdomain. This is the left-hand side of the eigenproblem.
  // The right-hand side is constructed from this matrix by multiplying it from both sides with the partition of unity.
  auto B = A;

  // // Now zero the matrix on rows and columns corresponding to interior dofs
  // Native<Vec> interior_marker(ovlp_paridxs.size());
  // interior_marker = 1;
  // const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  // Dune::Interface interface;
  // interface.build(*schwarz->getOverlappingIndices().first, allAttributes, allAttributes);
  // Dune::VariableSizeCommunicator communicator(interface);
  // AddVectorDataHandle<Native<Vec>> advdh;
  // advdh.setVec(interior_marker);
  // communicator.forward(advdh);

  // for (auto ri = B.begin(); ri != B.end(); ++ri) {
  //   for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
  //     if (interior_marker[ri.index()] == 1 or interior_marker[ci.index()] == 1) {
  //       *ci = 0.;
  //     }
  //   }
  // }

  // // And apply the Neumann corrections in the interior boundary
  // for (const auto &triple : own_ncorr_triples) {
  //   auto row = paridxs[triple.row].local();
  //   auto col = paridxs[triple.col].local();
  //   B[row][col] -= triple.val; // TODO: Store the triples using local indices

  //   if (interior_marker[row] == 1 or interior_marker[col] == 1) {
  //     std::cout << "THIS SHOULD NOT HAPPEN\n";
  //   }
  // }

  // Finally multiply it with the partition of unity from the left and right
  for (auto ri = B.begin(); ri != B.end(); ++ri) {
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      *ci *= pou[ri.index()] * pou[ci.index()];
    }
  }

  // Regularize
  // for (std::size_t i = 0; i < B.N(); ++i) {
  //   B[i][i] += 0.000001;
  // }

  std::vector<double> eigenvalues;
  std::vector<Vec> eigenvectors;

  using Triplet = Eigen::Triplet<double>;
  auto N = static_cast<Eigen::Index>(A.N());
  Eigen::SparseMatrix<double> A_(N, N);
  Eigen::SparseMatrix<double> B_(N, N);

  std::vector<Triplet> triplets;
  triplets.reserve(A.nonzeroes());
  Dune::flatMatrixForEach(A, [&](auto &&entry, std::size_t i, std::size_t j) { triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry}); });
  A_.setFromTriplets(triplets.begin(), triplets.end());

  triplets.clear();
  triplets.reserve(A.nonzeroes());
  Dune::flatMatrixForEach(B, [&](auto &&entry, std::size_t i, std::size_t j) { triplets.push_back({static_cast<int>(i), static_cast<int>(j), entry}); });
  B_.setFromTriplets(triplets.begin(), triplets.end());

  A_.makeCompressed();
  B_.makeCompressed();

  using OpType = Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
  using BOpType = Spectra::SparseSymMatProd<double>;
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

  OpType op(A_, B_);
  BOpType Bop(B_);

  Spectra::SymGEigsShiftSolver<OpType, BOpType, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, nev, 2 * nev, -0.001);
  geigs.init();

  // Find largest eigenvalue of the shifted problem (which corresponds to the smallest of the original problem)
  // with max. 1000 iterations and sort the resulting eigenvalues from small to large.
  auto nconv = geigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-8, Spectra::SortRule::SmallestAlge);

  if (geigs.info() == Spectra::CompInfo::Successful) {
    const auto evalues = geigs.eigenvalues();
    const auto evecs = geigs.eigenvectors();

    eigenvalues.resize(nconv);
    std::copy(evalues.begin(), evalues.end(), eigenvalues.begin());

    Vec vec(evecs.rows());
    eigenvectors.resize(nconv);
    std::fill(eigenvectors.begin(), eigenvectors.end(), vec);

    for (int i = 0; i < nconv; ++i) {
      for (int j = 0; j < evecs.rows(); ++j) {
        eigenvectors[i][j] = evecs(j, i);
      }
    }
  }
  else {
    std::cerr << "ERROR: Eigensolver did not converge";
  }

  for (auto &vec : eigenvectors) {
    for (std::size_t i = 0; i < vec.N(); ++i) {
      if (dirichlet_mask_ovlp[i] > 0) {
        vec[i] = 0;
      }
      else {
        vec[i] *= pou[i];
      }
    }
  }
  return eigenvectors;
}
