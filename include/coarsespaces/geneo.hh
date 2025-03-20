#pragma once

#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/foreach.hh>
#include <dune/istl/io.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/umfpack.hh>

#include <memory>
#include <spdlog/spdlog.h>

#include <vector>

#include "datahandles.hh"
#include "helpers.hh"
#include "logger.hh"
#include "spectral_coarsespace.hh"

template <class Vec, class Mat, class RemoteIndices>
std::vector<Vec> buildGenEOCoarseSpace(const RemoteIndices &ovlp_ids, const Mat &Aovlp, const std::vector<TripleWithRank> &remote_ncorr_triples,
                                       [[maybe_unused]] const std::vector<TripleWithRank> &own_ncorr_triples, const Vec &dirichlet_mask_novlp, const Vec &pou, const Dune::ParameterTree &ptree)
{
  auto *eigensolver_event = Logger::get().registerEvent("GenEO", "solve eigenproblem");
  auto *neumann_corr = Logger::get().registerEvent("GenEO", "setup Neumann");

  spdlog::info("Setting up GenEO coarse space");

  // For the GenEO coarse space, we first need to assemble the Neumann matrix on the overlapping subdomain.
  // This can be done by copying the Dirichlet matrix and applying the Neumann correction that we created
  // earlier.
  Logger::get().startEvent(neumann_corr);
  auto A = Aovlp;

  const auto &ovlp_paridxs = *ovlp_ids.second;
  for (const auto &triple : remote_ncorr_triples) {
    // The triples use global indices, so we first have to convert them to local indices
    // on the overlapping subdomain. Also, we might have received some indices that are
    // outside of our overlapping subdomain, so we first have to check that.
    if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
      auto lrow = ovlp_paridxs[triple.row].local();
      auto lcol = ovlp_paridxs[triple.col].local();

      // assert(lrow >= ovlp_paridxs.size() && lcol >= ovlp_paridxs.size() && "Should only change entries outside of the original domain");

      A[lrow][lcol] -= triple.val;
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

  // Prepare the preconditioner for the eigenproblem
  using Prec = Dune::SeqILDL<Mat, Vec, Vec>;
  std::unique_ptr<Prec> prec{nullptr};
  if (ptree.get("prec_eigensolver", false)) {
    // for (std::size_t i = 0; i < A.N(); ++i) {
    //   A[i][i] += 0.01;
    // }
    prec = std::make_unique<Prec>(A, 1.0);
    // for (std::size_t i = 0; i < A.N(); ++i) {
    //   A[i][i] -= 0.01;
    // }
  }

  // Now we have the Neumann matrix on the overlapping subdomain. This is the left-hand side of the eigenproblem.
  // The right-hand side is constructed from this matrix by multiplying it from both sides with the partition of unity.
  auto B = A;

  // if (own_ncorr_triples.size() > 0) {
  //   // Here, we zero the matrix first and then copy the values that we need back.
  //   B = 0;
  //   Vec interior_marker(ovlp_paridxs.size());
  //   interior_marker = 1;
  //   advdh.setVec(interior_marker);
  //   communicator.forward(advdh);

  //   for (auto ri = B.begin(); ri != B.end(); ++ri) {
  //     for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
  //       if (interior_marker[ci.index()] > 1) {
  //         *ci = A[ri.index()][ci.index()];
  //       }
  //     }
  //   }

  //   // And apply the Neumann corrections in the interior boundary
  //   for (const auto &triple : own_ncorr_triples) {
  //     auto row = ovlp_paridxs[triple.row].local();
  //     auto col = ovlp_paridxs[triple.col].local();
  //     B[row][col] -= triple.val; // TODO: Store the triples using local indices

  //     if (interior_marker[row] > 1) {
  //       std::cout << "Subtracting from overlap region\n";
  //     }
  //   }

  //   for (std::size_t i = 0; i < B.N(); ++i) {
  //     if (dirichlet_mask_ovlp[i] > 0) {
  //       for (auto ci = B[i].begin(); ci != B[i].end(); ++ci) {
  //         *ci = (ci.index() == i) ? 1.0 : 0.0;
  //       }
  //     }
  //     else {
  //       for (auto ci = B[i].begin(); ci != B[i].end(); ++ci) {
  //         if (dirichlet_mask_ovlp[ci.index()] > 0) {
  //           *ci = 0.0;
  //         }
  //       }
  //     }
  //   }
  // }

  // finally multiply it with the partition of unity from the left and right
  for (auto ri = B.begin(); ri != B.end(); ++ri) {
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      *ci *= pou[ri.index()] * pou[ci.index()];
    }
  }
  Logger::get().endEvent(neumann_corr);

  //  Regularize
  for (std::size_t i = 0; i < B.N(); ++i) {
    B[i][i] += 0.0001;
  }

  int rank;
  MPI_Comm_rank(ovlp_ids.first->communicator(), &rank);
  if (rank == 0) {
    Dune::writeMatrixToMatlab(A, "A.mat");
    Dune::writeMatrixToMatlab(B, "B.mat");
  }

  Eigensolver eigensolver = Eigensolver::Spectra;
  auto eigensolver_string = ptree.get("eigensolver", "spectra");
  if (eigensolver_string == "spectra") {
    eigensolver = Eigensolver::Spectra;
  }
  else if (eigensolver_string == "blopex") {
    eigensolver = Eigensolver::BLOPEX;
  }
  else {
    spdlog::warn("Unknow eigensolver type '{}' using Spectra instead", eigensolver_string);
  }

  Logger::get().startEvent(eigensolver_event);
  spdlog::info("Solving generalized eigenvalue problem for GenEO");
  std::vector<Vec> eigenvectors = solveGEVP<Vec>(A, B, eigensolver, ptree.get("nev", 10), prec.get());

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
  Logger::get().endEvent(eigensolver_event);
  return eigenvectors;
}
