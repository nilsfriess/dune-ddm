#pragma once

#include <algorithm>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/foreach.hh>
#include <dune/istl/io.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/umfpack.hh>

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

  auto geneo_type = ptree.get("geneo_type", "full");
  spdlog::info("Setting up GenEO coarse space in mode '{}'", geneo_type);
  Logger::get().startEvent(neumann_corr);

  // We begin by extending the Dirichlet mask to the overlapping subdomain
  Vec dirichlet_mask_ovlp(Aovlp.N());
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

  // We begin by assigning to some dofs their distance to the overlapping subdomain boundary. This information
  // is used in some places below.
  // TODO: Actually, I don't think this information is necessary and we can get away with less communication.
  //       Secondly, this code appears multiple times over the whole codebase, we should just refactor it into a function or class.
  int rank{};
  MPI_Comm_rank(ovlp_ids.first->communicator(), &rank);
  const auto &ovlp_paridxs = *ovlp_ids.second;
  IdentifyBoundaryDataHandle ibdh(Aovlp, ovlp_paridxs, rank);
  communicator.forward(ibdh);
  auto boundaryMask = ibdh.getBoundaryMask();

  std::vector<int> boundary_dst(ovlp_paridxs.size(), std::numeric_limits<int>::max() - 1);
  for (const auto &idxpair : ovlp_paridxs) {
    auto li = idxpair.local();
    if (boundaryMask[li]) {
      boundary_dst[li] = 0;
    }
  }

  // Here we take 2 * overlap plus some safety margin because we compute the distance from the overlapping subdomain boundary
  const auto overlap = ptree.get("overlap", 1);
  for (int round = 0; round <= 2 * overlap + 3; ++round) {
    for (int i = 0; i < boundary_dst.size(); ++i) {
      for (auto cIt = Aovlp[i].begin(); cIt != Aovlp[i].end(); ++cIt) {
        boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cIt.index()] + 1); // Increase distance from boundary by one
      }
    }
  }

  Mat A; // The left-hand side of the eigenproblem. Will be set below depending on the requested type
  Mat B; // The right-hand side of the eigenproblem. Will be set below depending on the requested type
  if (geneo_type == "full" or geneo_type == "overlap") {
    // For the full GenEO coarse space, we first need to assemble the Neumann matrix on the overlapping subdomain.
    // This can be done by copying the Dirichlet matrix and applying the Neumann correction that we created
    // earlier.
    A = Aovlp;

    for (const auto &triple : remote_ncorr_triples) {
      // The triples use global indices, so we first have to convert them to local indices
      // on the overlapping subdomain. Also, we might have received some indices that are
      // outside of our overlapping subdomain, so we first have to check that.
      if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
        auto lrow = ovlp_paridxs[triple.row].local();
        auto lcol = ovlp_paridxs[triple.col].local();

        A[lrow][lcol] -= triple.val;
      }
      else {
        spdlog::trace("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
      }
    }

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
    // The right-hand side is constructed from this matrix, its exact form depends on the requested GenEO type. In
    // "full" case, we're almost done, we just need to multipliy B with the partiton of unity from both sides. In the
    // "overlap" case (the classical GenEO coarse space), we need to eliminate all dofs that are not in the overlap.
    B = A;
    // This is the classical GenEO coarse space. Here, we have to eliminate all interior dofs of the B matrix.
    if (geneo_type == "overlap") {
      // Elminate all entries that correspond to dofs with a distance > 2*overlap from the boundary
      for (auto ri = B.begin(); ri != B.end(); ++ri) {
        // TODO: We can skip some rows early here (those that correspond to dofs in the overlap with no neighbours in the interior)
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if (boundary_dst.at(ri.index()) > 2 * overlap or boundary_dst.at(ci.index()) > 2 * overlap) {
            *ci = 0;
          }
        }
      }

      // Finally apply the corrections to get Neumann boundary conditions on the interior boundary
      for (const auto &triple : own_ncorr_triples) {
        if (boundary_dst.at(triple.row) == 2 * overlap and boundary_dst.at(triple.col) == 2 * overlap) {
          B[triple.row][triple.col] -= triple.val;
        }
        else {
          spdlog::get("all_ranks")->warn("Unexpected triple at distance ({}, {})", boundary_dst.at(triple.row), boundary_dst.at(triple.col));
        }
      }

      // Dirichlet conditions
      for (std::size_t i = 0; i < B.N(); ++i) {
        if (dirichlet_mask_ovlp[i] > 0) {
          for (auto ci = B[i].begin(); ci != B[i].end(); ++ci) {
            *ci = (ci.index() == i) ? 1.0 : 0.0;
          }
        }
        else {
          for (auto ci = B[i].begin(); ci != B[i].end(); ++ci) {
            if (dirichlet_mask_ovlp[ci.index()] > 0) {
              *ci = 0.0;
            }
          }
        }
      }
    }

    // In both cases, the right-hand side matrix has to multiplied by the partition of unity from the left and right.
    for (auto ri = B.begin(); ri != B.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        *ci *= pou[ri.index()] * pou[ci.index()];
      }
    }
  }
  else if (geneo_type == "ring") {
    // In this case, we solve the eigenvalue problem only in ther overlapping region (which we will refer to here as the ring).
    // The solution on the whole overlapping subdomain is then obtained by extending each eigenvector to the interior in a
    // certain way (see below, after the solution of the eigenproblem).

    // First we have to create the matrix on the ring. To this end, we first need to identify all degrees of freedom inside the ring.
    std::vector<std::size_t> ring_to_subdomain(boundary_dst.size());
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
      if (boundary_dst[i] <= 2 * overlap) {
        ring_to_subdomain[cnt++] = i;
      }
    }
    ring_to_subdomain.resize(cnt);

    // We create a ring-to-subdomain mapping by simply sorting this list of dofs
    std::sort(ring_to_subdomain.begin(), ring_to_subdomain.end());

    // We also create the inverse mapping (subdomain-to-ring)
    std::unordered_map<std::size_t, std::size_t> subdomain_to_ring;
    subdomain_to_ring.reserve(ring_to_subdomain.size());
    for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
      subdomain_to_ring[ring_to_subdomain[i]] = i;
    }

    // Now create the matrix
    auto avg = Aovlp.nonzeroes() / Aovlp.N() + 2;
    auto N = ring_to_subdomain.size();
    A.setBuildMode(Mat::implicit);
    A.setImplicitBuildModeParameters(avg, 0.2);
    A.setSize(N, N);

    for (auto ri = Aovlp.begin(); ri != Aovlp.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        if (subdomain_to_ring.contains(ri.index()) && subdomain_to_ring.contains(ci.index())) {
          A.entry(subdomain_to_ring[ri.index()], subdomain_to_ring[ci.index()]) = 0.0;
        }
      }
    }
    A.compress();

    for (auto ri = A.begin(); ri != A.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        *ci = Aovlp[ring_to_subdomain[ri.index()]][ring_to_subdomain[ci.index()]];
      }
    }

    // We again need to apply the Neumann corrections. First do the "outside" boundary ...
    for (const auto &triple : remote_ncorr_triples) {
      // The triples use global indices, so we first have to convert them to local indices
      // on the overlapping subdomain. Also, we might have received some indices that are
      // outside of our overlapping subdomain, so we first have to check that.
      if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
        auto lrow = ovlp_paridxs[triple.row].local();
        auto lcol = ovlp_paridxs[triple.col].local();

        // Check if the given entry corresponds to a dof inside the ring
        if (subdomain_to_ring.contains(lrow) && subdomain_to_ring.contains(lcol)) {
          A[subdomain_to_ring[lrow]][subdomain_to_ring[lcol]] -= triple.val;
        }
      }
      else {
        spdlog::trace("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
      }
    }
    // ... then the "inside" boundary
    for (const auto &triple : own_ncorr_triples) {
      if (boundary_dst.at(triple.row) == 2 * overlap and boundary_dst.at(triple.col) == 2 * overlap) {
        if (subdomain_to_ring.contains(triple.row) && subdomain_to_ring.contains(triple.col)) {
          A[subdomain_to_ring[triple.row]][subdomain_to_ring[triple.col]] -= triple.val;
        }
      }
      else {
        spdlog::get("all_ranks")->warn("Unexpected triple at distance ({}, {})", boundary_dst.at(triple.row), boundary_dst.at(triple.col));
      }
    }

    // Dirichlet conditions
    for (std::size_t i = 0; i < A.N(); ++i) {
      if (dirichlet_mask_ovlp[ring_to_subdomain[i]] > 0) {
        for (auto ci = A[i].begin(); ci != A[i].end(); ++ci) {
          *ci = (ci.index() == i) ? 1.0 : 0.0;
        }
      }
      else {
        for (auto ci = A[i].begin(); ci != A[i].end(); ++ci) {
          if (dirichlet_mask_ovlp[ring_to_subdomain[ci.index()]] > 0) {
            *ci = 0.0;
          }
        }
      }
    }

    B = A; // B starts as A, but has to be multiplied by the partition of unity

    for (auto ri = B.begin(); ri != B.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        *ci *= pou[ring_to_subdomain[ri.index()]] * pou[ring_to_subdomain[ci.index()]];
      }
    }
  }
  else {
    spdlog::error("Unknown GenEO type '{}', aborting");
    MPI_Abort(MPI_COMM_WORLD, 3);
  }

  Logger::get().endEvent(neumann_corr);

  //  Regularize
  // for (std::size_t i = 0; i < B.N(); ++i) {
  //   B[i][i] += 0.0001;
  // }

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
  std::vector<Vec> eigenvectors = solveGEVP<Vec>(A, B, eigensolver, ptree.get("nev", 10));

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

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Abort(MPI_COMM_WORLD, 0);

  Logger::get().endEvent(eigensolver_event);
  return eigenvectors;
}
