#pragma once

#include <algorithm>
#include <cstddef>
#include <dune/istl/solvers.hh>
#include <vector>

#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/foreach.hh>
#include <dune/istl/io.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/umfpack.hh>

#include <spdlog/spdlog.h>

#include "datahandles.hh"
#include "helpers.hh"
#include "logger.hh"
#include "spectral_coarsespace.hh"

template <class Vec, class Mat, class RemoteIndices>
std::vector<Vec> buildGenEOCoarseSpace(const RemoteIndices &ovlp_ids, const Mat &Aovlp, const std::vector<TripleWithRank> &remote_ncorr_triples, const std::vector<TripleWithRank> &own_ncorr_triples,
                                       const std::vector<bool> &interior_dof_mask, const Vec &dirichlet_mask_novlp, const Vec &pou, const Dune::ParameterTree &ptree)
{
  MPI_Barrier(MPI_COMM_WORLD);
  auto *eigensolver_event = Logger::get().registerEvent("GenEO", "solve eigenproblem");
  auto *neumann_corr = Logger::get().registerEvent("GenEO", "create matrix");

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
  // is used, e.g., to identify the overlap region below.
  // TODO: Actually, I don't think this information is necessary and we can get away with less communication.
  //       Secondly, this code appears multiple times over the whole codebase, we should just refactor it into a function or class.
  int rank{};
  MPI_Comm_rank(ovlp_ids.first->communicator(), &rank);
  const auto &ovlp_paridxs = *ovlp_ids.second;
  // IdentifyBoundaryDataHandle ibdh(Aovlp, ovlp_paridxs, rank);
  // communicator.forward(ibdh);
  // auto boundaryMask = ibdh.getBoundaryMask();

  // std::vector<int> boundary_dst(ovlp_paridxs.size(), std::numeric_limits<int>::max() - 1);
  // for (const auto &idxpair : ovlp_paridxs) {
  //   auto li = idxpair.local();
  //   if (boundaryMask[li]) {
  //     boundary_dst[li] = 0;
  //   }
  // }

  // // Here we take 2 * overlap plus some safety margin because we compute the distance from the overlapping subdomain boundary
  // const auto overlap = ptree.get("overlap", 1);
  // for (int round = 0; round <= 2 * (overlap + 10); ++round) {
  //   for (int i = 0; i < boundary_dst.size(); ++i) {
  //     for (auto cIt = Aovlp[i].begin(); cIt != Aovlp[i].end(); ++cIt) {
  //       boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cIt.index()] + 1); // Increase distance from boundary by one
  //     }
  //   }
  // }

  std::unordered_map<std::size_t, std::size_t> subdomain_to_ring;
  std::vector<std::size_t> ring_to_subdomain(ovlp_paridxs.size());

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

    // Make sure global Dirichlet conditions are correctly set. We have to eliminate symmetrically, because the eigensolver expects a symmetric problem.
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
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if ((ri.index() < interior_dof_mask.size() and interior_dof_mask[ri.index()]) or (ci.index() < interior_dof_mask.size() and interior_dof_mask[ci.index()])) {
            *ci = 0;
          }
        }
      }

      // Finally apply the corrections to get Neumann boundary conditions on the interior boundary
      for (const auto &triple : own_ncorr_triples) {
        B[triple.row][triple.col] -= triple.val;
      }

      // Make sure global Dirichlet conditions are correctly set. Again, we have to eliminate symmetrically, because the eigensolver expects a symmetric problem.
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

#ifndef NDEBUG
      if (rank == ptree.get("debug_rank", 0)) {
        Dune::writeMatrixToMatlab(A, "A.mat");
        Dune::writeMatrixToMatlab(B, "B.mat");
      }
#endif
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
    // These are simply all dofs not in the interior.
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < Aovlp.N(); ++i) {
      if ((i < interior_dof_mask.size() and not interior_dof_mask[i]) or i >= interior_dof_mask.size()) {
        ring_to_subdomain[cnt++] = i;
      }
    }
    ring_to_subdomain.resize(cnt);

    // We also create the inverse mapping (subdomain-to-ring)
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
      if (subdomain_to_ring.contains(triple.row) && subdomain_to_ring.contains(triple.col)) {
        A[subdomain_to_ring[triple.row]][subdomain_to_ring[triple.col]] -= triple.val;
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

  // Regularize
  const auto regularisation = ptree.get("eigensolver_regularisation", 0.0);
  if (regularisation > 0) {
    for (std::size_t i = 0; i < B.N(); ++i) {
      B[i][i] += regularisation;
    }
  }

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

  std::vector<Vec> eigenvectors = solveGEVP<Vec>(A, B, eigensolver, ptree.get("nev", 10), ptree);
  Logger::get().endEvent(eigensolver_event);

  // If we have only solved the eigenvalue problem on the ring, then we now need to extend the eigenvectors to the interior.
  if (geneo_type == "ring") {
    // To extend the solution to the interior, we create a matrix that lives on the interior plus one layer of the ring.
    // This layer will act as a Dirichlet boundary and as the Dirichlet values, we set the values of the computed eigenvectors.

    std::vector<std::uint8_t> interior_ovlp_mask(Aovlp.N(), 0);
    // First we add all "true" interior dofs
    for (std::size_t i = 0; i < interior_dof_mask.size(); ++i) {
      interior_ovlp_mask[i] = interior_dof_mask[i] ? 1 : 0;
    }

    // Next, we add the degrees of freedom that are connected to an interior dof
    for (auto subdomain_idx : ring_to_subdomain) {
      for (auto ci = Aovlp[subdomain_idx].begin(); ci != Aovlp[subdomain_idx].end(); ++ci) {
        if (interior_ovlp_mask[ci.index()] == 1 and interior_ovlp_mask[subdomain_idx] == 0) {
          interior_ovlp_mask[subdomain_idx] = 2; // DOFs at the border between overlap and interior are assigned the value 2
        }
      }
    }

    auto interior_dofs = std::count_if(interior_ovlp_mask.begin(), interior_ovlp_mask.end(), [](auto &&val) { return val != 0; });
    spdlog::get("all_ranks")
        ->info("Total dofs {}, interior dofs {}, ring dofs {}, ring+interior dofs {}", Aovlp.N(), interior_dofs, ring_to_subdomain.size(), (interior_dofs + ring_to_subdomain.size()));
    std::vector<std::size_t> interior_to_subdomain(interior_dofs);
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < interior_ovlp_mask.size(); ++i) {
      if (interior_ovlp_mask[i]) {
        interior_to_subdomain[cnt++] = i;
      }
    }

    // We also create the inverse mapping (subdomain-to-interior)
    std::unordered_map<std::size_t, std::size_t> subdomain_to_interior;
    subdomain_to_interior.reserve(interior_to_subdomain.size());
    for (std::size_t i = 0; i < interior_to_subdomain.size(); ++i) {
      subdomain_to_interior[interior_to_subdomain[i]] = i;
    }

    // Now create the matrix
    Mat Aint;
    auto avg = Aovlp.nonzeroes() / Aovlp.N() + 2;
    auto N = interior_to_subdomain.size();
    Aint.setBuildMode(Mat::implicit);
    Aint.setImplicitBuildModeParameters(avg, 0.2);
    Aint.setSize(N, N);

    for (auto ri = Aovlp.begin(); ri != Aovlp.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        if (subdomain_to_interior.contains(ri.index()) && subdomain_to_interior.contains(ci.index())) {
          Aint.entry(subdomain_to_interior[ri.index()], subdomain_to_interior[ci.index()]) = 0.0;
        }
      }
    }
    Aint.compress();

    for (auto ri = Aint.begin(); ri != Aint.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        *ci = Aovlp[interior_to_subdomain[ri.index()]][interior_to_subdomain[ci.index()]];
      }
    }

    // Set Dirichlet rows
    for (std::size_t i = 0; i < Aint.N(); ++i) {
      if (dirichlet_mask_ovlp[interior_to_subdomain[i]] > 0) {
        for (auto ci = Aint[i].begin(); ci != Aint[i].end(); ++ci) {
          *ci = (ci.index() == i) ? 1.0 : 0.0;
        }
      }
    }

    for (std::size_t i = 0; i < Aint.N(); ++i) {
      if (interior_ovlp_mask[interior_to_subdomain[i]] > 1) {
        for (auto ci = Aint[i].begin(); ci != Aint[i].end(); ++ci) {
          *ci = (ci.index() == i) ? 1.0 : 0.0;
        }
      }
    }

    Vec x_int(Aint.N());
    x_int = 0.0;
    std::vector<Vec> solutions(eigenvectors.size(), x_int);

    auto *factorise_interior = Logger::get().registerEvent("GenEO", "factorise interior");
    auto *solve_interior = Logger::get().registerEvent("GenEO", "solve interior");
    if (not ptree.get("geneo_ring_approximate_interior", false)) {
      Logger::get().startEvent(factorise_interior);
      Dune::UMFPack interior_solver(Aint);
      Logger::get().endEvent(factorise_interior);

      Vec b_int(Aint.N());

      Logger::get().startEvent(solve_interior);
      for (std::size_t k = 0; k < eigenvectors.size(); ++k) {
        b_int = 0;
        for (std::size_t i = 0; i < eigenvectors[k].N(); ++i) {
          if (interior_ovlp_mask[ring_to_subdomain[i]] > 1) {
            assert(subdomain_to_interior.contains(ring_to_subdomain[i]) && "Interior and ring must overlap");
            b_int[subdomain_to_interior[ring_to_subdomain[i]]] = eigenvectors[k][i];
          }
        }

        Dune::InverseOperatorResult res;
        interior_solver.apply(solutions[k], b_int, res);
      }
      Logger::get().endEvent(solve_interior);
    }
    else {
      Logger::get().startEvent(factorise_interior);
      Dune::SeqILU<Mat, Vec, Vec> ilu_solver(Aint, 1.0);
      Logger::get().endEvent(factorise_interior);

      Vec x_int(Aint.N());
      x_int = 0.0;
      std::vector<Vec> solutions(eigenvectors.size(), x_int);
      Vec b_int(Aint.N());

      Dune::MatrixAdapter<Mat, Vec, Vec> adapter(Aint);
      Dune::RestartedGMResSolver<Vec> solver(adapter, ilu_solver, 1e-8, 50, 10, 0);

      Logger::get().startEvent(solve_interior);
      for (std::size_t k = 0; k < eigenvectors.size(); ++k) {
        b_int = 0;
        for (std::size_t i = 0; i < eigenvectors[k].N(); ++i) {
          if (interior_ovlp_mask[ring_to_subdomain[i]] > 1) {
            assert(subdomain_to_interior.contains(ring_to_subdomain[i]) && "Interior and ring must overlap");
            b_int[subdomain_to_interior[ring_to_subdomain[i]]] = eigenvectors[k][i];
          }
        }

        Dune::InverseOperatorResult res;
        solver.apply(solutions[k], b_int, res);
      }
      Logger::get().endEvent(solve_interior);
    }

    Vec zero(Aovlp.N());
    std::vector<Vec> combined_vectors(eigenvectors.size(), zero);
    for (std::size_t k = 0; k < eigenvectors.size(); ++k) {
      for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
        combined_vectors[k][ring_to_subdomain[i]] = eigenvectors[k][i];
      }
      for (std::size_t i = 0; i < interior_to_subdomain.size(); ++i) {
        combined_vectors[k][interior_to_subdomain[i]] = solutions[k][i];
      }

#ifndef NDEBUG
      double sum = 0;
      for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
        sum += std::pow(eigenvectors[k][i] - combined_vectors[k][ring_to_subdomain[i]], 2);
      }
      sum = std::sqrt(sum);
      if (sum > 1e-12) {
        spdlog::get("all_ranks")->warn("Eigenvector on ring and computed solution in interior differ at boundary, difference is {}", sum);
      }
#endif
    }

    eigenvectors = std::move(combined_vectors);
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
