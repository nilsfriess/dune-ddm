#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/foreach.hh>
#include <dune/istl/io.hh>

#include <spdlog/spdlog.h>

#include "coarsespaces/energy_minimal_extension.hh"
#include "datahandles.hh"
#include "eigensolvers.hh"
#include "helpers.hh"
#include "logger.hh"

template <class Vec, class Mat, class RemoteIndices>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> buildGenEOCoarseSpace(const RemoteIndices &ovlp_ids, const Mat &Aovlp, const std::vector<TripleWithRank> &remote_ncorr_triples,
                                                                                   const std::vector<TripleWithRank> &own_ncorr_triples, const std::vector<bool> &interior_dof_mask,
                                                                                   const Vec &dirichlet_mask_novlp, const Vec &pou, const Dune::ParameterTree &ptree)
{
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

  int rank{};
  MPI_Comm_rank(ovlp_ids.first->communicator(), &rank);
  const auto &ovlp_paridxs = *ovlp_ids.second;

  std::unordered_map<std::size_t, std::size_t> subdomain_to_ring;
  std::vector<std::size_t> ring_to_subdomain(ovlp_paridxs.size());
  std::set<std::size_t> ring_boundary; // dofs on the interior ring boundary (using overlapping subdomain numbering)

  Mat A; // The left-hand side of the eigenproblem. Will be set below depending on the requested type
  Mat B; // The right-hand side of the eigenproblem. Will be set below depending on the requested type
  if (geneo_type == "full" or geneo_type == "overlap") {
    // The lhs matrix of the eigenproblem is the same in both cases
    A = apply_neumann_corrections(Aovlp, remote_ncorr_triples, dirichlet_mask_ovlp, ovlp_paridxs);

    // Next, build the rhs matrix of the eigenproblem.
    if (geneo_type == "overlap") {
      // This is the classical GenEO coarse space. Here, we start by copying Aovlp but ignore all interior dofs;
      auto avg = Aovlp.nonzeroes() / Aovlp.N() + 2;
      B.setBuildMode(Mat::implicit);
      B.setImplicitBuildModeParameters(avg, 0.2);
      B.setSize(Aovlp.N(), Aovlp.N());
      for (auto ri = Aovlp.begin(); ri != Aovlp.end(); ++ri) {
        if (ri.index() < interior_dof_mask.size() and interior_dof_mask[ri.index()]) {
          continue;
        }

        for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
          if (ci.index() < interior_dof_mask.size() and interior_dof_mask[ci.index()]) {
            continue;
          }

          B.entry(ri.index(), ci.index()) = *ci;
        }
      }
      B.compress();

      // Finally apply the corrections to get Neumann boundary conditions on the interior boundary
      for (const auto &triple : own_ncorr_triples) {
        B[triple.row][triple.col] -= triple.val;
      }
    }
    else if (geneo_type == "full") {
      B = Aovlp;
    }

    // In both cases, the right-hand side matrix has to multiplied by the partition of unity from the left and right.
    // We assume that the POU is zero on the subdomain boundary.
    for (auto ri = B.begin(); ri != B.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        *ci *= pou[ri.index()] * pou[ci.index()];
      }
    }

    // Lastly, eliminate the Dirichlet dofs symmetrically (since the eigensolver expectes a symmetric problem).
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
  else if (geneo_type == "ring") {
    // In this case, we solve the eigenvalue problem only in ther overlapping region (which we will refer to here as the ring).
    // The solution on the whole overlapping subdomain is then obtained by extending each eigenvector energy-minimally to
    // the interior.

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

    for (const auto &idx : ring_to_subdomain) {
      for (auto cit = Aovlp[idx].begin(); cit != Aovlp[idx].end(); ++cit) {
        if (not subdomain_to_ring.contains(cit.index())) {
          ring_boundary.insert(idx);
          break;
        }
      }
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
          A.entry(subdomain_to_ring[ri.index()], subdomain_to_ring[ci.index()]) = *ci;
        }
      }
    }
    A.compress();

    std::size_t apply_cnt = 0;
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
          apply_cnt++;
        }
        else {
          spdlog::get("all_ranks")->error("Global index ({}, {}) does not exist in ring", triple.row, triple.col);
        }
      }
      else {
        spdlog::debug("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
      }
    }
    spdlog::get("all_ranks")->debug("Applied {} outer corrections for A", apply_cnt);
    apply_cnt = 0;

    // ... then the "inside" boundary
    for (const auto &triple : own_ncorr_triples) {
      if (subdomain_to_ring.contains(triple.row) && subdomain_to_ring.contains(triple.col)) {
        if (not ring_boundary.contains(triple.row) or not ring_boundary.contains(triple.col)) {
          spdlog::get("all_ranks")->warn("Local index ({}, {}) not on interior/ring boundary", triple.row, triple.col);
        }
        A[subdomain_to_ring[triple.row]][subdomain_to_ring[triple.col]] -= triple.val;
        apply_cnt++;
      }
      else {
        spdlog::get("all_ranks")->error("Local index ({}, {}) does not exist in ring", triple.row, triple.col);
      }
    }

    spdlog::get("all_ranks")->debug("Applied {} inner corrections for A", apply_cnt);

    // Dirichlet conditions
    for (std::size_t i = 0; i < A.N(); ++i) {
      if (subdomain_to_ring.contains(i) and dirichlet_mask_ovlp[ring_to_subdomain[i]] > 0) {
        for (auto ci = A[i].begin(); ci != A[i].end(); ++ci) {
          *ci = (ci.index() == i) ? 1.0 : 0.0;
        }
      }
      else {
        for (auto ci = A[i].begin(); ci != A[i].end(); ++ci) {
          if (subdomain_to_ring.contains(ci.index()) and dirichlet_mask_ovlp[ring_to_subdomain[ci.index()]] > 0) {
            *ci = 0.0;
          }
        }
      }
    }

    B = A; // B starts as A, but has to be multiplied by the partition of unity

    // apply_cnt = 0;
    // for (const auto &triple : own_ncorr_triples2) {
    //   if (subdomain_to_ring.contains(triple.row) && subdomain_to_ring.contains(triple.col)) {
    //     B[subdomain_to_ring[triple.row]][subdomain_to_ring[triple.col]] -= triple.val;
    //     apply_cnt++;
    //   }
    //   else {
    //     spdlog::get("all_ranks")->error("Local index ({}, {}) does not exist in ring", triple.row, triple.col);
    //   }
    // }
    // spdlog::get("all_ranks")->debug("Applied {} additional inner corrections for B", apply_cnt);

    // if (not ptree.get("geneo_ring_inner_neumann", true)) {
    //   // If the rhs matrix should not have Neumann boundary conditions on the inner boundary, we re-extract the corresponding values from the original matrix again
    //   for (const auto &triple : own_ncorr_triples) {
    //     if (subdomain_to_ring.contains(triple.row) && subdomain_to_ring.contains(triple.col)) {
    //       B[subdomain_to_ring[triple.row]][subdomain_to_ring[triple.col]] = Aovlp[triple.row][triple.col];
    //     }
    //   }

    //   // Reapply Dirichlet conditions
    //   for (std::size_t i = 0; i < B.N(); ++i) {
    //     if (dirichlet_mask_ovlp[ring_to_subdomain[i]] > 0) {
    //       for (auto ci = B[i].begin(); ci != B[i].end(); ++ci) {
    //         *ci = (ci.index() == i) ? 1.0 : 0.0;
    //       }
    //     }
    //     else {
    //       for (auto ci = B[i].begin(); ci != B[i].end(); ++ci) {
    //         if (dirichlet_mask_ovlp[ring_to_subdomain[ci.index()]] > 0) {
    //           *ci = 0.0;
    //         }
    //       }
    //     }
    //   }
    // }

    // for (auto ri = B.begin(); ri != B.end(); ++ri) {
    //   for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
    //     if (ring_boundary.contains(ri.index()) or ring_boundary.contains(ci.index())) {
    //       *ci = 0;
    //     }
    //     else {
    //       *ci *= pou[ring_to_subdomain[ri.index()]] * pou[ring_to_subdomain[ci.index()]];
    //     }
    //   }
    // }

    // Now create the B matrix
    // B.setBuildMode(Mat::implicit);
    // B.setImplicitBuildModeParameters(avg, 0.2);
    // B.setSize(N, N);

    // for (auto ri = Aovlp.begin(); ri != Aovlp.end(); ++ri) {
    //   for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
    //     if (subdomain_to_ring.contains(ri.index()) && subdomain_to_ring.contains(ci.index()) && !ring_boundary.contains(ri.index()) && !ring_boundary.contains(ci.index())) {
    //       B.entry(subdomain_to_ring[ri.index()], subdomain_to_ring[ci.index()]) = *ci;
    //     }
    //   }
    // }
    // B.compress();

    // apply_cnt = 0;
    // // We again need to apply the Neumann corrections. First do the "outside" boundary ...
    // for (const auto &triple : remote_ncorr_triples) {
    //   // The triples use global indices, so we first have to convert them to local indices
    //   // on the overlapping subdomain. Also, we might have received some indices that are
    //   // outside of our overlapping subdomain, so we first have to check that.
    //   if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
    //     auto lrow = ovlp_paridxs[triple.row].local();
    //     auto lcol = ovlp_paridxs[triple.col].local();

    //     // Check if the given entry corresponds to a dof inside the ring
    //     if (subdomain_to_ring.contains(lrow) && subdomain_to_ring.contains(lcol)) {
    //       B[subdomain_to_ring[lrow]][subdomain_to_ring[lcol]] -= triple.val;
    //       apply_cnt++;
    //     }
    //     else {
    //       spdlog::get("all_ranks")->error("Global index ({}, {}) does not exist in ring", triple.row, triple.col);
    //     }
    //   }
    //   else {
    //     spdlog::debug("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
    //   }
    // }
    // spdlog::get("all_ranks")->debug("Applied {} outer corrections for B", apply_cnt);
    // apply_cnt = 0;

    // ... then the "inside" boundary
    // for (const auto &triple : own_ncorr_triples2) {
    //   if (subdomain_to_ring.contains(triple.row) && subdomain_to_ring.contains(triple.col)) {
    //     if (ring_boundary.contains(triple.row) or ring_boundary.contains(triple.col)) {
    //       spdlog::get("all_ranks")->warn("Local index ({}, {}) on interior/ring boundary but should not be", triple.row, triple.col);
    //     }
    //     B[subdomain_to_ring[triple.row]][subdomain_to_ring[triple.col]] -= triple.val;
    //     apply_cnt++;
    //   }
    //   else {
    //     spdlog::get("all_ranks")->error("Local index ({}, {}) does not exist in ring", triple.row, triple.col);
    //   }
    // }

    IdentifyBoundaryDataHandle ibdh(Aovlp, *ovlp_ids.second);
    communicator.forward(ibdh);
    auto &boundary_mask = ibdh.get_boundary_mask();

    std::vector<int> boundary_dst(boundary_mask.size(), std::numeric_limits<int>::max() - 1);
    for (std::size_t i = 0; i < boundary_mask.size(); ++i) {
      if (boundary_mask[i]) {
        boundary_dst[i] = 0;
      }
    }

    int overlap = ptree.get("overlap", 1);
    int shrink = ptree.get("pou_shrink", 0);
    for (int round = 0; round <= 4 * overlap; ++round) {
      for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
        for (auto cIt = Aovlp[i].begin(); cIt != Aovlp[i].end(); ++cIt) {
          boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cIt.index()] + 1); // Increase distance from boundary by one
        }
      }
    }

    auto pou_copy = pou;
    for (std::size_t i = 0; i < pou.N(); ++i) {
      if (boundary_dst[i] >= 2 * overlap - shrink) {
        pou_copy[i] = 0;
      }
    }

    // spdlog::get("all_ranks")->debug("Applied {} inner corrections for B", apply_cnt);

    // for (const auto &triple : own_ncorr_triples) {
    //   if (subdomain_to_ring.contains(triple.row) && subdomain_to_ring.contains(triple.col)) {
    //     B[subdomain_to_ring[triple.row]][subdomain_to_ring[triple.col]] = 0;
    //   }
    //   else {
    //     spdlog::get("all_ranks")->error("Local index ({}, {}) does not exist in ring", triple.row, triple.col);
    //   }
    // }

    // Dirichlet conditions
    for (std::size_t i = 0; i < B.N(); ++i) {
      if (subdomain_to_ring.contains(i) and dirichlet_mask_ovlp[ring_to_subdomain[i]] > 0) {
        for (auto ci = B[i].begin(); ci != B[i].end(); ++ci) {
          *ci = (ci.index() == i) ? 1.0 : 0.0;
        }
      }
      else {
        for (auto ci = B[i].begin(); ci != B[i].end(); ++ci) {
          if (subdomain_to_ring.contains(ci.index()) and dirichlet_mask_ovlp[ring_to_subdomain[ci.index()]] > 0) {
            *ci = 0.0;
          }
        }
      }
    }

    for (auto ri = B.begin(); ri != B.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        if (ring_boundary.contains(ring_to_subdomain[ri.index()]) or ring_boundary.contains(ring_to_subdomain[ci.index()])) {
          *ci = 0;
        }
        else {
          *ci *= pou_copy[ring_to_subdomain[ri.index()]] * pou_copy[ring_to_subdomain[ci.index()]];
        }
      }
    }
  }
  else {
    spdlog::error("Unknown GenEO type '{}', aborting", geneo_type);
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

  Eigensolver eigensolver = Eigensolver::Spectra;
  auto eigensolver_string = ptree.get("eigensolver", "spectra");
  std::shared_ptr<Dune::Preconditioner<Dune::BlockVector<Dune::FieldVector<double, 1>>, Dune::BlockVector<Dune::FieldVector<double, 1>>>> prec{nullptr};

  if (eigensolver_string == "spectra") {
    eigensolver = Eigensolver::Spectra;
  }
  else if (eigensolver_string == "blopex") {
    eigensolver = Eigensolver::BLOPEX;
    if (ptree.get("blopex_prec_dirichlet", true)) {
      // prec = std::make_shared<Dune::InverseOperator2Preconditioner<Solver>>(subdomain_solver);
      // TODO: Fix this
    }
    else {
      prec = std::make_shared<Dune::SeqILDL<Mat, Vec, Vec>>(A);
    }
  }
  else {
    spdlog::warn("Unknown eigensolver type '{}' using Spectra instead", eigensolver_string);
  }

  Logger::get().startEvent(eigensolver_event);
  spdlog::info("Solving generalized eigenvalue problem for GenEO");

  if (rank == 0) {
    Dune::writeMatrixToMatlab(A, "A.mat");
    Dune::writeMatrixToMatlab(B, "B.mat");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  auto eigenvectors = solveGEVP(A, B, eigensolver, ptree);

  Logger::get().endEvent(eigensolver_event);

  // If we have only solved the eigenvalue problem on the ring, then we now need to extend the eigenvectors to the interior.
  if (geneo_type == "ring") {
    if (ptree.get("geneo_ring_extend_exact", true)) {
      // Create the matrix that lives on the interior dofs. To this end, first create a interior-to-subdomain map.
      auto N = std::count_if(interior_dof_mask.begin(), interior_dof_mask.end(), [](auto val) { return val; });
      spdlog::get("all_ranks")
          ->debug("For GenEO ring problem: total dofs {}, interior dofs {}, ring dofs {}, ring+interior dofs {}", Aovlp.N(), N, ring_to_subdomain.size(), (N + ring_to_subdomain.size()));
      std::vector<std::size_t> interior_to_subdomain(N);
      std::size_t cnt = 0;
      for (std::size_t i = 0; i < interior_dof_mask.size(); ++i) {
        if (interior_dof_mask[i] > 0) {
          interior_to_subdomain[cnt++] = i;
        }
      }

      auto *factorise_interior = Logger::get().registerEvent("GenEO", "factorise interior");
      auto *solve_interior = Logger::get().registerEvent("GenEO", "solve interior");

      // The energy-minimal extension is computed using the values of the eigenvectors "one layer into the ring"

      // The interior now becomes the old interior+the ring boundary
      for (const auto &idx : ring_boundary) {
        interior_to_subdomain.push_back(idx);
      }

      std::vector<std::size_t> inside_ring_boundary_to_subdomain;
      inside_ring_boundary_to_subdomain.reserve(ring_boundary.size());
      for (const auto &idx : ring_to_subdomain) {
        for (auto cit = Aovlp[idx].begin(); cit != Aovlp[idx].end(); ++cit) {
          if (not ring_boundary.contains(idx) and ring_boundary.contains(cit.index())) {
            inside_ring_boundary_to_subdomain.push_back(idx);
            break;
          }
        }
      }
      spdlog::get("all_ranks")->debug("Identified {} dofs on the inside ring boundary", inside_ring_boundary_to_subdomain.size());

      // Invert the mapping
      std::unordered_map<std::size_t, std::size_t> subdomain_to_inside_ring_boundary;
      subdomain_to_inside_ring_boundary.reserve(inside_ring_boundary_to_subdomain.size());
      for (std::size_t i = 0; i < inside_ring_boundary_to_subdomain.size(); ++i) {
        subdomain_to_inside_ring_boundary[inside_ring_boundary_to_subdomain[i]] = i;
      }

      Logger::get().startEvent(factorise_interior);
      EnergyMinimalExtension<Mat, Vec> extension(Aovlp, interior_to_subdomain, inside_ring_boundary_to_subdomain, ptree.get("geneo_ring_inexact_interior_solver", false));
      Logger::get().endEvent(factorise_interior);

      double eigenvectors_use_portion = ptree.get("geneo_ring_eigenvectors_use_portion", 1.0);
      auto eigenvectors_actual = static_cast<std::size_t>(std::ceil(eigenvectors.size() * eigenvectors_use_portion));

      Vec zero(Aovlp.N());
      zero = 0;
      std::vector<Vec> combined_vectors(eigenvectors_actual, zero);
      Logger::get().startEvent(solve_interior);
      for (std::size_t k = 0; k < eigenvectors_actual; ++k) {
        const auto &evec = eigenvectors[k];

        Vec evec_dirichlet(inside_ring_boundary_to_subdomain.size());
        for (std::size_t i = 0; i < evec.N(); ++i) {
          auto subdomain_idx = ring_to_subdomain[i];
          if (subdomain_to_inside_ring_boundary.contains(subdomain_idx)) {
            evec_dirichlet[subdomain_to_inside_ring_boundary[subdomain_idx]] = evec[i];
          }
        }

        auto interior_vec = extension.extend(evec_dirichlet);
        // First set the values in the ring
        for (std::size_t i = 0; i < evec.N(); ++i) {
          combined_vectors[k][ring_to_subdomain[i]] = evec[i];
        }
        // Next fill the interior values (note that the interior and ring now overlap, so this overrides some values)
        for (std::size_t i = 0; i < interior_vec.N(); ++i) {
          combined_vectors[k][interior_to_subdomain[i]] = interior_vec[i];
        }
      }
      Logger::get().endEvent(solve_interior);

      eigenvectors = std::move(combined_vectors);
    }
    else {
      assert(false && "Not implemented");
    }
  }

  for (auto &vec : eigenvectors) {
    for (std::size_t i = 0; i < vec.N(); ++i) {
      if (dirichlet_mask_ovlp[i] > 0) {
        vec[i] = 0;
      }
      else {
        if (ptree.get("basis_vec_mult_pou", true)) {
          vec[i] *= pou[i];
        }
      }
    }
  }

  for (auto &vec : eigenvectors) {
    vec *= 1. / vec.two_norm();
  }

  return eigenvectors;
}
