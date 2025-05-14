#pragma once

#include "../datahandles.hh"
#include "../helpers.hh"
#include "coarsespaces/energy_minimal_extension.hh"
#include "eigensolvers.hh"
#include "helpers.hh"

#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/bvector.hh>

enum class DOFType : std::uint8_t { Interior, Boundary, Dirichlet };

template <class Vec, class Mat, class RemoteIndices>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> buildMsGFEMCoarseSpace(const RemoteIndices &ovlp_ids, const Mat &Aovlp, const std::vector<TripleWithRank> &remote_ncorr_triples,
                                                                                    const std::vector<TripleWithRank> &own_ncorr_triples, const std::vector<bool> &interior_dof_mask,
                                                                                    const Vec &dirichlet_mask_novlp, const Vec &pou, const Dune::ParameterTree &ptree)
{
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

  // Next, create a mask that identifies that subdomain boundary (without the global boundary)
  IdentifyBoundaryDataHandle ibdh(Aovlp, *ovlp_ids.second);
  communicator.forward(ibdh);
  auto boundary_mask = ibdh.get_boundary_mask();

  // Find out what type of MsGFEM eigenproblem we should solve
  auto msgfem_type = ptree.get("msgfem_type", "standard");
  spdlog::info("Setting up MsGFEM coarse space in mode '{}'", msgfem_type);

  // Use in the'ring' case
  std::unordered_map<std::size_t, std::size_t> subdomain_to_ring;
  std::vector<std::size_t> ring_to_subdomain(ovlp_ids.second->size());

  Mat A; // The left-hand side of the eigenproblem. Will be set below depending on the requested type
  Mat B; // The right-hand side of the eigenproblem. Will be set below depending on the requested type
  if (msgfem_type == "standard") {
    // First we setup the Neumann subdomain matrix
    // TODO: Here we create the matrix Aneu and then copy its contents to another matrix (A).
    //       We could instead just copy Aovlp to A and apply the Neumann corrections there.
    Mat Aneu = apply_neumann_corrections(Aovlp, remote_ncorr_triples, dirichlet_mask_ovlp, *ovlp_ids.second);

    /* Next, split the dofs into three sets that form a partition of the subdomain dofs:
       B1 = { interior subdomain dofs }  (Neumann boundary dofs are counted as interior)
       B2 = { subdomain boundary dofs that are not part of the global boundary (Dirichlet or Neumann) }
       B3 = { subdomain boundary dofs that are also part of the global Dirichlet boundary }
     */
    std::vector<DOFType> dof_partitioning(Aneu.N());
    std::size_t num_interior = 0;
    std::size_t num_boundary = 0;
    std::size_t num_dirichlet = 0;
    for (std::size_t i = 0; i < Aneu.N(); ++i) {
      if (dirichlet_mask_ovlp[i] > 0) {
        dof_partitioning[i] = DOFType::Dirichlet;
        num_dirichlet++;
      }
      else if (boundary_mask[i]) {
        // TODO: Here we add dofs on the Neumann boundary to the set B2 defined above, which shouldn't contain
        //       dofs on the Neumann boundary. I don't think there is a simple way to identify the Neumann boundary,
        //       we would need to create a mask during assembly.
        dof_partitioning[i] = DOFType::Boundary;
        num_boundary++;
      }
      else {
        dof_partitioning[i] = DOFType::Interior;
        num_interior++;
      }
    }
    spdlog::get("all_ranks")->debug("Partitioned dofs, have {} in interior, {} on subdomain boundary, {} on Dirichlet boundary", num_interior, num_boundary, num_dirichlet);

    // Create a reordered index set: first interior dofs, then boundary dofs, then Dirichlet dofs
    std::vector<std::size_t> reordering(Aneu.N());
    std::size_t cnt_interior = 0;
    std::size_t cnt_boundary = num_interior;
    std::size_t cnt_dirichlet = num_interior + num_boundary;
    for (std::size_t i = 0; i < reordering.size(); ++i) {
      if (dof_partitioning[i] == DOFType::Interior) {
        reordering[i] = cnt_interior++;
      }
      else if (dof_partitioning[i] == DOFType::Boundary) {
        reordering[i] = cnt_boundary++;
      }
      else {
        reordering[i] = cnt_dirichlet++;
      }
    }

    // Assemble the left-hand side of the eigenproblem
    const auto n_big = num_interior + num_boundary + num_interior; // size of the big eigenproblem, including the harmonicity constraint
    const auto avg = 2 * (Aneu.nonzeroes() / Aneu.N());
    A.setBuildMode(Mat::implicit);
    A.setImplicitBuildModeParameters(avg, 0.2);
    A.setSize(n_big, n_big);

    // Assemble the part corresponding to the a-harmonic constraint
    for (auto rit = Aneu.begin(); rit != Aneu.end(); ++rit) {
      auto ii = rit.index();
      auto ri = reordering[ii];
      if (dof_partitioning[ii] != DOFType::Interior) {
        continue;
      }

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto jj = cit.index();
        auto rj = reordering[jj];

        if (dof_partitioning[jj] != DOFType::Dirichlet) {
          A.entry(rj, num_interior + num_boundary + ri) = *cit;
          A.entry(num_interior + num_boundary + ri, rj) = *cit;
        }
      }
    }

    // Assemble the remaining part of the matrix
    for (auto rit = Aneu.begin(); rit != Aneu.end(); ++rit) {
      auto ii = rit.index();
      auto ri = reordering[ii];
      if (dof_partitioning[ii] == DOFType::Dirichlet) { // Skip Dirchlet dofs
        continue;
      }

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto jj = cit.index();
        auto rj = reordering[jj];

        if (dof_partitioning[jj] != DOFType::Dirichlet) {
          A.entry(ri, rj) = *cit;
        }
      }
    }
    A.compress();

    // Next, assemble the right-hand side of the eigenproblem
    B.setBuildMode(Mat::implicit);
    B.setImplicitBuildModeParameters(avg, 0.2);
    B.setSize(n_big, n_big);

    for (auto rit = Aneu.begin(); rit != Aneu.end(); ++rit) {
      auto ii = rit.index();
      auto ri = reordering[ii];
      if (dof_partitioning[ii] != DOFType::Interior) {
        continue;
      }

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto jj = cit.index();
        auto rj = reordering[jj];

        if (dof_partitioning[jj] == DOFType::Interior) {
          B.entry(ri, rj) = pou[ii] * pou[jj] * (*cit);
        }
      }
    }

    B.compress();

    auto eigenvectors = solveGEVP(A, B, Eigensolver::Spectra, ptree);

    Vec v(Aovlp.N());
    v = 0;
    std::vector<Vec> eigenvectors_actual(eigenvectors.size(), v);
    for (std::size_t k = 0; k < eigenvectors.size(); ++k) {
      for (std::size_t i = 0; i < Aovlp.N(); ++i) {
        if (dof_partitioning[i] != DOFType::Dirichlet) {
          eigenvectors_actual[k][i] = eigenvectors[k][reordering[i]];
        }
      }
    }

    for (auto &vec : eigenvectors_actual) {
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

    for (auto &vec : eigenvectors_actual) {
      vec *= 1. / vec.two_norm();
    }

    return eigenvectors_actual;
  }
  else if (msgfem_type == "ring") {
    // TODO: There is some duplicate code here and the GenEO coarse space setup function.

    // First we have to create the matrix on the ring. To this end, we first
    // need to identify all degrees of freedom inside the ring.These are
    // simply all dofs not in the interior.
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

    // Next, identify the inside ring boundary
    std::set<std::size_t> ring_boundary; // dofs on the interior ring boundary (using overlapping subdomain numbering)
    for (const auto &idx : ring_to_subdomain) {
      for (auto cit = Aovlp[idx].begin(); cit != Aovlp[idx].end(); ++cit) {
        if (not subdomain_to_ring.contains(cit.index())) {
          ring_boundary.insert(idx);
          break;
        }
      }
    }

    // Next, partition the ring dofs as in the standard case
    std::vector<DOFType> dof_partitioning(ring_to_subdomain.size());
    std::size_t num_interior = 0;
    std::size_t num_boundary = 0;
    std::size_t num_dirichlet = 0;
    for (std::size_t i = 0; i < dof_partitioning.size(); ++i) {
      if (dirichlet_mask_ovlp[ring_to_subdomain[i]] > 0) {
        dof_partitioning[i] = DOFType::Dirichlet;
        num_dirichlet++;
      }
      else if (boundary_mask[ring_to_subdomain[i]] or ring_boundary.contains(ring_to_subdomain[i])) {
        // TODO: Here we add dofs on the Neumann boundary to the set B2 defined above, which shouldn't contain
        //       dofs on the Neumann boundary. I don't think there is a simple way to identify the Neumann boundary,
        //       we would need to create a mask during assembly.
        dof_partitioning[i] = DOFType::Boundary;
        num_boundary++;
      }
      else {
        dof_partitioning[i] = DOFType::Interior;
        num_interior++;
      }
    }
    spdlog::get("all_ranks")->debug("Partitioned dofs inside ring, have {} in interior, {} on subdomain boundary, {} on Dirichlet boundary", num_interior, num_boundary, num_dirichlet);

    // Create a reordered index set: first interior dofs, then boundary dofs, then Dirichlet dofs
    std::vector<std::size_t> reordering(ring_to_subdomain.size());
    std::size_t cnt_interior = 0;
    std::size_t cnt_boundary = num_interior;
    std::size_t cnt_dirichlet = num_interior + num_boundary;
    for (std::size_t i = 0; i < reordering.size(); ++i) {
      if (dof_partitioning[i] == DOFType::Interior) {
        reordering[i] = cnt_interior++;
      }
      else if (dof_partitioning[i] == DOFType::Boundary) {
        reordering[i] = cnt_boundary++;
      }
      else {
        reordering[i] = cnt_dirichlet++;
      }
    }

    // Assemble the left-hand side of the eigenproblem
    const auto n_big = num_interior + num_boundary + num_interior; // size of the big eigenproblem, including the harmonicity constraint
    const auto avg = 2 * (Aovlp.nonzeroes() / Aovlp.N());
    A.setBuildMode(Mat::implicit);
    A.setImplicitBuildModeParameters(avg, 0.2);
    A.setSize(n_big, n_big);

    // Assemble the part corresponding to the a-harmonic constraint
    // TODO: Can we avoid querying the subdomain-to-ring map so often here (e.g. by iterating the ring-to-subdomain instead of the matrix)?
    for (auto rit = Aovlp.begin(); rit != Aovlp.end(); ++rit) {
      auto sii = rit.index();
      if (!subdomain_to_ring.contains(sii)) {
        continue;
      }

      auto ii = subdomain_to_ring[sii];
      auto ri = reordering[ii];
      if (dof_partitioning[ii] != DOFType::Interior) {
        continue;
      }

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto sjj = cit.index();
        if (!subdomain_to_ring.contains(sjj)) {
          continue;
        }

        auto jj = subdomain_to_ring[sjj];
        auto rj = reordering[jj];

        if (dof_partitioning[jj] != DOFType::Dirichlet) {
          A.entry(rj, num_interior + num_boundary + ri) = *cit;
          A.entry(num_interior + num_boundary + ri, rj) = *cit;
        }
      }
    }

    // Assemble the remaining part of the matrix
    for (auto rit = Aovlp.begin(); rit != Aovlp.end(); ++rit) {
      auto sii = rit.index();
      if (!subdomain_to_ring.contains(sii)) {
        continue;
      }

      auto ii = subdomain_to_ring[sii];
      auto ri = reordering[ii];
      if (dof_partitioning[ii] == DOFType::Dirichlet) { // Skip Dirchlet dofs
        continue;
      }

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto sjj = cit.index();
        if (!subdomain_to_ring.contains(sjj)) {
          continue;
        }

        auto jj = subdomain_to_ring[sjj];
        auto rj = reordering[jj];

        if (dof_partitioning[jj] != DOFType::Dirichlet) {
          A.entry(ri, rj) = *cit;
        }
      }
    }
    A.compress();

    // Apply Neumann corrections. First the "outside" corrections
    const auto &ovlp_paridxs = *ovlp_ids.second;
    for (const auto &triple : remote_ncorr_triples) {
      if (ovlp_paridxs.exists(triple.row) && ovlp_paridxs.exists(triple.col)) {
        auto slrow = ovlp_paridxs[triple.row].local();
        auto slcol = ovlp_paridxs[triple.col].local();

        if (!subdomain_to_ring.contains(slrow) or !subdomain_to_ring.contains(slcol)) {
          continue;
        }

        auto rlrow = subdomain_to_ring[slrow];
        auto rlcol = subdomain_to_ring[slcol];

        if (dof_partitioning[rlrow] == DOFType::Dirichlet or dof_partitioning[rlcol] == DOFType::Dirichlet) {
          continue;
        }

        auto lrow = reordering[rlrow];
        auto lcol = reordering[rlcol];

        A[lrow][lcol] -= triple.val;
      }
      else {
        spdlog::debug("Global index ({}, {}) does not exist in subdomain", triple.row, triple.col);
      }
    }

    // Next, the "inner" corrections
    for (const auto &triple : own_ncorr_triples) {
      if (subdomain_to_ring.contains(triple.row) and subdomain_to_ring.contains(triple.col)) {
        auto rlrow = subdomain_to_ring[triple.row];
        auto rlcol = subdomain_to_ring[triple.col];

        if (dof_partitioning[rlrow] == DOFType::Dirichlet or dof_partitioning[rlcol] == DOFType::Dirichlet) {
          continue;
        }

        auto lrow = reordering[rlrow];
        auto lcol = reordering[rlcol];

        A[lrow][lcol] -= triple.val;
      }
      else {
        spdlog::get("all_ranks")->error("Local index ({}, {}) does not exist in ring", triple.row, triple.col);
      }
    }

    // Next, assemble the right-hand side of the eigenproblem
    B.setBuildMode(Mat::implicit);
    B.setImplicitBuildModeParameters(avg, 0.2);
    B.setSize(n_big, n_big);

    for (auto rit = Aovlp.begin(); rit != Aovlp.end(); ++rit) {
      auto sii = rit.index();
      if (!subdomain_to_ring.contains(sii)) {
        continue;
      }

      auto ii = subdomain_to_ring[sii];
      auto ri = reordering[ii];
      if (dof_partitioning[ii] == DOFType::Dirichlet) { // Skip Dirchlet dofs
        continue;
      }

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto sjj = cit.index();
        if (!subdomain_to_ring.contains(sjj)) {
          continue;
        }

        auto jj = subdomain_to_ring[sjj];
        auto rj = reordering[jj];

        if (dof_partitioning[jj] != DOFType::Dirichlet) {
          B.entry(ri, rj) = *cit;
        }
      }
    }
    B.compress();

    // // Apply inner corrections
    // for (const auto &triple : own_ncorr_triples2) {
    //   if (subdomain_to_ring.contains(triple.row) and subdomain_to_ring.contains(triple.col)) {
    //     auto rlrow = subdomain_to_ring[triple.row];
    //     auto rlcol = subdomain_to_ring[triple.col];

    //     if (dof_partitioning[rlrow] == DOFType::Dirichlet or dof_partitioning[rlcol] == DOFType::Dirichlet) {
    //       continue;
    //     }

    //     auto lrow = reordering[rlrow];
    //     auto lcol = reordering[rlcol];

    //     B[lrow][lcol] -= triple.val;
    //   }
    //   else {
    //     spdlog::get("all_ranks")->error("Local index ({}, {}) does not exist in ring", triple.row, triple.col);
    //   }
    // }

    // // TODO: Find a more efficient way to zero the zero-blocks in B.
    // for (auto rit = Aovlp.begin(); rit != Aovlp.end(); ++rit) {
    //   auto sii = rit.index();
    //   if (!subdomain_to_ring.contains(sii)) {
    //     continue;
    //   }

    //   auto ii = subdomain_to_ring[sii];
    //   auto ri = reordering[ii];
    //   if (dof_partitioning[ii] != DOFType::Interior) {
    //     continue;
    //   }

    //   for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
    //     auto sjj = cit.index();
    //     if (!subdomain_to_ring.contains(sjj)) {
    //       continue;
    //     }

    //     auto jj = subdomain_to_ring[sjj];
    //     auto rj = reordering[jj];

    //     if (dof_partitioning[jj] != DOFType::Dirichlet) {
    //       B[rj][num_interior + num_boundary + ri] = 0;
    //       B[num_interior + num_boundary + ri][rj] = 0;
    //     }
    //   }
    // }

    // for (auto rit = Aovlp.begin(); rit != Aovlp.end(); ++rit) {
    //   auto sii = rit.index();
    //   if (!subdomain_to_ring.contains(sii)) {
    //     continue;
    //   }

    //   auto ii = subdomain_to_ring[sii];
    //   auto ri = reordering[ii];
    //   if (dof_partitioning[ii] != DOFType::Boundary) {
    //     continue;
    //   }

    //   for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
    //     auto sjj = cit.index();
    //     if (!subdomain_to_ring.contains(sjj)) {
    //       continue;
    //     }

    //     auto jj = subdomain_to_ring[sjj];
    //     auto rj = reordering[jj];

    //     if (dof_partitioning[jj] != DOFType::Boundary) {
    //       continue;
    //     }

    //     B[rj][ri] = 0;
    //     B[ri][rj] = 0;
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

    for (auto rit = Aovlp.begin(); rit != Aovlp.end(); ++rit) {
      auto sii = rit.index();
      if (!subdomain_to_ring.contains(sii)) {
        continue;
      }

      auto ii = subdomain_to_ring[sii];
      auto ri = reordering[ii];
      if (dof_partitioning[ii] == DOFType::Dirichlet) { // Skip Dirchlet dofs
        continue;
      }

      for (auto cit = rit->begin(); cit != rit->end(); ++cit) {
        auto sjj = cit.index();
        if (!subdomain_to_ring.contains(sjj)) {
          continue;
        }

        auto jj = subdomain_to_ring[sjj];
        auto rj = reordering[jj];

        if (dof_partitioning[jj] != DOFType::Dirichlet) {
          B[ri][rj] *= pou_copy[sii] * pou_copy[sjj];
        }
      }
    }

    std::map<std::size_t, std::size_t> reverse_ordering;
    for (std::size_t i = 0; i < reordering.size(); ++i) {
      reverse_ordering[reordering[i]] = i;
    }

    spdlog::get("all_ranks")->info("ring_to_subdomain.size() = {}", ring_to_subdomain.size());

    auto eigenvectors_with_constraint = solveGEVP(A, B, Eigensolver::Spectra, ptree);

    Vec v(ring_to_subdomain.size());
    v = 0;
    std::vector<Vec> eigenvectors(eigenvectors_with_constraint.size(), v);
    for (std::size_t k = 0; k < eigenvectors_with_constraint.size(); ++k) {
      for (std::size_t i = 0; i < ring_to_subdomain.size(); ++i) {
        if (dof_partitioning[i] != DOFType::Dirichlet) {
          eigenvectors[k][i] = eigenvectors_with_constraint[k][reordering[i]];
        }
      }
    }

    // Compute energy-minimising extension
    const auto inner_boundary_dist = overlap + (overlap - shrink) - 1;
    const auto N = std::count_if(boundary_dst.begin(), boundary_dst.end(), [&](auto val) { return val >= inner_boundary_dist; });
    std::vector<std::size_t> interior_to_subdomain;
    interior_to_subdomain.reserve(N);
    for (std::size_t i = 0; i < interior_dof_mask.size(); ++i) {
      if (boundary_dst[i] > inner_boundary_dist) {
        interior_to_subdomain.push_back(i);
      }
    }

    std::vector<std::size_t> inside_ring_boundary_to_subdomain;
    inside_ring_boundary_to_subdomain.reserve(N);
    for (const auto &idx : ring_to_subdomain) {
      if (boundary_dst[idx] == inner_boundary_dist) {
        inside_ring_boundary_to_subdomain.push_back(idx);
      }
    }
    spdlog::get("all_ranks")->debug("Identified {} dofs on the inside ring boundary", inside_ring_boundary_to_subdomain.size());

    // Invert the mapping
    std::unordered_map<std::size_t, std::size_t> subdomain_to_inside_ring_boundary;
    subdomain_to_inside_ring_boundary.reserve(inside_ring_boundary_to_subdomain.size());
    for (std::size_t i = 0; i < inside_ring_boundary_to_subdomain.size(); ++i) {
      subdomain_to_inside_ring_boundary[inside_ring_boundary_to_subdomain[i]] = i;
    }

    EnergyMinimalExtension<Mat, Vec> extension(Aovlp, interior_to_subdomain, inside_ring_boundary_to_subdomain, ptree.get("geneo_ring_inexact_interior_solver", false));

    double eigenvectors_use_portion = ptree.get("msgfem_ring_eigenvectors_use_portion", 1.0);
    auto eigenvectors_actual = static_cast<std::size_t>(std::ceil(eigenvectors.size() * eigenvectors_use_portion));

    Vec zero(Aovlp.N());
    zero = 0;
    std::vector<Vec> combined_vectors(eigenvectors_actual, zero);
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

    eigenvectors = std::move(combined_vectors);

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
  else {
    spdlog::error("Unknown MsGFEM type '{}', aborting", msgfem_type);
    MPI_Abort(MPI_COMM_WORLD, 13);
  }

  return {};
}
