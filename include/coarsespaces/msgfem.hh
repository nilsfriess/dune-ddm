#pragma once

#include "../datahandles.hh"
#include "../helpers.hh"
#include "eigensolvers.hh"
#include "helpers.hh"

#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/bvector.hh>

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

  Mat A; // The left-hand side of the eigenproblem. Will be set below depending on the requested type
  Mat B; // The right-hand side of the eigenproblem. Will be set below depending on the requested type
  if (msgfem_type == "standard") {
    // First we setup the Neumann subdomain matrix
    Mat Aneu = apply_neumann_corrections(Aovlp, remote_ncorr_triples, dirichlet_mask_ovlp, *ovlp_ids.second);

    /* Next, split the dofs into three sets that form a partition of the subdomain dofs:
       B1 = { interior subdomain dofs }  (Neumann boundary dofs are counted as interior)
       B2 = { subdomain boundary dofs that are not part of the global boundary (Dirichlet or Neumann) }
       B3 = { subdomain boundary dofs that are also part of the global Dirichlet boundary }
     */
    enum class DOFType : std::uint8_t { Interior, Boundary, Dirichlet };

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
    for (auto ri = Aneu.begin(); ri != Aneu.end(); ++ri) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        *ci *= pou[ri.index()] * pou[ci.index()];
      }
    }

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
          B.entry(ri, rj) = *cit;
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
    assert(false && "Not implemented");
  }
  else {
    spdlog::error("Unknown MsGFEM type '{}', aborting", msgfem_type);
    MPI_Abort(MPI_COMM_WORLD, 13);
  }

  return {};
}
