#pragma once

#include "coarsespaces/energy_minimal_extension.hh"
#include "helpers.hh"

#include <dune/common/fvector.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/bvector.hh>

#include <vector>

enum class NicolaidesType { Standard, Ring };

template <class Vec, class Mat, class RemoteIndices>
std::vector<Dune::BlockVector<Dune::FieldVector<double, 1>>> buildNicolaidesCoarseSpace(const RemoteIndices &ovlp_ids, const Mat &Aovlp, const std::vector<Vec> &template_vecs,
                                                                                        const std::vector<bool> &interior_dof_mask, const Vec &pou, const Dune::ParameterTree &ptree)
{
  struct CopyGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] = v; }
  };

  // Extend template vectors to overlapping index set and multiply with partition of unity
  const AttributeSet ownerAttribute{Attribute::owner};
  const AttributeSet copyAttribute{Attribute::copy};

  Dune::Interface owner_copy_interface;
  owner_copy_interface.build(ovlp_ids, ownerAttribute, copyAttribute);
  Dune::BufferedCommunicator owner_copy_comm;
  owner_copy_comm.build<Vec>(owner_copy_interface);

  std::vector<Vec> extended_template_vecs(template_vecs.size(), Vec(ovlp_ids.sourceIndexSet().size()));
  for (std::size_t i = 0; i < template_vecs.size(); ++i) {
    extended_template_vecs[i] = 0;
    for (std::size_t j = 0; j < template_vecs[i].N(); ++j) {
      extended_template_vecs[i][j] = template_vecs[i][j];
    }
    owner_copy_comm.forward<CopyGatherScatter>(extended_template_vecs[i]);

    for (std::size_t j = 0; j < extended_template_vecs[i].N(); ++j) {
      extended_template_vecs[i][j] *= pou[j];
    }
  }

  if (ptree.get("nicolaides_typelsp", "standard") == "ring") {
    std::vector<std::size_t> ring_to_subdomain(Aovlp.N());
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < Aovlp.N(); ++i) {
      if ((i < interior_dof_mask.size() and not interior_dof_mask[i]) or i >= interior_dof_mask.size()) {
        ring_to_subdomain[cnt++] = i;
      }
    }
    ring_to_subdomain.resize(cnt);

    auto N = std::count_if(interior_dof_mask.begin(), interior_dof_mask.end(), [](auto val) { return val; });
    std::vector<std::size_t> interior_to_subdomain(N);
    cnt = 0;
    for (std::size_t i = 0; i < interior_dof_mask.size(); ++i) {
      if (interior_dof_mask[i] > 0) {
        interior_to_subdomain[cnt++] = i;
      }
    }

    EnergyMinimalExtension<Mat, Vec> extension(Aovlp, interior_to_subdomain, ring_to_subdomain);

    Vec interior_vec(ring_to_subdomain.size());
    std::vector<Vec> template_vecs_interior(template_vecs.size(), interior_vec);
    for (std::size_t i = 0; i < template_vecs.size(); ++i) {
      for (std::size_t k = 0; k < ring_to_subdomain.size(); ++k) {
        template_vecs_interior[i][k] = extended_template_vecs[i][ring_to_subdomain[k]];
      }

      extended_template_vecs[i] = extension.extend(template_vecs_interior[i]);
    }
  }

  return extended_template_vecs;
}
