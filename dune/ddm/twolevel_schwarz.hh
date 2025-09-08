#pragma once

#include "coarsespaces/coarse_spaces.hh"
#include "combined_preconditioner.hh"
#include "galerkin_preconditioner.hh"
#include "helpers.hh"
#include "nonoverlapping_operator.hh"
#include "overlap_extension.hh"
#include "schwarz.hh"

#include <dune/common/parametertree.hh>
#include <dune/istl/preconditioner.hh>

#include <memory>

template <class Mat, class RemoteIndices, class X = Dune::BlockVector<Dune::FieldVector<double, 1>>, class Y = X>
class TwoLevelSchwarz : public Dune::Preconditioner<X, Y> {
  template <class Vec>
  struct CopyGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] = v; }
  };

public:
  using FineLevel = SchwarzPreconditioner<X, Mat>;
  using CoarseLevel = GalerkinPreconditioner<X>;

  template <class Vec>
  TwoLevelSchwarz(std::shared_ptr<Mat> A_novlp, const RemoteIndices &remote_ids, const std::vector<Vec> &template_vecs, const Dune::ParameterTree &ptree,
                  const std::string &subtree_name = "twolevelschwarz")
      : prec(ptree, subtree_name)
  {
    const auto &subtree = ptree.sub(subtree_name);

    int overlap = subtree.get("overlap", 1);

    ExtendedRemoteIndices ext_indices(remote_ids, *A_novlp, overlap);
    A = std::make_shared<Mat>(ext_indices.create_overlapping_matrix(*A_novlp));

    auto pou = std::make_shared<PartitionOfUnity>(*A, ext_indices, ptree);
    const AttributeSet ownerAttribute{Attribute::owner};
    const AttributeSet copyAttribute{Attribute::copy};

    Dune::Interface owner_copy_interface;
    owner_copy_interface.build(ext_indices.get_remote_indices(), ownerAttribute, copyAttribute);
    Dune::BufferedCommunicator owner_copy_comm;
    owner_copy_comm.build<Vec>(owner_copy_interface);

    std::vector<Vec> extended_template_vecs(template_vecs.size(), Vec(A->N()));
    for (std::size_t i = 0; i < template_vecs.size(); ++i) {
      extended_template_vecs[i] = 0;
      for (std::size_t j = 0; j < template_vecs[i].N(); ++j) {
        extended_template_vecs[i][j] = template_vecs[i][j];
      }
      owner_copy_comm.forward<CopyGatherScatter<Vec>>(extended_template_vecs[i]);
    }

    POUCoarseSpace coarse_space(extended_template_vecs, *pou);
    auto fine = std::make_shared<FineLevel>(A, ext_indices.get_remote_indices(), pou, subtree, "fine");
    auto coarse = std::make_shared<CoarseLevel>(*A, coarse_space.get_basis(), ext_indices.get_remote_indices(), subtree, "coarse");

    auto op = std::make_shared<NonOverlappingOperator<Mat, Vec>>(A_novlp, remote_ids);
    prec.set_op(op);
    prec.add(fine);
    prec.add(coarse);
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void pre(X &, Y &) override {}
  void post(X &) override {}

  void apply(X &v, const Y &d) override { prec.apply(v, d); }

private:
  std::shared_ptr<Mat> A; /// The overlapping matrix

  std::shared_ptr<FineLevel> fine;
  std::shared_ptr<CoarseLevel> coarse;
  CombinedPreconditioner<X, Y> prec;
};
