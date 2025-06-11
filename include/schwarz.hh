#pragma once

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/parametertree.hh>
#include <dune/istl/io.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>

#include <cstdint>
#include <limits>
#include <memory>
#include <mpi.h>
#include <umfpack.h>

#include "datahandles.hh"
#include "helpers.hh"
#include "logger.hh"
#include "overlap_extension.hh"
#include "spdlog/spdlog.h"

template <typename Mat, typename X, typename Y>
class NonoverlappingOperator : public Dune::AssembledLinearOperator<Mat, X, Y> {
public:
  using matrix_type = Mat;
  using domain_type = X;
  using range_type = Y;
  using field_type = typename X::field_type;

private:
  const Mat &A;
  std::shared_ptr<Dune::BufferedCommunicator> communicator;

  Logger::Event *apply_event;
  Logger::Event *applyscaleadd_event;

  struct AddGatherScatter {
    using DataType = typename Dune::CommPolicy<X>::IndexedType;

    static DataType gather(const X &x, std::size_t i) { return x[i]; }
    static void scatter(X &x, DataType v, std::size_t i) { x[i] += v; }
  };

public:
  NonoverlappingOperator(const Mat &A, std::shared_ptr<Dune::BufferedCommunicator> communicator) : A(A), communicator(std::move(communicator))
  {
    auto *family = Logger::get().registerFamily("NonovlpOperator");
    apply_event = Logger::get().registerEvent(family, "apply");
    applyscaleadd_event = Logger::get().registerEvent(family, "applyscaleadd");
  }

  NonoverlappingOperator(const NonoverlappingOperator &) = delete;
  NonoverlappingOperator(const NonoverlappingOperator &&) = delete;
  NonoverlappingOperator &operator=(const NonoverlappingOperator &) = delete;
  NonoverlappingOperator &operator=(const NonoverlappingOperator &&) = delete;
  ~NonoverlappingOperator() = default;

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void apply(const X &x, Y &y) const override
  {
    Logger::ScopedLog sl(apply_event);
    A.mv(x, y);
    communicator->forward<AddGatherScatter>(y);
  }

  void applyscaleadd(field_type alpha, const X &x, Y &y) const override
  {
    Logger::ScopedLog sl(applyscaleadd_event);

    Y y1(y.N());
    y1 = 0;
    A.usmv(alpha, x, y1);
    communicator->forward<AddGatherScatter>(y1);
    y += y1;
  }

  const Mat &getmat() const override { return A; }
};

enum class SchwarzType : std::uint8_t { Standard, Restricted };

enum class PartitionOfUnityType : std::uint8_t {
  None,     // The trivial partition of unity
  Standard, // 1 / number of subdomains that know a dof
  Distance  // Weighted by the distance to the overlapping subdomain boundary, see Toselli & Widlund, p. 84
};

template <class Vec, class Mat, class ExtendedRemoteIndices, class Solver = Dune::UMFPack<Mat>>
class SchwarzPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  AttributeSet ownerAttribute{Attribute::owner};
  AttributeSet copyAttribute{Attribute::copy};

  // using ParallelIndexSet = typename RemoteIndices::ParallelIndexSet;

  struct AddGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] += v; }
  };

  struct CopyGatherScatter {
    using DataType = double;

    static DataType gather(const Vec &x, std::size_t i) { return x[i]; }
    static void scatter(Vec &x, DataType v, std::size_t i) { x[i] = v; }
  };

public:
  SchwarzPreconditioner(const Mat &A, const ExtendedRemoteIndices &ext_indices, SchwarzType type = SchwarzType::Restricted, PartitionOfUnityType pou_type = PartitionOfUnityType::None,
                        bool factorise_at_first_iteration = false)
      : ext_indices(ext_indices), type(type), pou_type(pou_type), factorise_at_first_iteration(factorise_at_first_iteration)
  {
    spdlog::info("Setting up Schwarz preconditioner in {} mode", type == SchwarzType::Standard ? "standard" : "restricted");

    init(A);
    createPOU(0);
  }

  SchwarzPreconditioner(std::shared_ptr<Mat> Aovlp, const ExtendedRemoteIndices &ext_indices, const Dune::ParameterTree &ptree)
      : Aovlp(std::move(Aovlp)), ext_indices(ext_indices), factorise_at_first_iteration(ptree.get("schwarz_factorise_at_first_iteration", false))
  {
    auto type_string = ptree.get("type", "restricted");
    if (type_string == "restricted") {
      type = SchwarzType::Restricted;
    }
    else if (type_string == "standard") {
      type = SchwarzType::Standard;
    }
    else {
      DUNE_THROW(Dune::Exception, "Unknown Schwarz type, can either be 'restricted' or 'standard'");
    }

    spdlog::info("Setting up Schwarz preconditioner in {} mode", type == SchwarzType::Standard ? "standard" : "restricted");

    auto pou_string = ptree.get("pou", "standard");
    if (pou_string == "standard") {
      pou_type = PartitionOfUnityType::Standard;
    }
    else if (pou_string == "distance") {
      pou_type = PartitionOfUnityType::Distance;
    }
    else if (pou_string == "none") {
      pou_type = PartitionOfUnityType::None;
    }
    else {
      DUNE_THROW(Dune::Exception, "Unknown partition of unity type, can either be 'standard' or 'none'");
    }

    init();
    createPOU(ptree.get("pou_shrink", 0));
  }

  SchwarzPreconditioner(const Mat &A, const ExtendedRemoteIndices &ext_indices, const Dune::ParameterTree &ptree)
      : ext_indices(ext_indices), factorise_at_first_iteration(ptree.get("schwarz_factorise_at_first_iteration", false))
  {
    auto type_string = ptree.get("type", "restricted");
    if (type_string == "restricted") {
      type = SchwarzType::Restricted;
    }
    else if (type_string == "standard") {
      type = SchwarzType::Standard;
    }
    else {
      DUNE_THROW(Dune::Exception, "Unknown Schwarz type, can either be 'restricted' or 'standard'");
    }

    spdlog::info("Setting up Schwarz preconditioner in {} mode", type == SchwarzType::Standard ? "standard" : "restricted");

    auto pou_string = ptree.get("pou", "standard");
    if (pou_string == "standard") {
      pou_type = PartitionOfUnityType::Standard;
    }
    else if (pou_string == "distance") {
      pou_type = PartitionOfUnityType::Distance;
    }
    else if (pou_string == "none") {
      pou_type = PartitionOfUnityType::None;
    }
    else {
      DUNE_THROW(Dune::Exception, "Unknown partition of unity type, can either be 'standard' or 'none'");
    }

    init(&A);
    createPOU(ptree.get("pou_shrink", 0));
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void pre(Vec &, Vec &) override {}
  void post(Vec &) override {}

  void apply(Vec &x, const Vec &d) override
  {
    Logger::ScopedLog sl(apply_event);

    if (!solver) {
      Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Schwarz", "init")};
      solver = std::make_unique<Solver>(*Aovlp, 0);
      solver->setOption(UMFPACK_IRSTEP, 0);
    }

    // 1. Copy local values from non-overlapping to overlapping defect
    Logger::get().startEvent(get_defect_event);
    *d_ovlp = 0;
    for (std::size_t i = 0; i < d.size(); ++i) {
      (*d_ovlp)[i] = d[i];
    }

    // int rank{};
    // MPI_Comm_rank(ovlpindices.first->communicator(), &rank);
    // if (rank == 0) {
    //   Dune::printvector(std::cout, *d_ovlp, "defect", "");
    // }

    // 2. Get remaining values from other ranks
    owner_copy_comm.forward<CopyGatherScatter>(*d_ovlp);
    Logger::get().endEvent(get_defect_event);

    // 3. Solve using the overlapping subdomain matrix
    Logger::get().startEvent(subdomain_solve_event);
    Dune::InverseOperatorResult res;
    *x_ovlp = 0.0;
    solver->apply(reinterpret_cast<double *>(x_ovlp->data()), reinterpret_cast<double *>(d_ovlp->data()));
    Logger::get().endEvent(subdomain_solve_event);

    // 4. Make the solution consistent according to the type of the Schwarz method
    Logger::get().startEvent(add_solution_event);
    if (type == SchwarzType::Standard) {
      all_all_comm.forward<AddGatherScatter>(*x_ovlp);
    }
    else if (type == SchwarzType::Restricted) {
      for (std::size_t i = 0; i < pou->N(); ++i) {
        (*x_ovlp)[i] *= (*pou)[i];
      }
      all_all_comm.forward<AddGatherScatter>(*x_ovlp);
    }

    // 4. Restrict the solution to the non-overlapping subdomain
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = (*x_ovlp)[i];
    }
    Logger::get().endEvent(add_solution_event);
  }

  Solver &getSolver() { return *solver; }
  std::shared_ptr<Mat> getOverlappingMat() const { return Aovlp; }
  std::shared_ptr<Vec> getPartitionOfUnity() const { return pou; }

  // RemoteParallelIndices<RemoteIndices> getOverlappingIndices() const { return ovlpindices; }

private:
  void init(const Mat *A = nullptr)
  {
    apply_event = Logger::get().registerEvent("Schwarz", "apply");
    subdomain_solve_event = Logger::get().registerEvent("Schwarz", "local solve");
    get_defect_event = Logger::get().registerEvent("Schwarz", "get defect");
    add_solution_event = Logger::get().registerEvent("Schwarz", "add solution");
    auto *init_event = Logger::get().registerEvent("Schwarz", "init");

    Logger::ScopedLog sl(init_event);

    all_all_interface.build(ext_indices.get_remote_indices(), allAttributes, allAttributes);
    all_all_comm.build<Vec>(all_all_interface);

    owner_copy_interface.build(ext_indices.get_remote_indices(), ownerAttribute, copyAttribute);
    owner_copy_comm.build<Vec>(owner_copy_interface);

    if (A) {
      Aovlp = std::make_shared<Mat>(ext_indices.create_overlapping_matrix(*A));
    }
    assert(Aovlp);
    if (not factorise_at_first_iteration) {
      solver = std::make_unique<Solver>(*Aovlp, 0);
      solver->setOption(UMFPACK_IRSTEP, 0);
    }
    else {
      solver = nullptr;
    }

    d_ovlp = std::make_unique<Vec>(Aovlp->N());
    x_ovlp = std::make_unique<Vec>(Aovlp->N());
  }

  void createPOU(int shrink)
  {
    int rank{};
    MPI_Comm_rank(ext_indices.get_remote_indices().communicator(), &rank);

    const auto overlap = ext_indices.get_overlap();
    const auto &A = *Aovlp;

    std::vector<bool> boundary_mask;

    if (pou_type != PartitionOfUnityType::None) {
      IdentifyBoundaryDataHandle ibdh(A, ext_indices.get_parallel_index_set());
      Dune::VariableSizeCommunicator<> var_comm(all_all_interface);
      var_comm.forward(ibdh);
      boundary_mask = ibdh.get_boundary_mask();
    }

    switch (pou_type) {
    case PartitionOfUnityType::Standard: {
      pou = std::make_shared<Vec>(x_ovlp->N());
      *pou = 1.0;
      for (std::size_t i = 0; i < pou->N(); ++i) {
        if (boundary_mask[i]) {
          (*pou)[i] = 0.0;
        }
      }

      all_all_comm.forward<AddGatherScatter>(*pou);

      for (std::size_t i = 0; i < pou->N(); ++i) {
        if (!boundary_mask[i]) {
          (*pou)[i] = 1. / (*pou)[i];
        }
        else {
          (*pou)[i] = 0.0;
        }
      }
    } break;

    case PartitionOfUnityType::Distance: {
      std::vector<int> boundary_dst(ext_indices.size(), std::numeric_limits<int>::max() - 1);
      for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
        if (boundary_mask[i]) {
          boundary_dst[i] = 0;
        }
      }

      // TODO: I don't understand why something as big as 4*overlap is necessary here, shouldn't 2*overlap suffice??
      for (int round = 0; round <= 4 * overlap; ++round) {
        for (std::size_t i = 0; i < boundary_dst.size(); ++i) {
          for (auto cIt = A[i].begin(); cIt != A[i].end(); ++cIt) {
            boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cIt.index()] + 1); // Increase distance from boundary by one
          }
        }
      }

      pou = std::make_shared<Vec>(x_ovlp->N());
      *pou = 1;
      for (std::size_t i = 0; i < pou->N(); ++i) {
        if (boundary_dst[i] <= 4 * overlap) {
          if (boundary_dst[i] <= shrink) {
            (*pou)[i] = 0;
          }
          else {
            (*pou)[i] = boundary_dst[i] - shrink;
          }
        }
      }

      auto pou_sum = *pou;
      all_all_comm.forward<AddGatherScatter>(pou_sum);

      for (std::size_t i = 0; i < pou->N(); ++i) {
        if (!boundary_mask[i]) {
          (*pou)[i] /= pou_sum[i];
        }
        else {
          (*pou)[i] = 0.0;
        }
      }
    } break;

    case PartitionOfUnityType::None: {
      pou = std::make_shared<Vec>(*x_ovlp);
      *pou = 1.0;

      for (const auto &idx : ext_indices.get_parallel_index_set()) {
        if (idx.local().attribute() != Attribute::owner) {
          (*pou)[idx.local()] = 0;
        }
      }
    } break;
    }
  }

  std::shared_ptr<Mat> Aovlp;

  std::unique_ptr<Solver> solver;
  std::unique_ptr<Vec> d_ovlp; // Defect on overlapping index set
  std::unique_ptr<Vec> x_ovlp; // Solution on overlapping index set

  std::shared_ptr<Vec> pou; // partition of unity (might be null)

  Dune::Interface all_all_interface;
  Dune::BufferedCommunicator all_all_comm;

  Dune::Interface owner_copy_interface;
  Dune::BufferedCommunicator owner_copy_comm;

  // RemoteParallelIndices<RemoteIndices> ovlpindices;
  const ExtendedRemoteIndices &ext_indices;

  SchwarzType type = SchwarzType::Restricted;
  PartitionOfUnityType pou_type;

  bool factorise_at_first_iteration;

  Logger::Event *apply_event{nullptr};
  Logger::Event *subdomain_solve_event{nullptr};
  Logger::Event *get_defect_event{nullptr};
  Logger::Event *add_solution_event{nullptr};
};
