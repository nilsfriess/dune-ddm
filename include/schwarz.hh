#pragma once

#include <dune/common/exceptions.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/istl/io.hh>
#include <dune/istl/preconditioner.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/umfpack.hh>

#include <cstdint>
#include <memory>
#include <mpi.h>

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

    A.usmv(alpha, x, y);
    communicator->forward<AddGatherScatter>(y);
  }

  const Mat &getmat() const override { return A; }
};

enum class SchwarzType : std::uint8_t { Standard, Restricted };

enum class PartitionOfUnityType : std::uint8_t { None, Standard };

template <class Vec, class Mat, class RemoteIndices, class Solver = Dune::UMFPack<Mat>>
class SchwarzPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  AttributeSet ownerAttribute{Attribute::owner};
  AttributeSet copyAttribute{Attribute::copy};

  using ParallelIndexSet = typename RemoteIndices::ParallelIndexSet;

public:
  SchwarzPreconditioner(const Mat &A, const RemoteIndices &remoteindices, int overlap, SchwarzType type = SchwarzType::Restricted, PartitionOfUnityType pou_type = PartitionOfUnityType::None)
      : N(A.N()), type(type), pou_type(pou_type)
  {
    spdlog::info("Setting up Schwarz preconditioner in {} mode", type == SchwarzType::Standard ? "standard" : "restricted");

    init(A, overlap, remoteindices);
    createPOU(*Aovlp);
  }

  SchwarzPreconditioner(const Mat &A, const RemoteIndices &remoteindices, const Dune::ParameterTree &ptree) : N(A.N())
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
    else if (pou_string == "none") {
      pou_type = PartitionOfUnityType::None;
    }
    else {
      DUNE_THROW(Dune::Exception, "Unknown partition of unity type, can either be 'standard' or 'none'");
    }

    init(A, ptree.get("overlap", 1), remoteindices);
    createPOU(*Aovlp);
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void pre(Vec &, Vec &) override {}
  void post(Vec &) override {}

  void apply(Vec &x, const Vec &d) override
  {
    Logger::ScopedLog sl(apply_event);

    // 1. Copy local values from non-overlapping to overlapping defect
    for (std::size_t i = 0; i < N; ++i) {
      (*d_ovlp)[i] = d[i];
    }

    // 2. Get remaining values from other ranks
    cvdh.setVec(*d_ovlp);
    owner_copy_comm->forward(cvdh);

    int rank = 0;
    MPI_Comm comm = ovlpindices.first->communicator();
    MPI_Comm_rank(comm, &rank);
    // if (rank == 0)
    //   Dune::printvector(std::cout, (*d_ovlp), "Residual", "");

    // 3. Solve using the overlapping subdomain matrix
    Dune::InverseOperatorResult res;
    *x_ovlp = 0.0;
    solver->apply(*x_ovlp, *d_ovlp, res);

    // 4. Make the solution consistent according to the type of the Schwarz method
    if (type == SchwarzType::Standard) {
      avdh.setVec(*x_ovlp);
      all_all_comm->forward(avdh);
    }
    else if (type == SchwarzType::Restricted) {
      if (pou_type == PartitionOfUnityType::Standard) {
        for (int i = 0; i < pou->N(); ++i) {
          (*x_ovlp)[i] *= (*pou)[i];
        }
        avdh.setVec(*x_ovlp);
        all_all_comm->forward(avdh);
      }
      else {
        cvdh.setVec(*x_ovlp);
        owner_copy_comm->forward(cvdh);
      }
    }

    // 4. Restrict the solution to the non-overlapping subdomain
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = (*x_ovlp)[i];
    }
  }

  std::shared_ptr<Mat> getOverlappingMat() const { return Aovlp; }
  std::shared_ptr<Vec> getPartitionOfUnity() const { return pou; }

  RemoteParallelIndices<RemoteIndices> getOverlappingIndices() const { return ovlpindices; }

private:
  void init(const Mat &A, int overlap, const RemoteIndices &remoteindices)
  {
    apply_event = Logger::get().registerEvent("Schwarz", "apply");
    auto *init_event = Logger::get().registerEvent("Schwarz", "init");

    Logger::ScopedLog sl(init_event);

    spdlog::info("Extending overlap by {} in SchwarzPreconitioner", overlap);
    ovlpindices = extendOverlap(remoteindices, A, overlap);

    int rank = 0;
    MPI_Comm comm = remoteindices.communicator();
    MPI_Comm_rank(comm, &rank);

    all_all_interface.build(*ovlpindices.first, allAttributes, allAttributes);
    all_all_comm = std::make_unique<Dune::VariableSizeCommunicator<>>(all_all_interface);

    owner_copy_interface.build(*ovlpindices.first, ownerAttribute, copyAttribute);
    owner_copy_comm = std::make_unique<Dune::VariableSizeCommunicator<>>(owner_copy_interface);

    Aovlp = std::make_shared<Mat>(createOverlappingMatrix(A, *ovlpindices.first));
    solver = std::make_unique<Solver>(*Aovlp, 0);

    d_ovlp = std::make_unique<Vec>(Aovlp->N());
    x_ovlp = std::make_unique<Vec>(Aovlp->N());
  }

  void createPOU(const Mat &A)
  {
    int rank{};
    MPI_Comm_rank(ovlpindices.first->communicator(), &rank);
    IdentifyBoundaryDataHandle ibdh(A, *ovlpindices.second, rank);
    all_all_comm->forward(ibdh);
    auto boundaryMask = ibdh.getBoundaryMask();

    switch (pou_type) {
    case PartitionOfUnityType::Standard: {
      pou = std::make_shared<Vec>(x_ovlp->N());
      *pou = 1.0;
      for (std::size_t i = 0; i < pou->N(); ++i) {
        if (boundaryMask[i]) {
          (*pou)[i] = 0.0;
        }
      }

      avdh.setVec(*pou);
      all_all_comm->forward(avdh);

      for (int i = 0; i < pou->N(); ++i) {
        if (!boundaryMask[i]) {
          (*pou)[i] = 1. / (*pou)[i];
        } else {
          (*pou)[i] = 0.0;
        }
      }
    } break;

    case PartitionOfUnityType::None: {
      pou = std::make_shared<Vec>(*x_ovlp);
      *pou = 1.0;

      for (const auto &idx : *ovlpindices.second) {
        if (idx.local().attribute() != Attribute::owner) {
          (*pou)[idx.local()] = 0;
        }
      }
    } break;
    }
  }

  std::size_t N;
  std::shared_ptr<Mat> Aovlp;

  std::unique_ptr<Solver> solver;
  std::unique_ptr<Vec> d_ovlp; // Defect on overlapping index set
  std::unique_ptr<Vec> x_ovlp; // Solution on overlapping index set

  std::shared_ptr<Vec> pou; // partition of unity (might be null)

  Dune::Interface all_all_interface;
  std::unique_ptr<Dune::VariableSizeCommunicator<>> all_all_comm;

  Dune::Interface owner_copy_interface;
  std::unique_ptr<Dune::VariableSizeCommunicator<>> owner_copy_comm;

  CopyVectorDataHandle<Vec> cvdh;
  AddVectorDataHandle<Vec> avdh;

  RemoteParallelIndices<RemoteIndices> ovlpindices;

  SchwarzType type = SchwarzType::Restricted;
  PartitionOfUnityType pou_type;

  Logger::Event *apply_event{};
};
