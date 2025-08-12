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
#include <memory>
#include <mpi.h>
#include <umfpack.h>

#include "helpers.hh"
#include "logger.hh"
#include "pou.hh"
#include "spdlog/spdlog.h"
#include "strumpack.hh"

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

#if DUNE_DDM_HAVE_STRUMPACK
template <class Vec, class Mat, class ExtendedRemoteIndices, class Solver = Dune::STRUMPACK<Mat, Vec>>
#else
template <class Vec, class Mat, class ExtendedRemoteIndices, class Solver = Dune::UMFPack<Mat>>
#endif
class SchwarzPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  AttributeSet ownerAttribute{Attribute::owner};
  AttributeSet copyAttribute{Attribute::copy};

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
  SchwarzPreconditioner(std::shared_ptr<Mat> Aovlp, const ExtendedRemoteIndices &ext_indices, std::shared_ptr<PartitionOfUnity> pou, SchwarzType type = SchwarzType::Restricted,
                        bool factorise_at_first_iteration = false)
      : Aovlp(std::move(Aovlp)), ext_indices(ext_indices), type(type), pou(std::move(pou)), factorise_at_first_iteration(factorise_at_first_iteration)
  {
    init();
  }

  SchwarzPreconditioner(std::shared_ptr<Mat> Aovlp, const ExtendedRemoteIndices &ext_indices, std::shared_ptr<PartitionOfUnity> pou, const Dune::ParameterTree &ptree, const std::string &subtree_prefix = "schwarz")
      : Aovlp(std::move(Aovlp)), ext_indices(ext_indices), pou(std::move(pou))
  {
    const auto &subtree = ptree.sub(subtree_prefix);
    factorise_at_first_iteration = subtree.get("factorise_at_first_iteration", false);
    auto type_string = subtree.get("type", "restricted");
    if (type_string == "restricted") {
      type = SchwarzType::Restricted;
    }
    else if (type_string == "standard") {
      type = SchwarzType::Standard;
    }
    else {
      DUNE_THROW(Dune::NotImplemented, "Unknown Schwarz type '" + type_string + "'");
    }

    init();
  }

  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::nonoverlapping; }

  void pre(Vec &, Vec &) override {}
  void post(Vec &) override {}

  void apply(Vec &x, const Vec &d) override
  {
    Logger::ScopedLog sl(apply_event);

    if (!solver) {
      Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Schwarz", "init")};
      solver = std::make_unique<Solver>(*Aovlp);
#ifndef DUNE_DDM_HAVE_STRUMPACK
      solver->setOption(UMFPACK_IRSTEP, 0);
#endif
    }

    // 1. Copy local values from non-overlapping to overlapping defect
    Logger::get().startEvent(get_defect_event);
    *d_ovlp = 0;
    for (std::size_t i = 0; i < d.size(); ++i) {
      (*d_ovlp)[i] = d[i];
    }

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
      if (pou) {
        for (std::size_t i = 0; i < pou->size(); ++i) {
          (*x_ovlp)[i] *= (*pou)[i];
        }
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
  std::shared_ptr<PartitionOfUnity> getPartitionOfUnity() const { return pou; }

private:
  void init()
  {
    spdlog::info("Setting up Schwarz preconditioner in {} mode", type == SchwarzType::Standard ? "standard" : "restricted");

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

    if (not factorise_at_first_iteration) {
      solver = std::make_unique<Solver>(*Aovlp);
#ifndef DUNE_DDM_HAVE_STRUMPACK
      solver->setOption(UMFPACK_IRSTEP, 0);
#endif
    }
    else {
      solver = nullptr;
    }

    d_ovlp = std::make_unique<Vec>(Aovlp->N());
    x_ovlp = std::make_unique<Vec>(Aovlp->N());
  }

  std::shared_ptr<Mat> Aovlp;

  std::unique_ptr<Solver> solver;
  std::unique_ptr<Vec> d_ovlp; // Defect on overlapping index set
  std::unique_ptr<Vec> x_ovlp; // Solution on overlapping index set

  std::shared_ptr<PartitionOfUnity> pou{nullptr}; // partition of unity (might be null)

  Dune::Interface all_all_interface;
  Dune::BufferedCommunicator all_all_comm;

  Dune::Interface owner_copy_interface;
  Dune::BufferedCommunicator owner_copy_comm;

  const ExtendedRemoteIndices &ext_indices; // TODO: Make this a shared_ptr

  SchwarzType type;

  bool factorise_at_first_iteration;

  Logger::Event *apply_event{nullptr};
  Logger::Event *subdomain_solve_event{nullptr};
  Logger::Event *get_defect_event{nullptr};
  Logger::Event *add_solution_event{nullptr};
};
