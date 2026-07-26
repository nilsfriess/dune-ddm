#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dune/ddm/coarsespaces/coarse_spaces.hh"
#include "dune/ddm/combined_preconditioner.hh"
#include "dune/ddm/galerkin_preconditioner.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/overlap_extension.hh"
#include "dune/ddm/pou.hh"
#include "dune/ddm/schwarz.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/novlpschwarz.hh>
#include <dune/istl/solverfactory.hh>
#include <iostream>

template <class Communication, class Mat>
Mat make_algebraic_neumann(const Communication& comm, const Mat& A, const std::vector<bool>& boundary)
{
  auto [comm_next, A_next, boundary_next] = create_overlapping_matrix(comm, A, 1);
  Mat A_neumann = A;
  for (std::size_t i = 0; i < boundary.size(); ++i) {
    if (boundary[i]) {
      double sum = 0;
      for (auto cit = (*A_next)[i].begin(); cit != (*A_next)[i].end(); ++cit)
        if (cit.index() >= A.N()) sum += std::abs(*cit);
      A_neumann[i][i] -= sum;
    }
  }
  return A_neumann;
}

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  const auto rank = helper.rank();
  setup_loggers(rank, argc, argv);

  try {
    Dune::ParameterTree ptree;
    Dune::ParameterTreeParser ptreeparser;
    ptreeparser.readINITree("matrix_market_test.ini", ptree);
    ptreeparser.readOptions(argc, argv, ptree);

    using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
    using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;
    Matrix A;

    if (rank == 0) {
      if (!ptree.hasKey("input_matrix")) {
        std::cout << "Usage: " << argv[0] << " -input_matrix <matrix_file.mtx> \n";
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      auto filename = ptree.get<std::string>("input_matrix");
      Dune::loadMatrixMarket(A, filename);
      std::cout << "Loaded matrix from file " << filename << "\n"
                << "Matrix size: " << A.N() << "x" << A.M() << "\n";
    }

    auto [novlp_comm, localA, active] = distributeMatrixFrom0<Matrix, std::size_t>(A, MPI_COMM_WORLD, helper.size());
    using Comm = typename decltype(novlp_comm)::element_type;
    assert(active && "All ranks should get some part of the matrix");

    // Print size of local matrix and size of index set in communicator
    logger::debug_all("Local matrix size: {}x{}, local index set size: {}", localA->N(), localA->M(), novlp_comm->remoteIndices().sourceIndexSet().size());

    using Op = Dune::NonoverlappingSchwarzOperator<Matrix, Vector, Vector, Comm>;
    auto op = std::make_shared<Op>(localA, *novlp_comm);

    // Create overlapping index set and matrix
    int overlap = ptree.get("overlap", 1);
    auto [ovlp_comm, A_ovlp, boundary] = create_overlapping_matrix(*novlp_comm, *localA, overlap);

    auto pou = std::make_shared<PartitionOfUnity>(*A_ovlp, *ovlp_comm, ptree, overlap);
    auto prec = std::make_shared<CombinedPreconditioner<Vector>>(ptree);
    prec->set_op(op);
    auto fine_level = std::make_shared<SchwarzPreconditioner<Matrix, Vector, Comm>>(A_ovlp, ovlp_comm, pou, ptree);
    prec->add(fine_level);
    auto coarse_type = ptree.get("coarse_type", "none");
    if (coarse_type == "pou") {
      std::vector<Vector> template_vectors(1);
      template_vectors[0] = pou->vector();
      auto coarse_level = std::make_shared<GalerkinPreconditioner<Vector, Comm>>(*A_ovlp, template_vectors, ovlp_comm, ptree);
      prec->add(coarse_level);
    }
    else if (coarse_type == "algebraic_geneo") {
      auto A_neumann = make_algebraic_neumann(*ovlp_comm, *A_ovlp, boundary);
      auto coarse_space = build_geneo_coarse_space(A_neumann, A_neumann, *pou, ptree, "coarsespace");
      if (!coarse_space.basis.empty()) {
        auto coarse_level = std::make_shared<GalerkinPreconditioner<Vector, Comm>>(*A_ovlp, coarse_space.basis, ovlp_comm, ptree);
        prec->add(coarse_level);
      }
    }
    else if (coarse_type == "algebraic_msgfem") {
      auto A_neumann = make_algebraic_neumann(*ovlp_comm, *A_ovlp, boundary);
      std::vector<bool> dirichlet_mask(A_ovlp->N(), false); // No known Dirichlet DOFs in the algebraic setting
      auto coarse_space = build_msgfem_coarse_space(A_neumann, *pou, boundary, ptree, "coarsespace", A_ovlp.get(), dirichlet_mask, fine_level->get_solver().get());
      if (!coarse_space.basis.empty()) {
        auto coarse_level = std::make_shared<GalerkinPreconditioner<Vector, Comm>>(*A_ovlp, coarse_space.basis, ovlp_comm, ptree);
        prec->add(coarse_level);
      }
    }
    else if (coarse_type != "none") {
      logger::error("Unknown coarse_type '{}', expected 'none', 'pou', 'algebraic_geneo', or 'algebraic_msgfem'", coarse_type);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Set up solver via factory
    Dune::initSolverFactories<Op>();
    auto solver_subtree = ptree.sub("solver");
    solver_subtree["verbose"] = rank == 0 ? solver_subtree["verbose"] : "0";
    auto solver = Dune::getSolverFromFactory(op, solver_subtree, prec);

    // Solve linear system A x = b with random rhs
    Vector x(localA->N());
    Vector b(localA->N());
    b = 1.0;

    Dune::InverseOperatorResult res;
    solver->apply(x, b, res);

    Logger::get().report(MPI_COMM_WORLD);
  }
  catch (const Dune::Exception& e) {
    std::cerr << "[" << rank << "] Dune reported error: " << e << std::endl;
    return 1;
  }
  catch (const std::exception& e) {
    std::cerr << "[" << rank << "] Standard exception: " << e.what() << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << "[" << rank << "] Unknown exception!" << std::endl;
    return 1;
  }
}
