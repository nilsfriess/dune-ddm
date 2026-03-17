#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dune/ddm/coarsespaces/coarse_spaces.hh"
#include "dune/ddm/combined_preconditioner.hh"
#include "dune/ddm/galerkin_preconditioner.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/nonoverlapping_operator.hh"
#include "dune/ddm/overlap_extension.hh"
#include "dune/ddm/parallel_matrix_io.hh"
#include "dune/ddm/pou.hh"
#include "dune/ddm/schwarz.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/novlpschwarz.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/solverfactory.hh>
#include <iostream>
#include <type_traits>

/// Symmetrise a matrix in-place: returns (A + A^T) / 2.
///
/// The result has the union of the sparsity patterns of A and A^T.  For an
/// already-symmetric matrix the operation is a no-op up to floating-point
/// arithmetic.
template <class Matrix>
Matrix symmetrise_matrix(const Matrix& A)
{
  using size_type = typename Matrix::size_type;
  const size_type N = A.N();

  // Union of sparsity patterns of A and A^T
  std::vector<std::set<size_type>> symRows(N);
  for (size_type i = 0; i < N; ++i)
    for (auto ci = A[i].begin(); ci != A[i].end(); ++ci) {
      symRows[i].insert(ci.index());
      symRows[ci.index()].insert(i);
    }

  size_type maxRowSize = 0;
  for (size_type i = 0; i < N; ++i) maxRowSize = std::max(maxRowSize, static_cast<size_type>(symRows[i].size()));

  Matrix S(N, N, maxRowSize, 0.2, Matrix::implicit);
  for (size_type i = 0; i < N; ++i)
    for (size_type j : symRows[i]) S.entry(i, j) = 0;
  S.compress();

  // S[i][j] = (A[i][j] + A[j][i]) / 2:
  // iterate A once; each entry (i,j) contributes 0.5*A[i][j] to both S[i][j]
  // and S[j][i] (the transpose contribution).
  for (size_type i = 0; i < N; ++i)
    for (auto ci = A[i].begin(); ci != A[i].end(); ++ci) {
      const size_type j = ci.index();
      S[i][j] += 0.5 * (*ci);
      S[j][i] += 0.5 * (*ci);
    }

  return S;
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

    if (!ptree.hasKey("input_matrix")) {
      if (rank == 0) std::cout << "Usage: " << argv[0] << " -input_matrix <matrix_file.mtx> \n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    auto filename = ptree.get<std::string>("input_matrix");

    // Load matrix and partition with METIS (owners + boundary copies)
    auto par_matrix_data = readMatrixMarketParallel(helper, filename);
    const auto& localA = par_matrix_data.matrix;
    const auto& novlp_comm = par_matrix_data.communication;
    using Comm = std::decay_t<decltype(*novlp_comm)>;

    using Op = NonOverlappingOperator<Matrix, Vector, Vector, Comm>;
    auto op = std::make_shared<Op>(localA, novlp_comm);

    // Print size of local matrix and size of index set in communicator
    logger::debug_all("Local matrix size: {}x{}, local index set size: {}", localA->N(), localA->M(), novlp_comm->remoteIndices().sourceIndexSet().size());

    // Create overlapping index set and matrix
    int overlap = ptree.get("overlap", 1);
    auto [ovlp_comm, A_ovlp, boundary] = create_overlapping_matrix(*novlp_comm, *localA, overlap);
    logger::info_all("Overlapping matrix size {}x{}", A_ovlp->N(), A_ovlp->M());

    // Symmetrise the overlapping matrix: (A + A^T) / 2.
    // All preconditioner components and the approximate Neumann matrix use
    // this symmetrised version; the operator (op) keeps the original localA.
    auto A_sym = std::make_shared<Matrix>(symmetrise_matrix(*A_ovlp));

    auto pou = std::make_shared<PartitionOfUnity>(*A_sym, *ovlp_comm, ptree, overlap);
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
      auto A_neumann = make_algebraic_neumann(*ovlp_comm, *A_sym);
      auto coarse_basis = build_geneo_coarse_space(A_neumann, *A_sym, *pou, ptree, "coarsespace");
      if (!coarse_basis.empty()) {
        auto coarse_level = std::make_shared<GalerkinPreconditioner<Vector, Comm>>(*A_ovlp, coarse_basis, ovlp_comm, ptree);
        prec->add(coarse_level);
      }
    }
    else if (coarse_type == "algebraic_msgfem") {
      auto A_neumann = make_algebraic_neumann(*ovlp_comm, *A_sym);
      std::vector<bool> dirichlet_mask(A_sym->N(), false); // No known Dirichlet DOFs in the algebraic setting
      auto coarse_basis = build_msgfem_coarse_space(A_neumann, *pou, boundary, ptree, "coarsespace", A_ovlp.get(), dirichlet_mask, fine_level->get_solver().get());
      if (!coarse_basis.empty()) {
        auto coarse_level = std::make_shared<GalerkinPreconditioner<Vector, Comm>>(*A_ovlp, coarse_basis, ovlp_comm, ptree);
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

    // Load or create right-hand side
    Vector x(localA->N());
    Vector b(localA->N());
    if (ptree.hasKey("input_rhs")) {
      auto rhs_filename = ptree.get<std::string>("input_rhs");
      b = readVectorParallel(*novlp_comm, rhs_filename);
    }
    else {
      b = 1.0;
    }

    {
      Logger::ScopedLog sl(Logger::get().registerOrGetEvent("Solver", "apply"));

      Dune::InverseOperatorResult res;
      solver->apply(x, b, res);

      auto sp = Dune::NonOverlappingScalarProduct<Vector, Comm>(novlp_comm);
      double norm = sp.norm(x);
      logger::info("Solution norm: {}", norm);
    }

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
