#include "dune/ddm/galerkin_preconditioner.hh"
#define DUNE_ISTL_WITH_CHECKING 1

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dune/ddm/combined_preconditioner.hh"
#include "dune/ddm/helpers.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/overlap_extension.hh"
#include "dune/ddm/pou.hh"
#include "dune/ddm/schwarz.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/novlpschwarz.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solvers.hh>
#include <iostream>

void print_usage(char** argv)
{
  std::cout << "Usage: " << argv[0] << " input_matrix\n"
            << "  input_matrix: The matrix to load (in matrix market format)\n";
}

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  const auto rank = helper.rank();
  setup_loggers(rank, argc, argv);

  if (argc < 2) {
    if (rank == 0) print_usage(argv);
    return 1;
  }

  Dune::ParameterTree ptree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readOptions(argc, argv, ptree);

  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;
  Matrix A;

  if (rank == 0) {
    std::string filename(argv[1]);
    Dune::loadMatrixMarket(A, filename);
    std::cout << "Loaded matrix from file " << filename << "\n"
              << "Matrix size: " << A.N() << "x" << A.M() << "\n";
  }

  auto [novlp_comm, localA, active] = distributeMatrixFrom0<Matrix, std::size_t>(A, MPI_COMM_WORLD, helper.size());
  using Comm = typename decltype(novlp_comm)::element_type;
  assert(active && "All ranks should get some part of the matrix");

  // Print size of local matrix and size of index set in communicator
  logger::debug_all("Local matrix size: {}x{}, local index set size: {}", localA->N(), localA->M(), novlp_comm->remoteIndices().sourceIndexSet().size());

  auto op = std::make_shared<Dune::NonoverlappingSchwarzOperator<Matrix, Vector, Vector, Comm>>(localA, *novlp_comm);
  Dune::NonoverlappingSchwarzScalarProduct<Vector, Comm> sp(novlp_comm);

  // Create overlapping index set and matrix
  int overlap = ptree.get("overlap", 1);
  auto [ovlp_comm, boundary] = make_overlapping_communication(*novlp_comm, *localA, overlap);

  typename Comm::AllSet allset;
  Dune::Interface interface;
  interface.build(ovlp_comm->remoteIndices(), allset, allset);
  auto varcomm = std::make_unique<Dune::VariableSizeCommunicator<>>(interface);
  CreateMatrixDataHandle cmdh(*localA, ovlp_comm->indexSet());
  varcomm->forward(cmdh);
  auto A_ovlp = std::make_shared<Matrix>(cmdh.getOverlappingMatrix());
  AddMatrixDataHandle amdh(*localA, *A_ovlp, ovlp_comm->indexSet());
  varcomm->forward(amdh);

  auto pou = std::make_shared<PartitionOfUnity>(*A_ovlp, *ovlp_comm, ptree, overlap);
  auto prec = std::make_shared<CombinedPreconditioner<Vector>>(ptree);
  prec->set_op(op);
  auto fine_level = std::make_shared<SchwarzPreconditioner<Matrix, Vector, Comm>>(A_ovlp, ovlp_comm, pou, ptree);
  prec->add(fine_level);
  if (ptree.get("with_coarse", false)) {
    std::vector<Vector> template_vectors(1);
    template_vectors[0] = pou->vector();
    auto coarse_level = std::make_shared<GalerkinPreconditioner<Vector, Comm>>(*A_ovlp, template_vectors, ovlp_comm, ptree);
    prec->add(coarse_level);
  }

  // Set up solver
  // RestartedGMResSolver (const LinearOperator<X,Y>& op, Preconditioner<X,Y>& prec, scalar_real_type reduction, int restart, int maxit, int verbose) :
  Dune::RestartedGMResSolver<Vector> solver(*op, sp, *prec, 1e-6, 100, 1000, rank == 0 ? 2 : 0);

  // Solve linear system A x = b with random rhs
  Vector x(localA->N());
  Vector b(localA->N());
  b = 1.0;

  Dune::InverseOperatorResult res;
  solver.apply(x, b, res);
  if (rank == 0) std::cout << "Solver finished with " << res.iterations << " iterations and residual " << res.reduction << "\n";
}