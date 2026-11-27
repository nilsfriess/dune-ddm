#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

/* DG with TwoLevelSchwarzSolver on a grid with ghost cells (ALUGrid).
 *
 * Same problem as convectiondiffusiondg.cc, but set up the way a DG program on ALUGrid has to be:
 *   - the grid function space lives on an AllEntitySet (interior + ghost cells), so the face terms
 *     between two ranks are assembled;
 *   - ghost DoFs are constrained in the test space only (P0ParallelGhostConstraints): their rows are
 *     zero, the columns stay in the matrix;
 *   - the matrix is not additive (matrix_is_additive = false).
 *
 * ALUGrid creates one ghost entity per shared face, so a neighbour cell that touches this rank across
 * two faces appears twice, and both copies get the same global numbers. The program reports
 *   1. how many local DoFs share their global number with another local DoF (duplicate ghost DoFs),
 *   2. how many rows of the overlapping matrix have a zero or missing diagonal,
 *   3. whether the solver converged and the relative residual of the solution,
 * and returns 0 only if there are no zero-diagonal rows and the residual was reduced as requested.
 *
 * Run e.g.: mpirun -np 2 ./convectiondiffusiondg_ghosts -grid.type simplex -grid.cells 16
 */

#include "convectiondiffusiondg.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/twolevel_schwarz.hh"

#include <dune/alugrid/common/structuredgridfactory.hh>
#include <dune/alugrid/dgf.hh>
#include <dune/alugrid/grid.hh>
#if HAVE_DUNE_UGGRID
#include <dune/grid/uggrid.hh>
#endif
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/geometry/type.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/istl/solvercategory.hh>
#include <dune/pdelab/backend/istl.hh>
#include <dune/pdelab/common/partitionviewentityset.hh>
#include <dune/pdelab/constraints/common/constraints.hh>
#include <dune/pdelab/constraints/p0ghost.hh>
#include <dune/pdelab/finiteelementmap/opbfem.hh>
#include <dune/pdelab/finiteelementmap/qkdg.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspaceutilities.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/pdelab/gridoperator/gridoperator.hh>
#include <dune/pdelab/localoperator/convectiondiffusiondg.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>
#include <dune/pdelab/stationary/linearproblem.hh>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>

// Local DoFs whose global number (dune-ddm's make_communication) also belongs to another local DoF.
template <class Communication>
std::size_t count_duplicate_dofs(const Communication& comm)
{
  std::unordered_map<typename Communication::ParallelIndexSet::GlobalIndex, int> copies;
  for (const auto& idx : comm.indexSet()) ++copies[idx.global()];
  std::size_t n = 0;
  for (const auto& [gi, c] : copies)
    if (c > 1) n += c;
  return n;
}

// Builds the overlapping matrix the same way TwoLevelSchwarzSolver::apply does on its first call and
// counts the rows with a zero or missing diagonal entry (these make the subdomain solve singular),
// separately for the rank's own rows (first) and for the rows added by the overlap extension (second).
// Duplicate ghost DoFs show up in the first count, rows that were never sent in the second.
template <class NativeMat, class Communication>
std::pair<std::size_t, std::size_t> count_zero_diagonal_rows(NativeMat A, const Communication& novlp_comm, int overlap, bool matrix_is_additive)
{
  if (!matrix_is_additive) ::make_additive(A, novlp_comm);
  auto ovlp_comm = make_overlapping_communication(novlp_comm, A, overlap).first;

  typename Communication::AllSet allset;
  Dune::Interface interface;
  interface.build(ovlp_comm->remoteIndices(), allset, allset);
  Dune::VariableSizeCommunicator<> varcomm(interface);

  CreateMatrixDataHandle cmdh(A, ovlp_comm->indexSet());
  varcomm.forward(cmdh);
  NativeMat A_ovlp = cmdh.getOverlappingMatrix();
  AddMatrixDataHandle amdh(A, A_ovlp, ovlp_comm->indexSet());
  varcomm.forward(amdh);

  std::size_t zero_local = 0, zero_overlap = 0;
  for (std::size_t i = 0; i < A_ovlp.N(); ++i) {
    auto it = A_ovlp[i].find(i);
    if (it == A_ovlp[i].end() || std::abs((*it)[0][0]) == 0.0) ++(i < A.N() ? zero_local : zero_overlap);
  }
  return {zero_local, zero_overlap};
}

template <class ES, class Grid, class FEM>
int run(Grid& grid, const FEM& fem, const Dune::ParameterTree& ptree)
{
  using Dune::PDELab::Backend::native;

  using GV = typename Grid::LeafGridView;
  GV gv = grid.leafGridView();
  const auto& cc_comm = gv.comm();
  const int rank = cc_comm.rank();

  // AllEntitySet: all cells including the ghosts, so the faces between ranks are assembled.
  // OverlappingEntitySet (space.entityset = overlapping) leaves the ghosts out: for comparison only (problem 1, walls).
  ES es(gv);

  using CON = Dune::PDELab::P0ParallelGhostConstraints;
  using VBE = Dune::PDELab::ISTL::VectorBackend<>;
  using GFS = Dune::PDELab::GridFunctionSpace<ES, FEM, CON, VBE>;
  GFS gfs(es, fem);
  using X = Dune::PDELab::Backend::Vector<GFS, double>;

  using Problem = ConvectionDiffusionDGProblem<GV, double>;
  Problem problem;
  using BCType = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<Problem>;
  BCType bctype(gv, problem);

  // test space: ghost DoFs constrained (zero rows); trial space: no constraints (columns kept)
  using CC = typename GFS::template ConstraintsContainer<double>::Type;
  CC cc;
  Dune::PDELab::constraints(bctype, gfs, cc);
  CC ccu;
  // space.trial_constraints = true: ghost constraints on the trial space too, as in convectiondiffusiondg.cc
  // (the PDELab boilerplate uses one constraints container for both spaces). For comparison only.
  if (ptree.get("space.trial_constraints", false)) Dune::PDELab::constraints(bctype, gfs, ccu);

  using LOP = Dune::PDELab::ConvectionDiffusionDG<Problem, FEM>;
  LOP lop(problem);
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  MBE mbe(27);
  using GO = Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, double, double, double, CC, CC>;
  GO go(gfs, ccu, gfs, cc, lop, mbe);

  const bool matrix_is_additive = ptree.get("parallel.matrix_is_additive", false);
  const int overlap = ptree.get("twolevelschwarz.overlap", 1);

  // 1. grid and duplicate ghost DoFs
  std::size_t interior = 0, ghosts = 0;
  for (const auto& e : elements(gv, Dune::Partitions::interior)) { (void)e; ++interior; }
  for (const auto& e : elements(gv, Dune::Partitions::ghost)) { (void)e; ++ghosts; }
  // faces between an interior and a ghost cell; more faces than ghost cells means some ghost cell
  // touches this rank across two or more faces (the case that gives duplicates on ALUGrid)
  std::size_t shared_faces = 0;
  for (const auto& e : elements(gv, Dune::Partitions::interior))
    for (const auto& is : intersections(gv, e))
      if (is.neighbor() && is.outside().partitionType() == Dune::GhostEntity) ++shared_faces;
  auto full_comm = make_communication(gfs);
  const std::size_t dup = count_duplicate_dofs(*full_comm);
  std::cout << "[rank " << rank << "] interior cells " << interior << ", ghost cells " << ghosts << ", interior-ghost faces " << shared_faces << ", DoFs " << gfs.size()
            << ", duplicate ghost DoFs " << dup << std::endl;

  // 2. rows of the overlapping matrix with a zero or missing diagonal
  X x(gfs, 0.0);
  typename GO::Jacobian J(go);
  J = 0.0;
  go.jacobian(x, J);
  {
    // non-zero entries of this rank's Jacobian in the columns of ghost DoFs (the coupling to the other ranks)
    std::vector<char> ghost_dof(gfs.size(), 0);
    X marker(gfs, 0.0);
    Dune::PDELab::set_constrained_dofs(cc, 1.0, marker);
    std::size_t ghost_cols = 0;
    for (auto row = native(J).begin(); row != native(J).end(); ++row)
      for (auto it = row->begin(); it != row->end(); ++it)
        if (native(marker)[it.index()][0] == 1.0 && std::abs((*it)[0][0]) > 0.0 && it.index() != row.index()) ++ghost_cols;
    std::cout << "[rank " << rank << "] non-zero Jacobian entries in ghost columns (off-diagonal): " << ghost_cols << std::endl;
  }
  const auto [zero_local, zero_overlap] = count_zero_diagonal_rows(native(J), *full_comm, overlap, matrix_is_additive);

  // 3. solve with dune-ddm's two-level Schwarz solver
  using LS = TwoLevelSchwarzSolver<typename GO::Jacobian, X>;
  LS ls(gfs, ccu, ptree, "twolevelschwarz", matrix_is_additive);
  const double reduction = ptree.get("twolevelschwarz.solver.reduction", 1e-8);

  X r0(gfs, 0.0);
  go.residual(x, r0);
  const double r0norm = ls.norm(r0);

  bool converged = false;
  int iterations = -1;
  try {
    Dune::PDELab::StationaryLinearProblemSolver<GO, LS, X> slp(go, ls, x, reduction);
    slp.apply();
    converged = ls.result().converged;
    iterations = ls.result().iterations;
  }
  catch (const Dune::Exception& e) {
    std::cout << "[rank " << rank << "] solve failed: " << e.what() << std::endl;
  }
  converged = cc_comm.min(static_cast<int>(converged)) == 1;

  X r(gfs, 0.0);
  go.residual(x, r);
  const double rel = ls.norm(r) / r0norm;

  // integral and L2 norm of the solution over the interior cells: compare with the 1-rank run (walls change them)
  double integral = 0.0, l2sq = 0.0;
  {
    Dune::PDELab::DiscreteGridFunction<GFS, X> dgf(gfs, x);
    for (const auto& e : elements(gv, Dune::Partitions::interior)) {
      const auto geo = e.geometry();
      for (const auto& qp : Dune::QuadratureRules<double, 2>::rule(geo.type(), 4)) {
        Dune::FieldVector<double, 1> u;
        dgf.evaluate(e, qp.position(), u);
        const double w = qp.weight() * geo.integrationElement(qp.position());
        integral += w * u[0];
        l2sq += w * u[0] * u[0];
      }
    }
  }
  integral = cc_comm.sum(integral);
  const double l2 = std::sqrt(cc_comm.sum(l2sq));

  const std::size_t dup_sum = cc_comm.sum(dup), dup_max = cc_comm.max(dup);
  const std::size_t zl_sum = cc_comm.sum(zero_local), zo_sum = cc_comm.sum(zero_overlap);
  const bool ok = zl_sum + zo_sum == 0 && converged && std::isfinite(rel) && rel <= 10 * reduction;
  if (rank == 0) {
    std::cout << "\n=== convectiondiffusiondg_ghosts on " << cc_comm.size() << " ranks ===\n"
              << "duplicate ghost DoFs:           " << dup_sum << " (max " << dup_max << " on one rank)\n"
              << "zero-diagonal rows of A_ovlp:   " << zl_sum + zo_sum << " (overlap " << overlap << "): " << zl_sum << " own rows, " << zo_sum
              << " overlap rows\n"
              << "solver converged:               " << (converged ? "yes" : "no") << ", iterations " << iterations << "\n"
              << "relative residual |b - Ax|/|b|: " << rel << " (required <= " << 10 * reduction << ")\n"
              << "solution integral, L2 norm:     " << integral << ", " << l2 << "\n"
              << "result:                         " << (ok ? "PASSED" : "FAILED") << std::endl;
  }
  return ok ? 0 : 1;
}

template <class Grid, class FEM>
int run_es(Grid& grid, const FEM& fem, const Dune::ParameterTree& ptree)
{
  using GV = typename Grid::LeafGridView;
  const std::string es = ptree.get("space.entityset", std::string("all"));
  if (es == "overlapping") return run<Dune::PDELab::OverlappingEntitySet<GV>>(grid, fem, ptree);
  return run<Dune::PDELab::AllEntitySet<GV>>(grid, fem, ptree);
}

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  setup_loggers(helper.rank(), argc, argv);

  Dune::ParameterTree ptree;
  Dune::ParameterTreeParser ptreeparser;
  ptreeparser.readINITree("convectiondiffusiondg_ghosts.ini", ptree);
  ptreeparser.readOptions(argc, argv, ptree);

  const std::string type = ptree.get("grid.type", std::string("simplex"));
  const unsigned int cells = ptree.get("grid.cells", 16);
  const int refine = ptree.get("grid.refine", 0);
  const std::string file = ptree.get("grid.file", std::string("")); // gmsh file, simplex and ug-simplex only

  try {
    if (type == "simplex") {
      using Grid = Dune::ALUGrid<2, 2, Dune::simplex, Dune::nonconforming>;
      // grid.file: an unstructured triangle mesh of the unit square (gmsh), otherwise a structured one
      auto grid = file.empty() ? Dune::StructuredGridFactory<Grid>::createSimplexGrid({0., 0.}, {1., 1.}, {cells, cells})
                               : Dune::GmshReader<Grid>::read(file, false, false);
      grid->globalRefine(refine);
      grid->loadBalance();
      using FEM = Dune::PDELab::OPBLocalFiniteElementMap<double, double, 1, 2, Dune::GeometryType::simplex>;
      FEM fem;
      return run_es(*grid, fem, ptree);
    }
    if (type == "cube") {
      using Grid = Dune::ALUGrid<2, 2, Dune::cube, Dune::nonconforming>;
      auto grid = Dune::StructuredGridFactory<Grid>::createCubeGrid({0., 0.}, {1., 1.}, {cells, cells});
      grid->globalRefine(refine);
      grid->loadBalance();
      using FEM = Dune::PDELab::QkDGLocalFiniteElementMap<double, double, 1, 2>;
      FEM fem;
      return run_es(*grid, fem, ptree);
    }
#if HAVE_DUNE_UGGRID
    // UGGrid for comparison: one ghost cell per neighbour cell, not one per shared face
    if (type == "ug-simplex" || type == "ug-cube") {
      using Grid = Dune::UGGrid<2>;
      auto grid = type == "ug-simplex" ? (file.empty() ? Dune::StructuredGridFactory<Grid>::createSimplexGrid({0., 0.}, {1., 1.}, {cells, cells})
                                                       : Dune::GmshReader<Grid>::read(file, false, false))
                                       : Dune::StructuredGridFactory<Grid>::createCubeGrid({0., 0.}, {1., 1.}, {cells, cells});
      grid->globalRefine(refine);
      grid->loadBalance();
      if (type == "ug-simplex") {
        using FEM = Dune::PDELab::OPBLocalFiniteElementMap<double, double, 1, 2, Dune::GeometryType::simplex>;
        FEM fem;
        return run_es(*grid, fem, ptree);
      }
      using FEM = Dune::PDELab::QkDGLocalFiniteElementMap<double, double, 1, 2>;
      FEM fem;
      return run_es(*grid, fem, ptree);
    }
#endif
    if (helper.rank() == 0) std::cerr << "grid.type must be simplex, cube, ug-simplex or ug-cube, got " << type << std::endl;
    return 2;
  }
  catch (const Dune::Exception& e) {
    std::cerr << "Dune reported error: " << e << std::endl;
    return 3;
  }
}
