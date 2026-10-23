// Diffusion in 2D with a high-contrast coefficient kappa: -div(kappa grad u) = 1 on (0, 1)^2, u = 0 on the boundary.
// The example uses Q1 elements on a structured grid of squares.
//
// Read this example first. It shows the four steps that all examples do:
//   1. Make the grid. Then divide it between the MPI ranks.
//   2. Assemble the matrix A and the residual r.
//   3. Make the preconditioner and the solver. Then solve A d = r.
//   4. Calculate the solution x0 - d. Then write it to a VTK file.
//
// kappa is 1e6 in 8 thin channels and 1 in all other parts of the domain. The channels go across the boundaries
// between the subdomains. For this type of problem, the coarse space of the preconditioner is important.
//
// Set the preconditioner and the solver with options on the command line. This command uses the two-level Schwarz
// preconditioner with the GenEO coarse space:
//
//   mpirun -np 4 ./ex0 -prec.type tlschwarz -prec.coarse_space_type geneo -solver.verbosity 1
//
// With "-solver.verbosity 1", the solver shows the number of iterations. Run the example again with
// "-prec.coarse_space_type nicolaides". With this coarse space, the number of iterations is much larger.

#include "examples/assembly.hh"
#include "examples/pdelab_communication.hh"

#include <array>
#include <cmath>
#include <ddm/logger.hh>
#include <ddm/prec/prec.hh>
#include <ddm/solver/solver.hh>
#include <ddm/vec/vec.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <dune/pdelab.hh>
#pragma GCC diagnostic pop
#include <exception>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {
// Parameters for the diffusion equation with variable coefficient; Homogeneous Dirichlet conditions everywhere,
// source term f = 1 everywhere. The coefficient has 8 horizontal strips of height 1/8. Each strip has one channel
// in the middle. A channel is 1/32 high and goes from x = 1/8 to x = 7/8. kappa is kappa_max in the channels and
// 1 in all other parts of the domain.
template <class GV, class RF>
class Parameters : public Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF> {
public:
  double kappa(const Dune::FieldVector<double, 2>& x) const
  {
    const double y = 8 * x[1] - std::floor(8 * x[1]); // The position in the strip, from 0 to 1
    const bool in_channel = 1. / 8 < x[0] && x[0] < 7. / 8 && 3. / 8 < y && y < 5. / 8;
    return in_channel ? kappa_max_ : 1;
  }

  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF>::Traits;

  explicit Parameters(double kappa_max)
      : kappa_max_(kappa_max)
  {
  }

  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType K(0.0);
    const auto k = kappa(e.geometry().global(x));
    for (std::size_t i = 0; i < Traits::dimDomain; ++i) K[i][i] = k;
    return K;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType&, const typename Traits::DomainType&) const
  {
    return 1;
  }

  typename Traits::RangeFieldType g(const typename Traits::ElementType&, const typename Traits::DomainType&) const
  {
    return 0;
  }

private:
  double kappa_max_;
};

void run(const Dune::ParameterTree& config, const Dune::MPIHelper& helper)
{
  constexpr int dim = 2;
  using RF = double;

  // Step 1: Make a grid of 64 x 64 squares on the unit square. The option -grid.cells changes the number of squares.
  // ParMETIS divides the grid between the MPI ranks. The elements of a rank are the subdomain of that rank. The
  // preconditioner adds the overlap to the subdomains.
  const auto cells = config.get("grid.cells", std::array<unsigned int, dim>{64, 64});
  auto grid = Dune::StructuredGridFactory<Dune::UGGrid<dim>>::createCubeGrid({0., 0.}, {1., 1.}, cells);
  const auto gv = grid->leafGridView();
  using GV = std::remove_cvref_t<decltype(gv)>;
  auto part = Dune::ParMetisGridPartitioner<GV>::partition(gv, helper);
  grid->loadBalance(part, 0);

  // Step 2: Assemble A and r. PDELab gives the function space of the Q1 elements. The entity set contains only the
  // elements of this rank. It does not contain the ghost elements of UGGrid.
  using ES = Dune::PDELab::OverlappingEntitySet<GV>;
  using FEM = Dune::PDELab::QkLocalFiniteElementMap<ES, double, RF, 1>;
  using GFS = Dune::PDELab::GridFunctionSpace<ES, FEM, Dune::PDELab::ConformingDirichletConstraints, Dune::PDELab::ISTL::VectorBackend<>>;

  ES es(gv);
  FEM fem(es);
  GFS gfs(es, fem);
  gfs.name("u");

  // Find the degrees of freedom (dofs) on the boundary. DirichletConstraintsParameters sets a Dirichlet condition on
  // all parts of the boundary.
  Dune::PDELab::DirichletConstraintsParameters bc;
  using CC = GFS::ConstraintsContainer<RF>::Type;
  CC cc;
  Dune::PDELab::constraints(bc, gfs, cc);

  // The grid operator connects the function space, the constraints and the local operator. The functions in assembly.hh
  // use the grid operator only for its loop over the elements and faces.
  using Params = Parameters<ES, RF>;
  Params params(config.get("problem.kappa_max", 1e6));
  using LOP = Dune::PDELab::ConvectionDiffusionFEM<Params, FEM>;
  LOP lop(params);

  // The type of the grid operator contains a matrix backend, so we still have to specify it
  // here, but PDELab never assembles a matrix; we directly assembly into our own matrix type
  // using the functions in assembly.hh.
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, RF, RF, RF, CC, CC>;
  GO go(gfs, cc, gfs, cc, lop, MBE(9));

  // As in PDELab, the solver calculates a correction d. The start value x0 has the boundary values. In this example,
  // these values are zero. A is the matrix and r is the residual R(x0). The solution is x0 - d, and d is zero on the
  // boundary.
  using X = GO::Traits::Domain;
  X x0(gfs, 0.0);

  // make_communication() finds the ranks that have each dof. Each rank adds the element matrices of its elements to its
  // local matrix. The global matrix A is the sum of the local matrices of all ranks. The vector r is consistent: all
  // ranks that have a dof have the same value for this dof. Refer to assembly.hh for more information.
  auto comm = make_communication(gfs);
  auto A = ddm::examples::assemble_jacobian(config, go, x0, comm);
  auto r = ddm::examples::assemble_residual(go, x0, *A);

  // Step 3: Make the preconditioner and the solver. create_prec() uses the options that start with "prec.", and
  // create_solver() uses the options that start with "solver.". The solver starts with d = 0.
  auto P = ddm::create_prec<RF>(config.sub("prec"), A);
  auto solver = ddm::create_solver<RF>(config.sub("solver"), A, P);
  auto d = A->create_domain_vector();
  solver->solve(r, d);

  // Step 4: Copy d into a PDELab vector. Then calculate the solution x = x0 - d. Write the solution, kappa and the rank
  // of each element to a VTK file. With one rank, the file is ex0.vtu. With 4 ranks, the file is s0004-ex0.pvtu.
  X d_pdelab(gfs, 0.0);
  ddm::examples::copy(d, d_pdelab);
  X x = x0;
  x -= d_pdelab;
  Dune::VTKWriter writer(gv);
  Dune::PDELab::addSolutionToVTKWriter(writer, gfs, x);
  std::vector<double> kappas(gv.size(0));
  for (const auto& e : elements(gv)) kappas[gv.indexSet().index(e)] = params.kappa(e.geometry().center());
  writer.addCellData(kappas, "kappa");
  std::vector<int> ranks(gv.size(0), gv.comm().rank());
  writer.addCellData(ranks, "Rank");
  writer.write("ex0");
}
} // namespace

int main(int argc, char** argv)
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    Dune::ParameterTree config;
    Dune::ParameterTreeParser::readOptions(argc, argv, config);

    run(config, helper);

    // Show the time that each part of the library used
    Logger::get().report(MPI_COMM_WORLD);
  }
  catch (std::exception& e) {
    std::cout << e.what() << "\n";
    return 1;
  }
}
