// Diffusion in 2D: -div(kappa grad u) = 1 on (0, 1)^2, u = 0 on the boundary. Symmetric interior penalty DG with Q1
// elements on a structured grid of squares.
//
// kappa is the "skyscraper" coefficient: 1 in the background, with an 8 x 8 array of squares (the skyscrapers) whose
// values grow from 1e5 at (0, 0) to 1.5e6 at (1, 1), and four thin rotated channels with 1.5e5 and 2e6.
//
// As in ex1 and ex2, PDELab provides the function space, the local operator (ConvectionDiffusionDG without convection)
// and the grid operator, assemble_jacobian() and assemble_residual() in assembly.hh assemble into a ddm::Mat and a
// ddm::Vec with the grid loop of the grid operator. The DG dofs live on the elements. The function space includes
// UGGrid's ghost elements, so that the faces between two ranks can be assembled on one of them. The Dirichlet condition
// is imposed weakly by the penalty terms on the boundary, there are no constrained dofs.
//
// Example: mpirun -np 4 ./ex3 -grid.cells "128 128" -prec.type tlschwarz -prec.coarse_space_type geneo

#include "ddm/debug/vec_sink.hh"
#include "examples/assembly.hh"
#include "examples/pdelab_communication.hh"

#include <array>
#include <cmath>
#include <ddm/index.hh>
#include <ddm/mat/mat.hh>
#include <ddm/mat/pattern.hh>
#include <ddm/prec/prec.hh>
#include <ddm/solver/solver.hh>
#include <ddm/vec/vec.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/common/shared_ptr.hh>
#include <ddm/logger.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <dune/pdelab.hh>
#pragma GCC diagnostic pop
#include <exception>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

namespace {
// A channel is the rectangle (u0, u1) x (v0, v1) in the coordinates (u, v) with origin (cx, cy) whose u axis points in
// direction angle
struct Channel {
  double cx, cy, angle, u0, u1, v0, v1, kappa;
};
// Where channels overlap, the first one wins
constexpr std::array channels{
    Channel{0.15, 0.55, std::numbers::pi / 10, 0, 0.52, 0.01, 0.06, 2e6},
    Channel{0.15, 0.55, 0.325 * std::numbers::pi, -0.03, 0.03, -0.7, 0.05, 2e6},
    Channel{0.875, 0.25, 0.75 * std::numbers::pi, -0.025, 0.025, -0.25, 0.25, 1.5e5},
    Channel{0.825, 0.7, std::numbers::pi / 3, -0.025, 0.025, -0.275, 0.275, 1.5e5},
};

double kappa(const Dune::FieldVector<double, 2>& x)
{
  for (const auto& c : channels) {
    const double u = std::cos(c.angle) * (x[0] - c.cx) + std::sin(c.angle) * (x[1] - c.cy);
    const double v = -std::sin(c.angle) * (x[0] - c.cx) + std::cos(c.angle) * (x[1] - c.cy);
    if (c.u0 < u && u < c.u1 && c.v0 < v && v < c.v1) return c.kappa;
  }

  // The skyscraper of cell (i, j) of the 8 x 8 cells is the middle (0.2, 0.8)^2 of the cell
  const double i = std::floor(8 * x[0]);
  const double j = std::floor(8 * x[1]);
  const double s = 8 * x[0] - i;
  const double t = 8 * x[1] - j;
  if (0.2 < s && s < 0.8 && 0.2 < t && t < 0.8) return 1e5 * (1 + i + j);
  return 1;
}

// The boundary is Dirichlet everywhere (bctype() of the base class), there is no convection (b() of the base class)
template <class GV, class RF>
class Parameters : public Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF> {
public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF>::Traits;

  // ConvectionDiffusionDG evaluates this at the element centre only, so kappa is constant per element
  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType K(0.0);
    const auto k = kappa(e.geometry().global(x));
    for (std::size_t i = 0; i < Traits::dimDomain; ++i) K[i][i] = k;
    return K;
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType&, const typename Traits::DomainType&) const { return 1; }

  typename Traits::RangeFieldType g(const typename Traits::ElementType&, const typename Traits::DomainType&) const { return 0; }
};

void run(const Dune::ParameterTree& config, const Dune::MPIHelper& helper)
{
  constexpr int dim = 2;
  constexpr int degree = 2;
  using RF = double;

  const auto cells = config.get("grid.cells", std::array<unsigned int, dim>{128, 128});
  auto grid = Dune::StructuredGridFactory<Dune::UGGrid<dim>>::createCubeGrid({0., 0.}, {1., 1.}, cells);
  const auto gv = grid->leafGridView();
  using GV = std::remove_cvref_t<decltype(gv)>;
  auto part = Dune::ParMetisGridPartitioner<GV>::partition(gv, helper);
  grid->loadBalance(part, 0);
  grid->globalRefine(config.get("grid.refine", 0));

  // All elements, including the ghosts
  using ES = Dune::PDELab::AllEntitySet<GV>;
  using FEM = Dune::PDELab::QkDGLocalFiniteElementMap<double, RF, degree, dim>;
  using GFS = Dune::PDELab::GridFunctionSpace<ES, FEM, Dune::PDELab::NoConstraints, Dune::PDELab::ISTL::VectorBackend<>>;

  ES es(gv);
  FEM fem;
  GFS gfs(es, fem);
  gfs.name("u");

  using Params = Parameters<ES, RF>;
  Params params;
  using LOP = Dune::PDELab::ConvectionDiffusionDG<Params, FEM>;
  // Penalty factor of the interior penalty terms. The standard coercivity estimate on this grid needs alpha > 3/4 for Q1
  // and alpha > 1 for Q2 (because of the boundary faces), so PDELab's default of 1 leaves little or no margin.
  constexpr double alpha = 2;
  LOP lop(params, Dune::PDELab::ConvectionDiffusionDGMethod::SIPG, Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn, alpha);

  // Only the grid loop of the grid operator is used, PDELab assembles no matrix (the matrix backend is part of the
  // type). There are no constraints.
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, RF, RF, RF>;
  GO go(gfs, gfs, lop, MBE(5));

  // As in PDELab, we solve for a correction: with the Jacobian A and the residual r at x0 = 0, A d = r gives the
  // solution x0 - d
  using X = GO::Traits::Domain;
  X x0(gfs, 0.0);
  auto comm = make_communication(gfs);
  auto A = ddm::examples::assemble_jacobian(config, go, x0, comm);
  auto r = ddm::examples::assemble_residual(go, x0, *A);

  // Attach some vectors to the matrix that approximate low energy-modes (they will then be picked up by the Nicolaides coarse space,
  // if it is selected at runtime). Here we attach the constant function and the two linear functions f(x,y) = x and f(x,y) = y.
  std::vector<ddm::Vec<RF>> vs;
  const auto add_vector = [&](auto&& f) {
    X x(gfs, 0.0);
    Dune::PDELab::interpolate(f, gfs, x);
    ddm::examples::copy(x, vs.emplace_back(A->create_domain_vector()));
  };
  add_vector([](const auto&) { return 1.; });
  add_vector([](const auto& x) { return x[0]; });
  add_vector([](const auto& x) { return x[1]; });
  A->set_near_kernel(vs);

  // Define the VTKWriter already here, because the preconditioner might write something to the debug sink. The debug
  // vectors and the solution are written as DG functions, subsampled so that the discontinuous polynomials show.
  using DGF = Dune::PDELab::DiscreteGridFunction<GFS, X>;
  Dune::SubsamplingVTKWriter<GV> writer(gv, Dune::refinementIntervals(degree));
  const auto add_to_writer = [&](std::string_view name, const ddm::Vec<RF>& v) {
    auto u = std::make_shared<X>(gfs, 0.0);
    const auto view = v.host_view(ddm::read);
    auto& native = Dune::PDELab::Backend::native(*u);
    for (std::size_t i = 0; i < view.size(); ++i) native[i][0] = view[i];
    auto dgf = std::make_shared<DGF>(Dune::stackobject_to_shared_ptr(gfs), u);
    writer.addVertexData(std::make_shared<Dune::PDELab::VTKGridFunctionAdapter<DGF>>(dgf, std::string(name)));
  };
  ddm::debug::VecSink<RF> sink(config.get("debug_rank", 0), add_to_writer);

  auto P = ddm::create_prec<RF>(config.sub("prec"), A);
  auto solver = ddm::create_solver<RF>(config.sub("solver"), A, P);
  auto d = A->create_domain_vector();
  solver->solve(r, d);
  auto x = A->create_domain_vector();
  ddm::examples::copy(x0, x);
  x -= d;

  // VTK output of the solution and the coefficient
  add_to_writer(gfs.name(), x);
  // addCellData() with all arguments, the short form is ambiguous for SubsamplingVTKWriter
  std::vector<double> kappas(gv.size(0));
  for (const auto& e : elements(gv)) kappas[gv.indexSet().index(e)] = kappa(e.geometry().center());
  writer.addCellData(kappas, "kappa", 1, Dune::VTK::Precision::float32);
  std::vector<int> ranks(gv.size(0), gv.comm().rank());
  writer.addCellData(ranks, "Rank", 1, Dune::VTK::Precision::int32);
  writer.write("ex3");
}
} // namespace

int main(int argc, char** argv)
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    Dune::ParameterTree config;
    Dune::ParameterTreeParser::readOptions(argc, argv, config);

    run(config, helper);
    Logger::get().report(MPI_COMM_WORLD);
  }
  catch (std::exception& e) {
    std::cout << e.what() << "\n";
    return 1;
  }
}
