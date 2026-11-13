// Convection-diffusion in 3D: -div(kappa grad u) + div(b u) = 0 in the unit cube, with a swirling flow that carries a
// spot of concentration from the bottom face to the top face along a helix. Interior penalty DG (SIPG for the
// diffusion, upwinding for the convection) with Q1 elements on a structured grid of cubes.
//
// The velocity is a vortex around the vertical axis through the centre of the cube, plus a constant upward flow:
//
//   b = velocity_scale * (-swirl pi sin(pi x) cos(pi y), swirl pi cos(pi x) sin(pi y), 1),
//
// the curl of the stream function swirl sin(pi x) sin(pi y) plus (0, 0, 1). It is divergence free, so the symmetric part
// of the convection-diffusion operator is the diffusion, and tangential to the side faces of the cube. Its streamlines
// are helices around the axis, with about swirl pi^2 / (2 pi) turns near the axis while they rise through the cube.
//
// The diffusion coefficient is one of (problem.coefficient):
// - checkerboard: checkers^3 cubes, alternating between kappa_light and kappa_dark, as in the 2D example of Section 5.1
//   in arXiv:2509.17531 (ex2). With kappa_dark much smaller than kappa_light, the problem is convection dominated in
//   the dark cubes and has high contrast. In 3D (and DG, which only couples elements through faces) the light cubes
//   only touch along edges, so they are isolated inclusions: hardly harder for one-level methods than kappa = 1.
// - channels: straight rods with kappa_channel in a background with kappa_background, which run through the whole
//   cube and so cross many subdomain boundaries, the hard case for one-level methods and simple coarse spaces. The cube
//   is cut into channels slices of thickness 1/channels in z; the lower channel_width of each slice holds channels rods
//   with a square cross section of side channel_width, at y = j / channels along x in even slices and at
//   x = j / channels along y in odd slices. The rods do not touch each other. With channel_width = 1 / (2 channels),
//   the rods need at least 2 channels elements per direction, and their edges lie on element faces for multiples of
//   that.
//
// Boundary conditions: the bottom face (z = 0) is the inflow boundary, with the Dirichlet data u = exp(-|x - x_s|^2 /
// (2 w^2)), a Gaussian spot of width w = spot_width around x_s = spot (default off the axis, so that the spot is carried
// around it). The side faces have b . n = 0 and no flux (homogeneous Neumann), the top face is the outflow boundary.
// The Dirichlet condition is imposed weakly by the penalty terms, there are no constrained dofs.
//
// As in ex2, PDELab provides the function space, the local operator (ConvectionDiffusionDG) and the grid operator, and
// the matrix of the diffusion-only problem is attached to the matrix (see Mat::attach_symmetric_matrix()), so that
// MS-GFEM measures the harmonic functions in the energy of the diffusion. The coarse UGGrid is distributed with ParMETIS
// and then refined grid.refine times, so that no rank has to hold the fine grid.
//
// The options are read from the ini files in -config and the command line (see read_config() in run.hh). The ini files
// in examples/ini/ document the options of the experiments: ex6.ini the problem, the fine level, the eigensolver and
// GMRES, and the files in examples/ini/methods/ the preconditioner. With output.json, the configuration, the iteration
// count and the timings of all ranks are written to that file, and output.vtk (default true) writes the solution.
//
// Example: mpirun -np 64 ./ex6 -config "examples/ini/ex6.ini examples/ini/methods/msgfem_ring.ini"

#include "ddm/debug/vec_sink.hh"
#include "examples/assembly.hh"
#include "examples/pdelab_communication.hh"
#include "examples/run.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <ddm/logger.hh>
#include <ddm/mat/mat.hh>
#include <ddm/prec/prec.hh>
#include <ddm/solver/solver.hh>
#include <ddm/vec/vec.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <dune/pdelab.hh>
#pragma GCC diagnostic pop
#include <exception>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr int dim = 3;
using Point = Dune::FieldVector<double, dim>;

// The coefficients of the problem, see the header
struct Problem {
  explicit Problem(const Dune::ParameterTree& config)
      : velocity_scale(config.get("problem.velocity_scale", 1.))
      , swirl(config.get("problem.swirl", 1.))
      , coefficient(config.get("problem.coefficient", std::string("checkerboard")))
      , kappa_light(config.get("problem.kappa_light", 1.))
      , kappa_dark(config.get("problem.kappa_dark", 1e-6))
      , checkers(config.get("problem.checkers", 8))
      , kappa_channel(config.get("problem.kappa_channel", 1e6))
      , kappa_background(config.get("problem.kappa_background", 1.))
      , channels(config.get("problem.channels", 16))
      , channel_width(config.get("problem.channel_width", 1. / 32))
      , spot(config.get("problem.spot", Point{0.5, 0.25, 0.}))
      , spot_width(config.get("problem.spot_width", 0.06))
  {
    if (coefficient != "checkerboard" && coefficient != "channels") DUNE_THROW(Dune::Exception, std::format("Unknown problem.coefficient `{}`, known: checkerboard, channels", coefficient));
    if (coefficient == "channels" && !(channel_width > 0 && channel_width < 1. / channels))
      DUNE_THROW(Dune::Exception, std::format("problem.channel_width must lie in (0, 1 / channels), got {} with {} channels", channel_width, channels));
  }

  // x is the element centre (see Parameters::A())
  double kappa(const Point& x) const
  {
    if (coefficient == "channels") {
      // Is t in the lower channel_width of its slice of thickness 1 / channels?
      const auto in_band = [&](double t) { return t * channels - std::floor(t * channels) < channel_width * channels; };
      if (!in_band(x[2])) return kappa_background;
      const int slice = std::clamp(static_cast<int>(std::floor(x[2] * channels)), 0, channels - 1);
      return in_band(slice % 2 == 0 ? x[1] : x[0]) ? kappa_channel : kappa_background;
    }
    int sum = 0;
    for (int d = 0; d < dim; ++d) sum += std::clamp(static_cast<int>(std::floor(x[d] * checkers)), 0, checkers - 1);
    return sum % 2 == 0 ? kappa_light : kappa_dark;
  }

  Point velocity(const Point& x) const
  {
    using std::numbers::pi;
    return {-velocity_scale * swirl * pi * std::sin(pi * x[0]) * std::cos(pi * x[1]), velocity_scale * swirl * pi * std::cos(pi * x[0]) * std::sin(pi * x[1]), velocity_scale};
  }

  double inflow(const Point& x) const
  {
    auto d = x;
    d -= spot;
    d[2] = 0;
    return std::exp(-d.two_norm2() / (2 * spot_width * spot_width));
  }

  double kappa_min() const { return coefficient == "channels" ? std::min(kappa_channel, kappa_background) : std::min(kappa_light, kappa_dark); }

  std::string describe() const
  {
    if (coefficient == "channels") return std::format("kappa {} in {} channels of width {} per slice, {} elsewhere", kappa_channel, channels, channel_width, kappa_background);
    return std::format("kappa {} / {} on {}^3 checkers", kappa_light, kappa_dark, checkers);
  }

  // The largest |b|: the vortex is fastest at the middle of the side faces
  double max_velocity() const { return velocity_scale * std::sqrt(1 + swirl * swirl * std::numbers::pi * std::numbers::pi); }

  double velocity_scale;
  double swirl;
  std::string coefficient;
  double kappa_light;
  double kappa_dark;
  int checkers;
  double kappa_channel;
  double kappa_background;
  int channels;
  double channel_width;
  Point spot;
  double spot_width;
};

// Without convection, these are the parameters of the diffusion-only problem, whose matrix is attached to the matrix of
// the full problem
template <class GV, class RF>
class Parameters : public Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF> {
public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF>::Traits;

  Parameters(const Problem& problem, bool convection)
      : problem_(problem)
      , convection_(convection)
  {
  }

  // ConvectionDiffusionDG evaluates this at the element centre only, so kappa is constant per element
  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType K(0.0);
    const auto k = problem_.kappa(e.geometry().global(x));
    for (std::size_t i = 0; i < Traits::dimDomain; ++i) K[i][i] = k;
    return K;
  }

  typename Traits::RangeType b(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    if (!convection_) return typename Traits::RangeType(0.0);
    return problem_.velocity(e.geometry().global(x));
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType&, const typename Traits::DomainType&) const { return 0; }

  // Inflow (Dirichlet) at the bottom, outflow at the top, no flux through the side faces, where b . n = 0. The
  // ConvectionDiffusionDG skips the convection term on Neumann faces, which is exact there.
  auto bctype(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    const auto xg = is.geometry().global(x);
    if (xg[2] < 1e-8) return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
    if (xg[2] > 1 - 1e-8) return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Outflow;
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Neumann;
  }

  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    const auto xg = e.geometry().global(x);
    return xg[2] < 1e-8 ? problem_.inflow(xg) : 0.0;
  }

private:
  const Problem& problem_;
  bool convection_;
};

void run(const Dune::ParameterTree& config, const Dune::MPIHelper& helper)
{
  constexpr int degree = 1;
  using RF = double;

  const auto cells = config.get("grid.cells", std::array<unsigned int, dim>{16, 16, 16});
  auto grid = Dune::StructuredGridFactory<Dune::UGGrid<dim>>::createCubeGrid({0., 0., 0.}, {1., 1., 1.}, cells);
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

  const Problem problem(config);
  const auto peclet_max = problem.max_velocity() / problem.kappa_min();
  if (gv.comm().rank() == 0)
    std::cout << std::format("Velocity scale {}, swirl {}, {}, max |b| / kappa {:.3e}", problem.velocity_scale, problem.swirl, problem.describe(), peclet_max) << std::endl;

  using Params = Parameters<ES, RF>;
  Params params(problem, true);
  Params params_sym(problem, false);
  using LOP = Dune::PDELab::ConvectionDiffusionDG<Params, FEM>;
  // Penalty factor of the interior penalty terms, as in ex2
  constexpr double alpha = 2;
  LOP lop(params, Dune::PDELab::ConvectionDiffusionDGMethod::SIPG, Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn, alpha);
  LOP lop_sym(params_sym, Dune::PDELab::ConvectionDiffusionDGMethod::SIPG, Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn, alpha);

  // Only the grid loops of the grid operators are used, PDELab assembles no matrix (the matrix backend is part of the
  // type). There are no constraints.
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, RF, RF, RF>;
  GO go(gfs, gfs, lop, MBE(7));
  GO go_sym(gfs, gfs, lop_sym, MBE(7));

  // As in PDELab, we solve for a correction: with the Jacobian A and the residual r at x0 = 0, A d = r gives the
  // solution x0 - d. The Dirichlet data is in the residual, through the penalty terms.
  using X = GO::Traits::Domain;
  X x0(gfs, 0.0);
  auto comm = make_communication(gfs);
  std::shared_ptr<ddm::Mat<RF>> A;
  std::optional<ddm::Vec<RF>> r;
  {
    ddm::examples::TimedPhase phase("assemble");
    A = ddm::examples::assemble_jacobian(config, go, x0, comm);
    A->attach_symmetric_matrix(ddm::examples::assemble_jacobian(config, go_sym, x0, comm));
    r.emplace(ddm::examples::assemble_residual(go, x0, *A));
  }

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

  std::shared_ptr<ddm::Prec<RF>> P;
  std::shared_ptr<ddm::Solver<RF>> solver;
  {
    ddm::examples::TimedPhase phase("setup");
    P = ddm::create_prec<RF>(config.sub("prec"), A);
    solver = ddm::create_solver<RF>(config.sub("solver"), A, P);
  }
  auto d = A->create_domain_vector();
  {
    ddm::examples::TimedPhase phase("solve");
    solver->solve(*r, d);
  }
  auto x = A->create_domain_vector();
  ddm::examples::copy(x0, x);
  x -= d;

  const auto dofs = A->communication()->count_owners();
  if (gv.comm().rank() == 0) std::cout << "Dofs: " << dofs << ", ranks: " << gv.comm().size() << std::endl;
  ddm::examples::write_results(config, *solver, {{"dofs", std::to_string(dofs)}, {"peclet max", std::format("{:.6e}", peclet_max)}});

  // VTK output of the solution and the diffusion coefficient
  if (!config.get("output.vtk", true)) return;
  add_to_writer(gfs.name(), x);
  // addCellData() with all arguments, the short form is ambiguous for SubsamplingVTKWriter
  std::vector<double> kappas(gv.size(0));
  for (const auto& e : elements(gv)) kappas[gv.indexSet().index(e)] = problem.kappa(e.geometry().center());
  writer.addCellData(kappas, "kappa", 1, Dune::VTK::Precision::float32);
  std::vector<int> ranks(gv.size(0), gv.comm().rank());
  writer.addCellData(ranks, "Rank", 1, Dune::VTK::Precision::int32);
  writer.write(config.get("output.vtk_name", std::string("ex6")));
}
} // namespace

int main(int argc, char** argv)
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    const auto config = ddm::examples::read_config(argc, argv);
    run(config, helper);
    Logger::get().report(MPI_COMM_WORLD);
  }
  catch (std::exception& e) {
    std::cout << e.what() << "\n";
    return 1;
  }
}
