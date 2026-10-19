// Convection-diffusion in 2D: -div(kappa grad u) + div(b u) = 1 on (-1, 1)^2, u = 0 on the boundary. Interior penalty
// DG (SIPG for the diffusion, upwinding for the convection) with Q1 elements on a structured grid of squares.
//
// The coefficients are those of the example in Section 5.1 in arXiv:2509.17531. The velocity is the constant
// b = (2/3, 1) towards the top right, so the left and bottom sides are the inflow boundary and the top and
// right sides the outflow boundary. The diffusion coefficient is a checkerboard of 8 x 8 squares, alternating between 1
// and 1e-6.
//
// As in ex3, PDELab provides the function space and the local operator (ConvectionDiffusionDG), the global assembly into
// a ddm::Mat is done by assemble_dg() in assembly.hh. The function space includes UGGrid's ghost elements, so that the
// faces between two ranks can be assembled on one of them. The Dirichlet condition is imposed weakly by the penalty
// terms on the boundary, there are no constrained dofs. The matrix of the diffusion-only problem is assembled alongside
// and attached to the matrix (see Mat::attach_symmetric_matrix()).
//
// Example: mpirun -np 4 ./ex2 -grid.cells "128 128" -prec.type tlschwarz -prec.coarse_space_type nicolaides

#include "ddm/debug/vec_sink.hh"
#include "examples/assembly.hh"

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
#include <dune/ddm/logger.hh>
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
#include <string>
#include <string_view>
#include <vector>

namespace {
// Checkerboard of squares with side checker_size, the square at (0, 0) has kappa_light
constexpr double kappa_dark = 1e-6;
constexpr double kappa_light = 1;
constexpr double checker_size = 0.125;

double kappa(const Dune::FieldVector<double, 2>& x)
{
  const auto i = static_cast<int>(std::floor((x[0] + 1) / checker_size));
  const auto j = static_cast<int>(std::floor((x[1] + 1) / checker_size));
  return (i + j) % 2 == 0 ? kappa_light : kappa_dark;
}

// Without convection, these are the parameters of the diffusion-only problem, whose matrix is attached to the matrix of
// the full problem. The boundary is Dirichlet on the bottom and left boundary, outflow otherwise.
template <class GV, class RF>
class Parameters : public Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF> {
public:
  using Traits = typename Dune::PDELab::ConvectionDiffusionModelProblem<GV, RF>::Traits;

  explicit Parameters(bool convection)
      : convection_(convection)
  {
  }

  // ConvectionDiffusionDG evaluates this at the element centre only, so kappa is constant per element
  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType K(0.0);
    const auto k = kappa(e.geometry().global(x));
    for (std::size_t i = 0; i < Traits::dimDomain; ++i) K[i][i] = k;
    return K;
  }

  typename Traits::RangeType b(const typename Traits::ElementType&, const typename Traits::DomainType&) const
  {
    if (!convection_) return typename Traits::RangeType(0.0);
    return {2. / 3, 1};
  }

  typename Traits::RangeFieldType f(const typename Traits::ElementType&, const typename Traits::DomainType&) const
  {
    return 0;
  }

  auto bctype(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    auto xg = is.geometry().global(x);
    bool is_dirichlet = xg[0] < 1e-8 or xg[1] < 1e-8;
    return is_dirichlet ? Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet : Dune::PDELab::ConvectionDiffusionBoundaryConditions::Outflow;
  }

  typename Traits::RangeFieldType g(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    const auto xg = e.geometry().global(x);
    if (xg[0] < 1e-9) return 1;
    else return 0;
  }

private:
  bool convection_;
};

void run(const Dune::ParameterTree& config, const Dune::MPIHelper& helper)
{
  constexpr int dim = 2;
  constexpr int degree = 1;
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
  Params params(true);
  Params params_sym(false);
  using LOP = Dune::PDELab::ConvectionDiffusionDG<Params, FEM>;
  // Penalty factor of the interior penalty terms. The standard coercivity estimate on this grid needs alpha > 3/4 for Q1
  // and alpha > 1 for Q2 (because of the boundary faces), so PDELab's default of 1 leaves little or no margin.
  constexpr double alpha = 2;
  LOP lop(params, Dune::PDELab::ConvectionDiffusionDGMethod::SIPG, Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn, alpha);
  LOP lop_sym(params_sym, Dune::PDELab::ConvectionDiffusionDGMethod::SIPG, Dune::PDELab::ConvectionDiffusionDGWeights::weightsOn, alpha);

  auto [A, b] = ddm::examples::assemble_dg<RF>(config, gfs, lop, &lop_sym);

  // Define the VTKWriter already here, because the preconditioner might write something to the debug sink. The debug
  // vectors and the solution are written as DG functions, subsampled so that the discontinuous polynomials show.
  using X = Dune::PDELab::Backend::Vector<GFS, RF>;
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
  auto x = A->create_domain_vector();
  solver->solve(b, x);

  // VTK output of the solution and the diffusion coefficient
  add_to_writer(gfs.name(), x);
  // addCellData() with all arguments, the short form is ambiguous for SubsamplingVTKWriter
  std::vector<double> kappas(gv.size(0));
  for (const auto& e : elements(gv)) kappas[gv.indexSet().index(e)] = kappa(e.geometry().center());
  writer.addCellData(kappas, "kappa", 1, Dune::VTK::Precision::float32);
  std::vector<int> ranks(gv.size(0), gv.comm().rank());
  writer.addCellData(ranks, "Rank", 1, Dune::VTK::Precision::int32);
  writer.write("ex2");
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
