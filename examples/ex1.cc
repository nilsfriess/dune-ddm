// Linear elasticity in 3D: a steel beam of size 1 m x 1 m x 10 m along the z axis, clamped at z = 0 and free everywhere
// else, bending under its own weight (gravity in -y direction). P1 elements on a structured grid.
//
// Example: mpirun -np 4 ./ex1 -grid.cells "4 4 40" -prec.type tlschwarz -prec.coarse_space_type geneo

#include "examples/assembly.hh"

#include <array>
#include <ddm/index.hh>
#include <ddm/mat/mat.hh>
#include <ddm/mat/pattern.hh>
#include <ddm/prec/prec.hh>
#include <ddm/solver/solver.hh>
#include <ddm/vec/vec.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/ddm/logger.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/pdelab.hh>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

namespace {
// Parameters for Steel
constexpr double youngs_modulus = 210e9; // Pa
constexpr double poisson_ratio = 0.3;    // non-dim.
constexpr double density = 7850;         // kg/m^3
constexpr double gravity = 9.81;         // m/s^2
constexpr double length = 10;            // m, along z
constexpr double width = 1;              // m, in x and y

template <class GV, class RF>
class BeamParameters : public Dune::PDELab::LinearElasticityParameterInterface<Dune::PDELab::LinearElasticityParameterTraits<GV, RF>, BeamParameters<GV, RF>> {
public:
  using Traits = Dune::PDELab::LinearElasticityParameterTraits<GV, RF>;

  // Body force: the weight of the beam
  void f(const typename Traits::ElementType&, const typename Traits::DomainType&, typename Traits::RangeType& y) const { y = {0, -density * gravity, 0}; }

  // Clamped at z = 0
  template <class I>
  bool isDirichlet(const I& ig, const typename Traits::IntersectionDomainType& coord) const
  {
    return ig.geometry().global(coord)[2] < 1e-9;
  }

  void g(const typename Traits::ElementType&, const typename Traits::DomainType&, typename Traits::RangeType& y) const { y = 0; }

  // The Lame parameters, taken from here: https://handwiki.org/wiki/Physics:Lam%C3%A9_parameters
  RF lambda(const typename Traits::ElementType&, const typename Traits::DomainType&) const { return youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio)); }
  RF mu(const typename Traits::ElementType&, const typename Traits::DomainType&) const { return youngs_modulus / (2 * (1 + poisson_ratio)); }
};

void run(const Dune::ParameterTree& config, const Dune::MPIHelper& helper)
{
  constexpr int dim = 3;

  const auto cells = config.get("grid.cells", std::array<unsigned int, dim>{4, 4, 40});
  auto grid = Dune::StructuredGridFactory<Dune::UGGrid<dim>>::createSimplexGrid({0., 0., 0.}, {width, width, length}, cells);
  const auto gv = grid->leafGridView();
  using GV = std::remove_cvref_t<decltype(gv)>;
  auto part = Dune::ParMetisGridPartitioner<GV>::partition(gv, helper);
  grid->loadBalance(part, 0);
  grid->globalRefine(config.get("grid.refine", 0));

  // Define the types that PDELab needs to assemble the linear system and rhs
  using RF = double;
  using ES = Dune::PDELab::OverlappingEntitySet<GV>;
  using FEM = Dune::PDELab::PkLocalFiniteElementMap<ES, double, RF, 1>;
  using GFS = Dune::PDELab::VectorGridFunctionSpace<ES, FEM, dim, Dune::PDELab::ISTL::VectorBackend<>, Dune::PDELab::ISTL::VectorBackend<>, Dune::PDELab::ConformingDirichletConstraints>;

  ES es(gv);
  FEM fem(es);
  GFS gfs(es, fem);
  gfs.name("displacement");

  BeamParameters<ES, RF> params;
  typename GFS::template ConstraintsContainer<RF>::Type cc;
  Dune::PDELab::constraints(params, gfs, cc);
  Dune::PDELab::LinearElasticity lop(params);
  auto [A, b] = ddm::examples::assemble_conforming<RF>(config, gfs, lop, cc);

  // Now the problem is fully assembled and we can setup the preconditioner and solver and solve it
  auto P = ddm::create_prec<RF>(config.sub("prec"), A);
  auto solver = ddm::create_solver<RF>(config.sub("solver"), A, P);
  auto x = A->create_domain_vector();
  solver->solve(b, x);

  // VTK output of the displacement
  Dune::PDELab::Backend::Vector<GFS, RF> x_pdelab(gfs, 0.0);
  {
    const auto view = x.host_view(ddm::read);
    auto& native = Dune::PDELab::Backend::native(x_pdelab);
    for (std::size_t i = 0; i < view.size(); ++i) native[i][0] = view[i];
  }
  Dune::VTKWriter writer(gv);
  Dune::PDELab::addSolutionToVTKWriter(writer, gfs, x_pdelab);
  std::vector<int> ranks(gv.size(0), gv.comm().rank());
  writer.addCellData(ranks, "Rank");
  writer.write("ex1");
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
