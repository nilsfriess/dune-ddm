#pragma once

#include <dune/common/parallel/mpihelper.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/yaspgrid.hh>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#include <dune/pdelab.hh>
#pragma GCC diagnostic pop

#include <memory>
#include <tuple>

template <class Grid>
constexpr bool isYASPGrid()
{
  return requires(Grid g) { g.torus(); }; // We distinguish YASPGrid and UGGrid using the method "torus" which only exists in YASPGrid
}

template <class Grid>
std::unique_ptr<Grid> make_grid(const Dune::MPIHelper &helper, const Dune::ParameterTree &ptree, const std::string &subtree_name = "grid")
{
  const auto &subtree = ptree.sub(subtree_name);
  auto size = subtree.get("size", 128);
  auto par_refine = subtree.get("par_refine", 0);

  std::unique_ptr<Grid> grid;
  if constexpr (!isYASPGrid<Grid>()) {
    grid = Dune::StructuredGridFactory<Grid>::createCubeGrid({0, 0}, {1, 1}, {size, size});
    auto part = Dune::ParMetisGridPartitioner<typename Grid::LeafGridView>::partition(grid->leafGridView(), helper);
    grid->loadBalance(part, 0);
  }
  else {
    Dune::Yasp::PowerDPartitioning<2> partitioner;
    grid = std::unique_ptr<Grid>(new Grid({1., 1.}, {size, size}, std::bitset<2>{0}, 0, typename Grid::Communication(), &partitioner));
    grid->loadBalance();
  }
  grid->globalRefine(par_refine);
  return grid;
}

template <class GV>
auto assemble_problem(const GV &gv)
{
  // Set up finite element problem
  using RF = double;
  using DF = typename GV::ctype;

  using ES = Dune::PDELab::OverlappingEntitySet<GV>;
  using Problem = Dune::PDELab::ConvectionDiffusionModelProblem<ES, RF>;
  using BCType = Dune::PDELab::ConvectionDiffusionBoundaryConditionAdapter<Problem>;

  ES es(gv);
  Problem problem;
  BCType bctype(es, problem);

  using FEM = Dune::PDELab::QkLocalFiniteElementMap<ES, DF, RF, 1>;
  auto fem = std::make_shared<FEM>(es);

  using GFS = Dune::PDELab::GridFunctionSpace<ES, FEM, Dune::PDELab::ConformingDirichletConstraints, Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>>;
  using Vec = Dune::PDELab::Backend::Vector<GFS, RF>;
  using CC = typename GFS::template ConstraintsContainer<RF>::Type;

  GFS gfs(es, fem);
  gfs.name("Solution");
  CC cc;
  Dune::PDELab::constraints(bctype, gfs, cc);

  using LOP = Dune::PDELab::ConvectionDiffusionFEM<Problem, FEM>;
  LOP lop(problem);

  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, RF, RF, RF, CC, CC>;
  const auto nz = Dune::power(2 * 1 + 1, 2); // nonzeros per row = (2 * degree + 1) ^ 2;
  GO go(gfs, cc, gfs, cc, lop, MBE(nz));

  // Create a DOF vector, initialised so that it matches the Dirichlet BC's
  using BCAdapter = Dune::PDELab::ConvectionDiffusionDirichletExtensionAdapter<Problem>;
  Vec x(gfs, 0);
  BCAdapter bca(es, problem);
  Dune::PDELab::interpolate(bca, gfs, x);

  // Assemble residual and make consistent
  Vec d(gfs, 0);
  go.residual(x, d);
  Dune::PDELab::AddDataHandle adddhd(gfs, d);
  gfs.gridView().communicate(adddhd, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);

  // Assemble stiffness matrix
  using Mat = GO::Jacobian;
  auto A_ = std::make_shared<Mat>(go);
  go.jacobian(x, *A_);

  // Eliminate Dirichlet dofs symmetrically
  using Dune::PDELab::Backend::native;

  Vec dirichlet_mask(gfs);
  dirichlet_mask = 0;
  Dune::PDELab::set_constrained_dofs(cc, 1., dirichlet_mask);

  for (auto ri = native(*A_).begin(); ri != native(*A_).end(); ++ri) {
    if (native(dirichlet_mask)[ri.index()] > 0) {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        *ci = (ci.index() == ri.index()) ? 1.0 : 0.0;
      }
    }
    else {
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
        if (native(dirichlet_mask)[ci.index()] > 0) {
          *ci = 0.0;
        }
      }
    }
  }

  return std::make_tuple(A_->storage(), gfs, cc, x, d); // Get a shared_ptr to the native BCRSMatrix
}
