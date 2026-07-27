#if HAVE_CONFIG_H
#include "config.h"
#endif

#include "dune/ddm/combined_preconditioner.hh"
#include "dune/ddm/galerkin_preconditioner.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/overlapping_matrix_operator.hh"
#include "dune/ddm/pou.hh"
#include "dune/ddm/schwarz.hh"
#include "test_utils.hh"

#include <cstddef>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/referenceelements.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/solvers.hh>
#include <dune/localfunctions/lagrange/lagrangelfecache.hh>
#include <vector>

struct Problem {
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;

  /** @brief Assembles the Q1 stiffness matrix and load vector of
   *
   *    -div(a(x) grad u) = f(x) in Omega,  u = 0 on the Dirichlet part of the boundary.
   *
   *  The assembly runs over every element of @p gv, overlap elements included, so each rank ends
   *  up with the stiffness matrix of its own overlapping subdomain.
   *
   *  @param gv            grid view to assemble on
   *  @param is_dirichlet  predicate on a global coordinate; true for the vertices carrying a
   *                       homogeneous Dirichlet condition. It sees global coordinates, so ranks
   *                       sharing a vertex classify it identically.
   *  @param a             scalar diffusion coefficient, evaluated at global coordinates
   *  @param f             source term, evaluated at global coordinates
   */
  template <class GridView, class IsDirichlet, class Coefficient, class Source>
  Problem(const GridView& gv, IsDirichlet is_dirichlet, Coefficient a, Source f)
  {
    using DF = typename GridView::ctype;
    constexpr int dim = GridView::dimension;

    auto& indexset = gv.indexSet();
    const int n = indexset.size(dim);

    A = std::make_shared<Matrix>();
    A->setBuildMode(Matrix::BuildMode::implicit);
    A->setImplicitBuildModeParameters(std::pow(3, dim), 0.05);
    A->setSize(n, n);

    // Create sparsity pattern
    for (const auto& e : elements(gv)) {
      auto ndofs = e.subEntities(dim);
      for (unsigned int i = 0; i < ndofs; ++i)
        for (unsigned int j = 0; j < ndofs; ++j) A->entry(indexset.subIndex(e, i, dim), indexset.subIndex(e, j, dim)) = 0;
    }
    A->compress();

    b.resize(n);
    b = 0.;

    dirichlet.assign(n, false);
    for (const auto& v : vertices(gv)) dirichlet[indexset.index(v)] = is_dirichlet(v.geometry().corner(0));

    subdomain_boundary.assign(n, false);

    // Assemble the matrix entries
    Dune::LagrangeLocalFiniteElementCache<DF, double, dim, 1> fecache;

    std::vector<Dune::FieldVector<double, 1>> phi;                      // shape function values
    std::vector<Dune::FieldMatrix<double, 1, dim>> reference_gradients; // gradients on the reference element
    std::vector<Dune::FieldVector<double, dim>> gradients;              // ... pushed forward to the element
    std::vector<std::size_t> indices;                                   // global index of each local dof

    for (const auto& e : elements(gv)) {
      const auto& fe = fecache.get(e.type());
      const auto& localbasis = fe.localBasis();
      const auto geo = e.geometry();
      const std::size_t ndofs = localbasis.size();

      phi.resize(ndofs);
      reference_gradients.resize(ndofs);
      gradients.resize(ndofs);

      // For Q1 every dof sits on a vertex, so the local dof index maps to a codim-dim subentity.
      indices.resize(ndofs);
      for (std::size_t i = 0; i < ndofs; ++i) {
        const auto& key = fe.localCoefficients().localKey(i);
        assert(key.codim() == dim);
        indices[i] = indexset.subIndex(e, key.subEntity(), dim);
      }

      // Mark the vertices on the boundary of the local subdomain, i.e. of the union of the elements
      // this rank holds. neighbor() is false exactly when the element across the face is not part of
      // the local grid view, which happens both where the overlap stops and on the physical domain
      // boundary; boundary() separates the two.
      const auto& refelem = Dune::referenceElement(geo);
      for (const auto& is : intersections(gv, e)) {
        if (is.neighbor()) continue;
        const auto face = is.indexInInside();
        for (int k = 0; k < refelem.size(face, 1, dim); ++k) subdomain_boundary[indexset.subIndex(e, refelem.subEntity(face, 1, k, dim), dim)] = true;
      }

      // Exact for the bilinear form on affine cubes; the coefficient and the source are only
      // approximated, which is all this test needs.
      const auto& rule = Dune::QuadratureRules<DF, dim>::rule(e.type(), 2 * localbasis.order());
      for (const auto& qp : rule) {
        const auto& pos = qp.position();
        const auto jit = geo.jacobianInverseTransposed(pos);
        const double weight = qp.weight() * geo.integrationElement(pos);

        localbasis.evaluateFunction(pos, phi);
        localbasis.evaluateJacobian(pos, reference_gradients);
        for (std::size_t i = 0; i < ndofs; ++i) jit.mv(reference_gradients[i][0], gradients[i]);

        const auto global = geo.global(pos);
        const double a_x = a(global);
        const double f_x = f(global);

        for (std::size_t i = 0; i < ndofs; ++i) {
          b[indices[i]][0] += f_x * phi[i][0] * weight;
          for (std::size_t j = 0; j < ndofs; ++j) (*A)[indices[i]][indices[j]][0][0] += a_x * (gradients[i] * gradients[j]) * weight;
        }
      }
    }

    // Homogeneous Dirichlet conditions: replace each constrained row by the identity row and zero
    // its load entry. The columns are left alone, so A is not symmetric.
    for (auto ri = A->begin(); ri != A->end(); ++ri) {
      if (not dirichlet[ri.index()]) continue;
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) *ci = (ci.index() == ri.index()) ? 1. : 0.;
      b[ri.index()] = 0.;
    }
  }

  std::shared_ptr<Matrix> A;
  Vector b;

  /// Vertices carrying a homogeneous Dirichlet condition of the global problem.
  std::vector<bool> dirichlet;

  /// Vertices on the boundary of this rank's subdomain (including the global dirichlet boundary)
  std::vector<bool> subdomain_boundary;
};

/** @brief The identity, as a preconditioner that communicates nothing.
 *
 *  Lets a solver run genuinely unpreconditioned. Passing a sequential preconditioner instead would
 *  make the factory wrap it in Dune::BlockPreconditioner, which restores consistency itself and
 *  would hide the very property this is meant to check.
 */
template <class Vec>
struct IdentityPreconditioner : public Dune::Preconditioner<Vec, Vec> {
  Dune::SolverCategory::Category category() const override { return Dune::SolverCategory::overlapping; }
  void pre(Vec&, Vec&) override {}
  void post(Vec&) override {}
  void apply(Vec& v, const Vec& d) override { v = d; }
};

int main(int argc, char** argv)
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    setup_loggers(helper.rank(), argc, argv);

    const int dim = 2;
    const int gridsize = 64;
    const int overlap = 4;

    using Grid = Dune::YaspGrid<dim>;
    Grid grid({1., 1.}, {gridsize, gridsize}, 0ULL, overlap);
    auto gv = grid.leafGridView();

    auto comm = ddmtest::create_communication_for_grid(gv);
    using Communication = std::decay_t<decltype(*comm)>;

    // Homogeneous Dirichlet conditions on the whole boundary of the unit square.
    auto is_dirichlet = [](const auto& x) {
      for (int i = 0; i < dim; ++i)
        if (x[i] < 1e-10 or x[i] > 1. - 1e-10) return true;
      return false;
    };
    auto coefficient = [](const auto&) { return 1.; };
    auto source = [](const auto&) { return 1.; };

    Problem p(gv, is_dirichlet, coefficient, source);
    comm->copyOwnerToAll(p.b, p.b); // Make b consistent

    using Operator = OverlappingMatrixOperator<typename Problem::Matrix, typename Problem::Vector, typename Problem::Vector, Communication>;
    auto op = std::make_shared<Operator>(p.A, comm);

    Dune::initSolverFactories<Operator>();
    Dune::ParameterTree solver_tree;
    solver_tree["verbose"] = (helper.rank() == 0) ? "2" : "0";
    solver_tree["type"] = "cgsolver";
    solver_tree["reduction"] = "1e-8";
    solver_tree["maxit"] = "1000";
    solver_tree["restart"] = "30";

    using SchwarzPrec = SchwarzPreconditioner<typename Problem::Matrix, typename Problem::Vector, Communication>;
    using GalerkinPrec = GalerkinPreconditioner<typename Problem::Vector, Communication>;
    using Prec = CombinedPreconditioner<typename Problem::Vector>;

    auto pou = std::make_shared<PartitionOfUnity>(*p.A, *comm, PartitionOfUnityType::Standard);

    // Build fine-level preconditioner
    Dune::ParameterTree schwarz_tree;
    schwarz_tree["schwarz.type"] = "standard";
    schwarz_tree["schwarz.subdomain_solver.type"] = "umfpack";
    auto Ad = std::make_shared<typename Problem::Matrix>(*p.A);
    for (auto ri = Ad->begin(); ri != Ad->end(); ++ri) {
      if (p.subdomain_boundary[ri.index()] and not p.dirichlet[ri.index()])
        for (auto ci = ri->begin(); ci != ri->end(); ++ci) (*Ad)[ri.index()][ci.index()] = ri.index() == ci.index() ? 1. : 0.;
    }
    auto fine_prec = std::make_shared<SchwarzPrec>(Ad, comm, pou, schwarz_tree);

    // Build coarse-level preconditioner
    Dune::ParameterTree galerkin_tree;
    galerkin_tree["galerkin.type"] = "umfpack";
    typename Problem::Vector t(pou->vector().size());
    std::copy(pou->vector().begin(), pou->vector().end(), t.begin());
    auto coarse_prec = std::make_shared<GalerkinPrec>(*p.A, std::vector<typename Problem::Vector>{t}, comm, galerkin_tree);

    // Combine the two in an additive way
    Dune::ParameterTree combined_tree;
    auto prec = std::make_shared<Prec>(combined_tree);
    prec->add(fine_prec);
    prec->add(coarse_prec);

    auto solver = Dune::getSolverFromFactory(op, solver_tree, prec);

    // Solve the system
    Dune::InverseOperatorResult res;
    typename Problem::Vector x(p.b.size());
    x = 0.;

    auto rhs = p.b;
    solver->apply(x, rhs, res);

    // The same system solved by GMRes alone. This converges only because the operator returns a
    // consistent vector, so the residual can be used directly as a search direction; an operator
    // that projected its result onto the owners would produce nonsense here. See ddm.hh.
    Dune::ParameterTree unprec_tree;
    unprec_tree["verbose"] = (helper.rank() == 0) ? "1" : "0";
    unprec_tree["type"] = "restartedgmressolver";
    unprec_tree["reduction"] = "1e-8";
    unprec_tree["maxit"] = "5000";
    unprec_tree["restart"] = "30";

    Dune::InverseOperatorResult unprec_res;
    typename Problem::Vector x_unprec(p.b.size());
    x_unprec = 0.;
    auto unprec_rhs = p.b;
    auto unprec_solver = Dune::getSolverFromFactory(op, unprec_tree, std::make_shared<IdentityPreconditioner<typename Problem::Vector>>());
    unprec_solver->apply(x_unprec, unprec_rhs, unprec_res);

    // Both solves have to land on the same solution, up to the tolerance they were asked for.
    auto diff = x;
    diff -= x_unprec;
    const double err = comm->norm(diff) / std::max(1., comm->norm(x));
    if (helper.rank() == 0) std::cout << "preconditioned vs unpreconditioned: relative difference " << err << "\n";
    if (not unprec_res.converged or err > 1e-6) {
      if (helper.rank() == 0) std::cout << "TEST FAILED (structured_grid_test): unpreconditioned solve converged=" << unprec_res.converged << ", relative difference " << err << std::endl;
      return 1;
    }

    Dune::VTKWriter writer(gv);
    writer.addVertexData(x, "Solution");
    writer.write("poisson");
  }
  catch (const Dune::Exception& e) {
    std::cout << "Dune exception thrown: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
