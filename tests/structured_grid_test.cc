
#if HAVE_CONFIG_H
#include "config.h"
#endif

#include "dune/ddm/sycl/vec.hh"
  
#include "dune/ddm/combined_preconditioner.hh"
#include "dune/ddm/communication.hh"
#include "dune/ddm/consistent_parallel_matrix_operator.hh"
#include "dune/ddm/galerkin_preconditioner.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/pou.hh"
#include "dune/ddm/schwarz.hh"
#include "dune/ddm/sycl/mat.hh"
#include "test_utils.hh"

#include <cstddef>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/solverfactory.hh>
#include <dune/istl/solvers.hh>
#include <dune/localfunctions/lagrange/lagrangelfecache.hh>
#include <limits>
#include <vector>

struct Problem {
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;

  /** @brief Assembles the Q1 stiffness matrix and load vector of
   *
   *    -div(a(x) grad u) = f(x) in Omega,  u = 0 on the Dirichlet part of the boundary.
   *
   *  The grid view holds this rank's subdomain extended by overlap + 1 element layers. The
   *  assembly runs over every element of that view, so every vertex more than one element layer
   *  inside the view has its complete global stencil. The matrix is then truncated to the
   *  "patch", i.e. the vertices of the cells whose face neighbours are all in the view — one
   *  element layer short of the view boundary. Because rows are complete before truncation, the
   *  result is exactly the Galerkin restriction R A Rᵀ of the global matrix: artificial patch
   *  boundary rows keep their full diagonal and nonzero row sums (the truncated couplings play
   *  the role of Dirichlet data carried by the overlap), so the subdomain problems are
   *  invertible even where no physical boundary condition is ever imposed. The load vector and
   *  the Dirichlet mask are restricted in the same way. The local indices of the patch vertices
   *  are a contiguous renumbering of their grid view indices in ascending view index order, which
   *  also defines the correspondence to the communication object's index set.
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

    // The patch: vertices of all elements whose face neighbours are entirely in the view. Cells on
    // the outermost layer of the view are excluded, so every patch vertex has its complete element
    // stencil inside the view; assembling over the whole view and truncating columns outside the
    // patch afterwards therefore yields R A R^T.
    patch.assign(n, false);
    for (const auto& e : elements(gv)) {
      bool complete = true;
      for (const auto& is : intersections(gv, e)) {
        if (not is.neighbor()) {
          complete = false;
          break;
        }
      }
      if (not complete) continue;
      for (unsigned int i = 0; i < e.subEntities(dim); ++i) patch[indexset.subIndex(e, i, dim)] = true;
    }

    // Contiguous local renumbering of the patch vertices, in ascending grid view index order. This
    // is the same numbering that create_communication_for_grid() uses for the local indices of the
    // filtered index set, so matrix rows and communication indices correspond.
    local_of_view.assign(n, invalid);
    std::size_t npatch = 0;
    for (int i = 0; i < n; ++i)
      if (patch[i]) local_of_view[i] = npatch++;

    A->setSize(npatch, npatch);

    // Create sparsity pattern, dropping couplings to vertices outside the patch
    for (const auto& e : elements(gv)) {
      auto ndofs = e.subEntities(dim);
      for (unsigned int i = 0; i < ndofs; ++i) {
        if (not patch[indexset.subIndex(e, i, dim)]) continue;
        for (unsigned int j = 0; j < ndofs; ++j) {
          if (not patch[indexset.subIndex(e, j, dim)]) continue;
          A->entry(local_of_view[indexset.subIndex(e, i, dim)], local_of_view[indexset.subIndex(e, j, dim)]) = 0;
        }
      }
    }
    A->compress();

    b.resize(npatch);
    b = 0.;

    dirichlet.assign(npatch, false);
    for (const auto& v : vertices(gv)) {
      const auto lidx = indexset.index(v);
      if (patch[lidx]) dirichlet[local_of_view[lidx]] = is_dirichlet(v.geometry().corner(0));
    }

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
        indices[i] = local_of_view[indexset.subIndex(e, key.subEntity(), dim)];
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
          if (indices[i] == invalid) continue; // rows of dofs outside the patch are truncated away
          b[indices[i]][0] += f_x * phi[i][0] * weight;
          for (std::size_t j = 0; j < ndofs; ++j) {
            if (indices[j] == invalid) continue; // couplings to dofs outside the patch are dropped
            (*A)[indices[i]][indices[j]][0][0] += a_x * (gradients[i] * gradients[j]) * weight;
          }
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

  /// The overlapping subdomain the matrices/vectors live on: vertices of all elements whose face
  /// neighbours are in the grid view, i.e. one element layer short of the view boundary.
  std::vector<bool> patch;

  /// For every grid view vertex, its patch-local index (the row in A/b), or invalid if it is not
  /// part of the patch. Useful for scattering patch values back to the grid view, e.g. for output.
  std::vector<std::size_t> local_of_view;

  /// The local index used for grid view vertices outside the patch.
  static constexpr std::size_t invalid = std::numeric_limits<std::size_t>::max();
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

template <class Communication, class Matrix, class Vector>
bool solve_schwarz(const Dune::MPIHelper& helper, std::shared_ptr<Communication>& comm, std::shared_ptr<Matrix>& A, const Vector& b, Vector& x)
{
  using Operator = ConsistentParallelMatrixOperator<Matrix, Vector, Vector, Communication>;
  auto op = std::make_shared<Operator>(A, comm);

  Dune::initSolverFactories<Operator>();
  Dune::ParameterTree solver_tree;
  solver_tree["verbose"] = (helper.rank() == 0) ? "2" : "0";
  solver_tree["type"] = "cgsolver";
  solver_tree["reduction"] = "1e-8";
  solver_tree["maxit"] = "1000";
  solver_tree["restart"] = "30";

  using SchwarzPrec = SchwarzPreconditioner<Matrix, Vector>;
  using GalerkinPrec = GalerkinPreconditioner<Vector, Communication>;
  using Prec = CombinedPreconditioner<Vector>;

  auto pou = std::make_shared<PartitionOfUnity>(*A, *comm, PartitionOfUnityType::Standard);

  // Build fine-level preconditioner
  Dune::ParameterTree schwarz_tree;
  schwarz_tree["schwarz.type"] = "standard";
  schwarz_tree["schwarz.subdomain_solver.type"] = "umfpack";
  auto fine_prec = std::make_shared<SchwarzPrec>(A, *comm, pou, schwarz_tree);

  // Build coarse-level preconditioner
  Dune::ParameterTree galerkin_tree;
  galerkin_tree["galerkin.type"] = "umfpack";
  Vector t(pou->vector().size());
  std::copy(pou->vector().begin(), pou->vector().end(), t.begin());
  auto coarse_prec = std::make_shared<GalerkinPrec>(*A, std::vector<Vector>{t}, comm, galerkin_tree);

  // Combine the two in an additive way
  Dune::ParameterTree combined_tree;
  auto prec = std::make_shared<Prec>(combined_tree);
  prec->add(fine_prec);
  prec->add(coarse_prec);

  auto solver = Dune::getSolverFromFactory(op, solver_tree, prec);

  // Solve the system
  Dune::InverseOperatorResult res;
  x = 0.;
  auto rhs = b;
  solver->apply(x, rhs, res);

  return true;
}

template <class Communication, class Matrix, class Vector>
bool solve_reference(const Dune::MPIHelper& helper, std::shared_ptr<Communication>& comm, std::shared_ptr<Matrix>& A, const Vector& b, Vector& x)
{
  using Operator = ConsistentParallelMatrixOperator<Matrix, Vector, Vector, Communication>;
  auto op = std::make_shared<Operator>(A, comm);
  Dune::initSolverFactories<Operator>();

  Dune::ParameterTree unprec_tree;
  unprec_tree["verbose"] = (helper.rank() == 0) ? "1" : "0";
  unprec_tree["type"] = "restartedgmressolver";
  unprec_tree["reduction"] = "1e-8";
  unprec_tree["maxit"] = "5000";
  unprec_tree["restart"] = "30";

  Dune::InverseOperatorResult unprec_res;
  x = 0.;
  auto unprec_rhs = b;
  auto unprec_solver = Dune::getSolverFromFactory(op, unprec_tree, std::make_shared<IdentityPreconditioner<Vector>>());
  unprec_solver->apply(x, unprec_rhs, unprec_res);

  return true;
}

int main(int argc, char** argv)
{
  try {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    setup_loggers(helper.rank(), argc, argv);

    const int dim = 2;
    const int gridsize = 256;
    const int overlap = 4;

    // ----  Problem setup ----
    using Grid = Dune::YaspGrid<dim>;
    // One element layer more than the method's overlap: the outermost layer is needed only to
    // make the rows of the patch vertices complete during assembly; the matrices are truncated to
    // the patch (see class Problem above), so every local matrix equals R A Rᵀ and the subdomain
    // problems are invertible without imposing artificial boundary conditions on the subdomain boundary.
    Grid grid({1., 1.}, {gridsize, gridsize}, 0ULL, overlap + 1);
    auto gv = grid.leafGridView();

    // Homogeneous Dirichlet conditions on the whole boundary of the unit square.
    auto is_dirichlet = [](const auto& x) {
      for (int i = 0; i < dim; ++i)
        if (x[i] < 1e-10 or x[i] > 1. - 1e-10) return true;
      return false;
    };
    auto coefficient = [](const auto&) { return 1.; };
    auto source = [](const auto&) { return 1.; };

    Problem p(gv, is_dirichlet, coefficient, source);
    auto comm = ddmtest::create_communication_for_grid(gv, p.patch);
    if (comm->indexSet().size() != static_cast<std::size_t>(p.A->N()))
      DUNE_THROW(Dune::InvalidStateException, "communication index set (" << comm->indexSet().size() << ") and matrix (" << p.A->N() << ") differ in size");
    comm->copyOwnerToAll(p.b, p.b); // Make b consistent

    { // ISTL backend tests
      // ----  Solve system with unpreconditioned GMRes to get a reference solution ----
      typename Problem::Vector x_unprec(p.b.size());
      solve_reference(helper, comm, p.A, p.b, x_unprec);

      // ----  Solve system with CG preconditioned with additive Schwarz (ISTL backend) ----
      typename Problem::Vector x_schwarz(p.b.size());
      solve_schwarz(helper, comm, p.A, p.b, x_schwarz);
    }

    {
      using SyclVec = ddm::Sycl::Vec<double>;
      using SyclMat = ddm::Sycl::Mat<double>;

      auto vec_comm = std::make_shared<ddm::Communication>(ddm::make_communication_from_dune(*comm));

      sycl::queue q{sycl::property::queue::in_order{}};
      SyclVec x(q, p.b.size());
      auto b = SyclVec::from_host_vector(q, p.b);
      auto A = std::make_shared<SyclMat>(SyclMat::from_bcrs(q, *p.A));

      solve_reference(helper, vec_comm, A, b, x);
    }
    // Both solves have to land on the same solution, up to the tolerance they were asked for.
    // auto diff = x;
    // diff -= x_unprec;
    // // comm->norm() is collective: evaluate it on every rank and restrict only the printing to
    // // rank 0, otherwise the ranks run out of step and deadlock in the reduction.
    // const double err = comm->norm(diff) / std::max(1., comm->norm(x));
    // if (helper.rank() == 0) std::cout << "preconditioned vs unpreconditioned: relative difference " << err << "\n";
    // if (not unprec_res.converged or err > 1e-6) {
    //   if (helper.rank() == 0) std::cout << "TEST FAILED (structured_grid_test): unpreconditioned solve converged=" << unprec_res.converged << ", relative difference " << err << std::endl;
    //   return 1;
    // }

    // Dune::VTKWriter writer(gv);
    // // The solution lives on the patch, whose vertices are a subset of the grid view's; scatter it
    // // back for output. Vertices outside the patch are never written and get zero.
    // typename Problem::Vector x_vtk(gv.size(dim));
    // x_vtk = 0.;
    // for (std::size_t i = 0; i < p.local_of_view.size(); ++i)
    //   if (p.local_of_view[i] != Problem::invalid) x_vtk[i] = x[p.local_of_view[i]];
    // writer.addVertexData(x_vtk, "Solution");
    // writer.write("poisson");

    Logger::get().report(MPI_COMM_WORLD);
  }
  catch (const Dune::Exception& e) {
    std::cout << "Dune exception thrown: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
