#pragma once

#include <dune/common/fmatrix.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/localfunctions/lagrange/lagrangelfecache.hh>
#include <limits>
#include <memory>

namespace ddmtest {
template <class Scalar = double>
struct PoissonProblem {
  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;
  using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, 1>>;

  /** @brief Assembles the Q1 stiffness matrix and load vector of
   *
   *    -div(a(x) grad u) = f(x) in Omega,  u = 0 on the Dirichlet part of the boundary.
   *
   *  The grid view holds this rank's subdomain extended by overlap + 1 element layers. The
   *  assembly runs over every element of that view, so every vertex more than one element layer
   *  inside the view has its complete global stencil. The matrix is then truncated to the
   *  "patch", i.e. the vertices of the cells whose face neighbours are all in the view (=one
   *  element layer short of the view boundary). Because rows are complete before truncation, the
   *  result is exactly the Galerkin restriction R A R^T of the global matrix. The load vector and
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
  PoissonProblem(const GridView& gv, IsDirichlet is_dirichlet, Coefficient a, Source f)
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
    Dune::LagrangeLocalFiniteElementCache<DF, Scalar, dim, 1> fecache;

    std::vector<Dune::FieldVector<Scalar, 1>> phi;                      // shape function values
    std::vector<Dune::FieldMatrix<Scalar, 1, dim>> reference_gradients; // gradients on the reference element
    std::vector<Dune::FieldVector<Scalar, dim>> gradients;              // ... pushed forward to the element
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

      const auto& rule = Dune::QuadratureRules<DF, dim>::rule(e.type(), 2 * localbasis.order());
      for (const auto& qp : rule) {
        const auto& pos = qp.position();
        const auto jit = geo.jacobianInverseTransposed(pos);
        const double weight = qp.weight() * geo.integrationElement(pos);

        localbasis.evaluateFunction(pos, phi);
        localbasis.evaluateJacobian(pos, reference_gradients);
        for (std::size_t i = 0; i < ndofs; ++i) jit.mv(reference_gradients[i][0], gradients[i]);

        const auto global = geo.global(pos);
        const Scalar a_x = a(global);
        const Scalar f_x = f(global);

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
} // namespace ddmtest
