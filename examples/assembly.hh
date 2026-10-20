#pragma once

#include "ddm/index.hh"
#include "ddm/mat/mat.hh"
#include "examples/pdelab_communication.hh"

#include <dune/common/parametertree.hh>
#include <dune/grid/common/partitionset.hh>
#include <dune/pdelab/common/geometrywrapper.hh>
#include <dune/pdelab/gridfunctionspace/lfsindexcache.hh>
#include <dune/pdelab/gridfunctionspace/localfunctionspace.hh>
#include <dune/pdelab/gridfunctionspace/localvector.hh>
#include <dune/pdelab/gridoperator/common/localmatrix.hh>
#include <memory>
#include <vector>

namespace ddm::examples {
template <class T>
struct System {
  std::shared_ptr<ddm::Mat<T>> A;
  ddm::Vec<T> b;
};

// Assembles the stiffness matrix and the load vector element by element for a problem that
// uses conforming finite elements. The matrix gets one add_values() call per element, so
// the element matrices can be recorded (for use with GenEO or MS-GFEM).
//
// If lop_sym is given, the matrix of lop_sym is assembled alongside (only its jacobian_*() methods are used, and it
// gets the same Dirichlet rows) and attached to the matrix as its symmetric part, see Mat::attach_symmetric_matrix().
template <class RF, class GFS, class LOP, class CC>
System<RF> assemble_conforming(const Dune::ParameterTree& config, const GFS& gfs, const LOP& lop, const CC& cc, const LOP* lop_sym = nullptr)
{
  const auto& gv = gfs.gridView();

  auto comm = make_communication(gfs);
  const auto n = static_cast<ddm::Index>(comm->roots().size());

  using LFS = Dune::PDELab::LocalFunctionSpace<GFS>;
  LFS lfs(gfs);
  Dune::PDELab::LFSIndexCache<LFS> cache(lfs);
  std::vector<ddm::Index> dofs;
  const auto bind = [&](const auto& e) {
    lfs.bind(e);
    cache.update();
    dofs.resize(lfs.size());
    for (std::size_t i = 0; i < lfs.size(); ++i) dofs[i] = cache.containerIndex(i)[0];
  };

  // Sparsity pattern
  ddm::Pattern pattern(n, n, 81);
  for (const auto& e : elements(gv, Dune::Partitions::interior)) {
    bind(e);
    for (auto i : dofs)
      for (auto j : dofs) pattern.add(i, j);
  }
  pattern.finalize();

  // Record the element matrices (for GenEO/MsGFEM) unless the config says otherwise
  auto mat_config = config.sub("mat");
  if (!mat_config.hasKey("record_values")) mat_config["record_values"] = "true";
  auto A = ddm::create_mat<RF>(mat_config, pattern, comm);
  auto A_sym = lop_sym != nullptr ? ddm::create_mat<RF>(mat_config, pattern, comm) : nullptr;
  auto b = A->create_range_vector();

  Dune::PDELab::LocalMatrix<RF> local_mat;
  Dune::PDELab::LocalVector<RF> local_x;
  Dune::PDELab::LocalVector<RF> local_r;
  {
    auto b_view = b.host_view(ddm::read_write);
    for (const auto& e : elements(gv, Dune::Partitions::interior)) {
      bind(e);
      const auto m = lfs.size();
      Dune::PDELab::ElementGeometry eg(e);

      // Element matrices (the operators are linear, x is not used) and load vector of lop: lambda_volume() adds
      // -int f v, the residual of the right-hand side. Boundary terms (e.g. Robin or Neumann conditions) go into the
      // element matrices, so that they are recorded with them.
      local_x.assign(m, 0.0);
      local_r.assign(m, 0.0);
      auto r_view = local_r.weightedAccumulationView(1.0);
      for (auto [op, mat] : {std::pair{&lop, A.get()}, std::pair{lop_sym, A_sym.get()}}) {
        if (op == nullptr) continue;
        const bool rhs = op == &lop;
        local_mat.assign(m, m, 0.0);
        auto mat_view = local_mat.weightedAccumulationView(1.0);
        op->jacobian_volume(eg, lfs, local_x, lfs, mat_view);
        if (rhs) op->lambda_volume(eg, lfs, r_view);

        if constexpr (LOP::doAlphaBoundary || LOP::doLambdaBoundary) {
          unsigned int face = 0;
          for (const auto& is : intersections(gv, e)) {
            Dune::PDELab::IntersectionGeometry ig(is, face++);
            if (!is.boundary()) continue;
            if constexpr (LOP::doAlphaBoundary) op->jacobian_boundary(ig, lfs, local_x, lfs, mat_view);
            if constexpr (LOP::doLambdaBoundary) {
              if (rhs) op->lambda_boundary(ig, lfs, r_view);
            }
          }
        }

        mat->add_values(dofs, dofs, local_mat.base()); // both column-major
      }
      for (std::size_t i = 0; i < m; ++i) b_view[dofs[i]] -= local_r.base()[i];
    }
  }
  A->assemble();
  if (A_sym) A_sym->assemble();

  // Homogeneous Dirichlet conditions at the clamped end. Every rank holding a constrained dof has it in cc.
  std::vector<Index> dirichlet;
  for (const auto& [ci, row] : cc) dirichlet.push_back(ci[0]);
  A->zero_rows(dirichlet);
  if (A_sym) {
    A_sym->zero_rows(dirichlet);
    A->attach_symmetric_matrix(A_sym);
  }
  {
    auto b_view = b.host_view(ddm::read_write);
    for (auto i : dirichlet) b_view[i] = 0;
  }

  // The library expects consistent vectors
  A->communication()->reduce(b);
  return {A, std::move(b)};
}

// Assembles the stiffness matrix and the load vector for a problem that uses discontinuous Galerkin elements. The
// function space must include the ghost elements (e.g. PDELab::AllEntitySet). Every element is assembled on the rank
// where it is interior, every face between two elements once, from the element with the smaller global id. A face
// between two ranks is seen from both ranks (the element on the other side is a ghost), so this rule assembles it on
// one of them. The matrix gets one add_values() call per element (volume and boundary terms) and one per face (the
// couplings of the two elements), so these blocks can be recorded (for use with GenEO or MS-GFEM).
//
// The boundary conditions are imposed weakly, there are no constrained dofs.
//
// If lop_sym is given, the matrix of lop_sym is assembled alongside (only its jacobian_*() methods are used) and
// attached to the matrix as its symmetric part, see Mat::attach_symmetric_matrix().
template <class RF, class GFS, class LOP>
System<RF> assemble_dg(const Dune::ParameterTree& config, const GFS& gfs, const LOP& lop, const LOP* lop_sym = nullptr)
{
  const auto& gv = gfs.gridView();
  const auto& ids = gv.grid().globalIdSet();
  const auto assemble_face = [&](const auto& is) {
    return is.neighbor() && ids.id(is.inside()) < ids.id(is.outside());
  };

  auto comm = make_communication(gfs);
  const auto n = static_cast<ddm::Index>(comm->roots().size());

  using LFS = Dune::PDELab::LocalFunctionSpace<GFS>;
  using Cache = Dune::PDELab::LFSIndexCache<LFS>;
  LFS lfs_s(gfs); // the element
  LFS lfs_n(gfs); // its neighbour across a face
  Cache cache_s(lfs_s);
  Cache cache_n(lfs_n);
  std::vector<ddm::Index> dofs_s;
  std::vector<ddm::Index> dofs_n;
  std::vector<ddm::Index> face_dofs;
  const auto bind = [](LFS& lfs, Cache& cache, std::vector<ddm::Index>& dofs, const auto& e) {
    lfs.bind(e);
    cache.update();
    dofs.resize(lfs.size());
    for (std::size_t i = 0; i < lfs.size(); ++i) dofs[i] = cache.containerIndex(i)[0];
  };
  const auto bind_face = [&](const auto& is) {
    bind(lfs_n, cache_n, dofs_n, is.outside());
    face_dofs = dofs_s;
    face_dofs.insert(face_dofs.end(), dofs_n.begin(), dofs_n.end());
  };

  // Sparsity pattern, the same for both matrices: every element couples with itself and its neighbours
  ddm::Pattern pattern(n, n, static_cast<ddm::Index>(gfs.finiteElementMap().maxLocalSize() * (2 * GFS::Traits::GridView::dimension + 1)));
  for (const auto& e : elements(gv, Dune::Partitions::interior)) {
    bind(lfs_s, cache_s, dofs_s, e);
    for (auto i : dofs_s)
      for (auto j : dofs_s) pattern.add(i, j);
    for (const auto& is : intersections(gv, e)) {
      if (!assemble_face(is)) continue;
      bind_face(is);
      for (auto i : face_dofs)
        for (auto j : face_dofs) pattern.add(i, j);
    }
  }
  pattern.finalize();

  // Record the element and face matrices (for GenEO/MsGFEM) unless the config says otherwise
  auto mat_config = config.sub("mat");
  if (!mat_config.hasKey("record_values")) mat_config["record_values"] = "true";
  auto A = ddm::create_mat<RF>(mat_config, pattern, comm);
  auto A_sym = lop_sym != nullptr ? ddm::create_mat<RF>(mat_config, pattern, comm) : nullptr;
  auto b = A->create_range_vector();

  Dune::PDELab::LocalMatrix<RF> local_mat;
  Dune::PDELab::LocalMatrix<RF> mat_ss;
  Dune::PDELab::LocalMatrix<RF> mat_sn;
  Dune::PDELab::LocalMatrix<RF> mat_ns;
  Dune::PDELab::LocalMatrix<RF> mat_nn;
  std::vector<RF> face_mat;
  Dune::PDELab::LocalVector<RF> local_x_s;
  Dune::PDELab::LocalVector<RF> local_x_n;
  Dune::PDELab::LocalVector<RF> local_r;
  {
    auto b_view = b.host_view(ddm::read_write);
    for (const auto& e : elements(gv, Dune::Partitions::interior)) {
      bind(lfs_s, cache_s, dofs_s, e);
      const auto m_s = lfs_s.size();
      Dune::PDELab::ElementGeometry eg(e);

      // Element matrices (the operators are linear, x is not used) and load vector of lop, its residual at x = 0:
      // lambda_volume() adds -int f v, alpha_boundary() the data of the boundary conditions (e.g. the Dirichlet data g
      // in the penalty terms), its terms in u vanish for x = 0. The boundary faces add the terms of the boundary
      // conditions to the element matrices.
      local_x_s.assign(m_s, 0.0);
      local_r.assign(m_s, 0.0);
      auto r_view = local_r.weightedAccumulationView(1.0);
      for (auto [op, mat] : {std::pair{&lop, A.get()}, std::pair{lop_sym, A_sym.get()}}) {
        if (op == nullptr) continue;
        const bool rhs = op == &lop;
        local_mat.assign(m_s, m_s, 0.0);
        auto mat_view = local_mat.weightedAccumulationView(1.0);
        op->jacobian_volume(eg, lfs_s, local_x_s, lfs_s, mat_view);
        if (rhs) op->lambda_volume(eg, lfs_s, r_view);

        unsigned int face = 0;
        for (const auto& is : intersections(gv, e)) {
          Dune::PDELab::IntersectionGeometry ig(is, face++);
          if (is.boundary()) {
            op->jacobian_boundary(ig, lfs_s, local_x_s, lfs_s, mat_view);
            if (rhs) op->alpha_boundary(ig, lfs_s, local_x_s, lfs_s, r_view);
            continue;
          }
          if (!assemble_face(is)) continue;

          // Face matrix: the four blocks, joined to one column-major matrix for face_dofs
          bind_face(is);
          const auto m_n = lfs_n.size();
          local_x_n.assign(m_n, 0.0);
          mat_ss.assign(m_s, m_s, 0.0);
          mat_sn.assign(m_s, m_n, 0.0);
          mat_ns.assign(m_n, m_s, 0.0);
          mat_nn.assign(m_n, m_n, 0.0);
          auto ss_view = mat_ss.weightedAccumulationView(1.0);
          auto sn_view = mat_sn.weightedAccumulationView(1.0);
          auto ns_view = mat_ns.weightedAccumulationView(1.0);
          auto nn_view = mat_nn.weightedAccumulationView(1.0);
          op->jacobian_skeleton(ig, lfs_s, local_x_s, lfs_s, lfs_n, local_x_n, lfs_n, ss_view, sn_view, ns_view, nn_view);

          const auto m = m_s + m_n;
          face_mat.resize(m * m);
          for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < m; ++i)
              face_mat[j * m + i] = i < m_s ? (j < m_s ? mat_ss.getEntry(i, j) : mat_sn.getEntry(i, j - m_s)) : (j < m_s ? mat_ns.getEntry(i - m_s, j) : mat_nn.getEntry(i - m_s, j - m_s));
          mat->add_values(face_dofs, face_dofs, face_mat);
        }
        mat->add_values(dofs_s, dofs_s, local_mat.base()); // both column-major
      }
      for (std::size_t i = 0; i < m_s; ++i) b_view[dofs_s[i]] -= local_r.base()[i];
    }
  }
  A->assemble();
  if (A_sym) {
    A_sym->assemble();
    A->attach_symmetric_matrix(A_sym);
  }

  // The library expects consistent vectors
  A->communication()->reduce(b);
  return {A, std::move(b)};
}
} // namespace ddm::examples
