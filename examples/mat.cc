#include <algorithm>
#include <cstddef>
#include <ddm/communication.hh>
#include <ddm/debug/vec_sink.hh>
#include <ddm/index.hh>
#include <ddm/mat/mat.hh>
#include <ddm/mat/pattern.hh>
#include <ddm/prec/prec.hh>
#include <ddm/solver/solver.hh>
#include <deque>
#include <dune/common/exceptions.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/referenceelements.hh>
#include <dune/grid/common/partitionset.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/globalindexset.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/localfunctions/lagrange/lagrangelfecache.hh>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/* @brief A data handle to compute a minimum across entities of a parallel grid
 */
template <class GridView, class V>
class MinDataHandle : public Dune::CommDataHandleIF<MinDataHandle<GridView, V>, typename V::value_type> {
public:
  using DataType = typename V::value_type;

  MinDataHandle(const GridView& gv, V& v, int codim)
      : gv_(gv)
      , v_(v)
      , codim_(codim)
  {
  }

  bool contains(int /*dim*/, int codim) const { return codim == codim_; }
  bool fixedSize(int /*dim*/, int /*codim*/) const { return true; }

  template <class Entity>
  std::size_t size(const Entity&) const
  {
    return 1;
  }

  template <class Buffer, class Entity>
  void gather(Buffer& buf, const Entity& e) const
  {
    buf.write(v_[gv_.indexSet().index(e)]);
  }

  template <class Buffer, class Entity>
  void scatter(Buffer& buf, const Entity& e, std::size_t)
  {
    DataType x;
    buf.read(x);
    auto& mine = v_[gv_.indexSet().index(e)];
    if (x >= 0 && (mine < 0 || x < mine)) mine = x; // -1 means "I can't own this"
  }

private:
  const GridView& gv_;
  V& v_;
  int codim_;
};

template <class GridView>
std::shared_ptr<ddm::Communication> create_communication(const GridView& gv)
{
  const auto dim = GridView::dimension;
  const int rank = gv.comm().rank();
  Dune::GlobalIndexSet gidxs(gv, dim);

  // Unfortunately, GlobalIndexSet doesn't expose ownership data (even though it computes it internally),
  // so we need to compute it again here
  std::vector<int> owner(gv.size(dim), -1);
  for (const auto& v : vertices(gv)) {
    const auto pt = v.partitionType();
    if (pt == Dune::InteriorEntity || pt == Dune::BorderEntity) owner[gv.indexSet().index(v)] = rank;
  }
  MinDataHandle handle(gv, owner, dim);
  gv.communicate(handle, Dune::All_All_Interface, Dune::ForwardCommunication);

  std::vector<ddm::CommunicationNodes> nodes(gv.size(dim));
  for (const auto& v : vertices(gv)) {
    const auto idx = gv.indexSet().index(v);

    nodes[idx].rank = owner[idx];
    nodes[idx].gid = gidxs.index(v);
  }

  return std::make_shared<ddm::Communication>(gv.comm(), nodes);
}

// High-contrast coefficient on the unit square: kappa = contrast on some cells of a 16 x 16 lattice, 1 elsewhere. The
// pattern consists of four horizontal channels with alternating offsets, which cross many subdomain interfaces but end
// before the Dirichlet boundary, and small square inclusions between them. It is resolved exactly if grid.cells is a
// multiple of 16. Config: contrast (default 1e6), contrast = 1 gives the Laplacian.
template <class T>
auto make_coefficient(const Dune::ParameterTree& config)
{
  return [contrast = config.get("contrast", T(1e6))](const auto& x) -> T {
    const int i = std::clamp(static_cast<int>(16 * x[0]), 0, 15); // lattice column
    const int j = std::clamp(static_cast<int>(16 * x[1]), 0, 15); // lattice row
    if (i == 0 || i == 15 || j == 0 || j == 15) return T{1};

    const bool channel = j % 4 == 2 && (j % 8 == 2 ? i <= 11 : i >= 4);
    const bool inclusion = j % 4 == 0 && (i + j / 4) % 3 == 0;
    return channel || inclusion ? contrast : T{1};
  };
}

// Assembles the stiffness matrix and load vector of -div(kappa grad u) = 1 (kappa from make_coefficient(), evaluated at
// the element centres) with P1/Q1 elements and homogeneous Dirichlet conditions on the whole boundary (rows of boundary
// vertices are replaced by identity rows, with zero right-hand side). Works for any element type that has a first order
// Lagrange element, so also for simplex and mixed grids.
template <class T, class GridView>
std::pair<std::shared_ptr<ddm::Mat<T>>, ddm::Vec<T>> assemble(const Dune::ParameterTree& config, const GridView& gv)
{
  Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Example", "assemble")};

  constexpr int dim = GridView::dimension;
  using ctype = typename GridView::ctype;
  const auto& indexset = gv.indexSet();
  const auto n = static_cast<ddm::Index>(indexset.size(dim));

  // Provides the local finite element matching the geometry type of each element
  Dune::LagrangeLocalFiniteElementCache<ctype, T, dim, 1> fe_cache;

  // Global indices of the element's dofs, in the order of the local basis
  std::vector<ddm::Index> dofs;
  auto get_dofs = [&](const auto& e, const auto& fe) {
    const auto& coeffs = fe.localCoefficients();
    dofs.resize(coeffs.size());
    for (std::size_t i = 0; i < dofs.size(); ++i) dofs[i] = static_cast<ddm::Index>(indexset.subIndex(e, coeffs.localKey(i).subEntity(), dim));
  };

  // Create sparsity pattern
  const auto avg_nnz = 9;
  ddm::Pattern pattern(n, n, avg_nnz);
  for (const auto& e : elements(gv)) {
    get_dofs(e, fe_cache.get(e.type()));
    for (auto i : dofs)
      for (auto j : dofs) pattern.add(i, j);
  }
  pattern.finalize();

  // Assemble matrix and right-hand side
  auto comm = create_communication(gv);
  auto A = ddm::create_mat<T>(config.sub("mat"), pattern, std::move(comm));
  auto b = A->create_range_vector();
  auto b_view = b.host_view(ddm::write); // write: the previous contents are not loaded, so every entry must be set
  std::ranges::fill(b_view, T{0});

  const T f = 1;
  const auto kappa = make_coefficient<T>(config);
  std::vector<T> local;
  std::vector<Dune::FieldMatrix<T, 1, dim>> ref_gradients;
  std::vector<Dune::FieldVector<T, dim>> gradients;
  std::vector<Dune::FieldVector<T, 1>> values;
  for (const auto& e : elements(gv, Dune::Partitions::interior)) {
    const auto& fe = fe_cache.get(e.type());
    const auto& basis = fe.localBasis();
    const auto ndofs = basis.size();
    get_dofs(e, fe);
    local.assign(ndofs * ndofs, T{0});
    gradients.resize(ndofs);

    const auto geo = e.geometry();
    const T kappa_e = kappa(geo.center());
    for (const auto& qp : Dune::QuadratureRules<ctype, dim>::rule(e.type(), 2)) {
      const auto& jit = geo.jacobianInverseTransposed(qp.position());
      const auto factor = qp.weight() * geo.integrationElement(qp.position());

      basis.evaluateJacobian(qp.position(), ref_gradients);
      for (std::size_t i = 0; i < ndofs; ++i) jit.mv(ref_gradients[i][0], gradients[i]);

      for (std::size_t i = 0; i < ndofs; ++i)
        for (std::size_t j = 0; j < ndofs; ++j) local[i * ndofs + j] += factor * kappa_e * (gradients[i] * gradients[j]);

      basis.evaluateFunction(qp.position(), values);
      for (std::size_t i = 0; i < ndofs; ++i) b_view[static_cast<std::size_t>(dofs[i])] += factor * f * values[i][0];
    }

    A->add_values(dofs, dofs, local);
  }
  A->assemble();

  // Dirichlet conditions on the whole boundary: all vertices of boundary intersections. Vertices on several boundary
  // faces are added more than once, which zero_rows() does not mind.
  std::vector<ddm::Index> boundary;
  for (const auto& e : elements(gv)) {
    if (!e.hasBoundaryIntersections()) continue;
    const auto ref = Dune::referenceElement(e.geometry());
    for (const auto& is : intersections(gv, e)) {
      if (!is.boundary()) continue;
      const auto face = is.indexInInside();
      for (int k = 0; k < ref.size(face, 1, dim); ++k) boundary.push_back(static_cast<ddm::Index>(indexset.subIndex(e, ref.subEntity(face, 1, k, dim), dim)));
    }
  }
  A->zero_rows(boundary);
  for (auto i : boundary) b_view[static_cast<std::size_t>(i)] = T{0};
  b_view.release(); // b is moved out below, which is not allowed while the view is open

  // b is additive at this point, so we need to make it consistent
  A->communication()->reduce(b);

  return {A, std::move(b)};
}

template <class Scalar, class GridView>
void solve(const Dune::ParameterTree& config, const GridView& gv)
{
  // Assembles A and b, both are additive after this; the library expects all vector to be consistent
  auto [A, b] = assemble<Scalar>(config, gv);

  std::deque<std::vector<Scalar>> debug_data;
  Dune::VTKWriter writer(gv);
  ddm::debug::VecSink<Scalar> sink(config.get("debug_rank", 0), [&](std::string_view name, const ddm::Vec<Scalar>& v) {
    const auto view = v.host_view(ddm::read);
    writer.addVertexData(debug_data.emplace_back(view.begin(), view.end()), std::string(name));
  });

  auto P = ddm::create_prec<Scalar>(config.sub("prec"), A);
  auto solver = ddm::create_solver<Scalar>(config.sub("solver"), A, P);

  auto x = A->create_domain_vector();
  solver->solve(b, x);

  const auto view = x.host_view(ddm::read);
  writer.addVertexData(view, "solution");

  std::vector<int> rank_number(gv.size(0), A->communication()->rank());
  writer.addCellData(rank_number, "Rank");

  const auto kappa = make_coefficient<Scalar>(config);
  std::vector<Scalar> kappa_cells(gv.size(0));
  for (const auto& e : elements(gv)) kappa_cells[gv.indexSet().index(e)] = kappa(e.geometry().center());
  writer.addCellData(kappa_cells, "kappa");

  writer.write("mat");
}

int main(int argc, char** argv)
{
  try {
    const auto& mpi_helper = Dune::MPIHelper::instance(argc, argv);
    Dune::ParameterTree config;
    Dune::ParameterTreeParser parser;
    parser.readOptions(argc, argv, config);

    using Scalar = double;

    // Unit square, either a structured cube grid (yasp) or an unstructured triangle grid (ug)
    constexpr int dim = 2;
    const auto cells = config.get("grid.cells", 16);
    const auto type = config.get("grid.type", std::string("yasp"));
    if (type == "yasp") {
      Dune::YaspGrid<dim> grid({1.0, 1.0}, {cells, cells}, 0ULL, 0);
      solve<Scalar>(config, grid.leafGridView());
    }
    else if (type == "ug") {
      using Grid = Dune::UGGrid<dim>;
      const auto ucells = static_cast<unsigned int>(cells);
      auto grid = Dune::StructuredGridFactory<Grid>::createSimplexGrid({0.0, 0.0}, {1.0, 1.0}, {ucells, ucells});
      auto gv = grid->leafGridView();
      auto part = Dune::ParMetisGridPartitioner<decltype(gv)>::partition(gv, mpi_helper);
      grid->loadBalance(part, 0);
      solve<Scalar>(config, gv);

      Logger::get().report(MPI_COMM_WORLD);
    }
    else {
      DUNE_THROW(Dune::Exception, "unknown grid.type '" << type << "', expected yasp or ug");
    }
  }
  catch (std::exception& e) {
    std::cout << e.what() << "\n";
    return 1;
  }
  catch (...) {
    std::cout << "Unknown exception thrown\n";
    return 1;
  }
}
