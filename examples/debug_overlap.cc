#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "dune/ddm/datahandles.hh"
#include "dune/ddm/overlap_extension.hh"
#include "dune/ddm/pdelab_helper.hh"

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/pdelab/backend/istl.hh>
#include <dune/pdelab/constraints/conforming.hh>
#include <dune/pdelab/finiteelementmap/qkfem.hh>
#include <dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include <dune/pdelab/gridfunctionspace/vtk.hh>
#include <dune/pdelab/gridoperator/gridoperator.hh>
#include <dune/pdelab/localoperator/convectiondiffusiondg.hh>
#include <dune/pdelab/localoperator/convectiondiffusionfem.hh>
#include <dune/pdelab/localoperator/convectiondiffusionparameter.hh>

template <typename GV, typename RF>
struct MyTraits {
  using GridViewType = GV;
  using ElementType = typename GV::template Codim<0>::Entity;
  using DomainType = typename GV::template Codim<0>::Entity::Geometry::GlobalCoordinate;
  using RangeType = RF; // Scalar!
  using RangeFieldType = RF;
  using IntersectionType = typename GV::Intersection;
  using IntersectionDomainType = typename IntersectionType::Geometry::LocalCoordinate;
  using PermTensorType = Dune::FieldMatrix<RF, GV::dimension, GV::dimension>;
  static const int dimDomain = GV::dimension;
  using BoundaryConditionType = Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type;
};

// Simple parameter class for Poisson problem
template <typename GV, typename RF>
class Problem {
public:
  using Traits = MyTraits<GV, RF>;

  Problem() = default;

  typename Traits::PermTensorType A(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    typename Traits::PermTensorType I;
    for (std::size_t i = 0; i < Traits::dimDomain; ++i)
      for (std::size_t j = 0; j < Traits::dimDomain; ++j) I[i][j] = (i == j) ? 1.0 : 0.0;
    return I;
  }

  typename Traits::DomainType b(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return typename Traits::DomainType(0.0); }

  typename Traits::RangeType c(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return typename Traits::RangeType(0.0); }

  typename Traits::RangeType f(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const
  {
    return typename Traits::RangeType(0.0); // RHS doesn't matter for matrix structure
  }

  typename Traits::RangeType g(const typename Traits::ElementType& e, const typename Traits::DomainType& x) const { return typename Traits::RangeType(0.0); }

  Dune::PDELab::ConvectionDiffusionBoundaryConditions::Type bctype(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const
  {
    return Dune::PDELab::ConvectionDiffusionBoundaryConditions::Dirichlet;
  }

  typename Traits::RangeType j(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const { return typename Traits::RangeType(0.0); }

  typename Traits::RangeType o(const typename Traits::IntersectionType& is, const typename Traits::IntersectionDomainType& x) const { return typename Traits::RangeType(0.0); }

  bool permeabilityIsConstantPerCell() const { return true; }
};

int main(int argc, char** argv)
{
  auto& helper = Dune::MPIHelper::instance(argc, argv);

  // 1. Grid Setup (YaspGrid with overlap 1)
  constexpr int dim = 2;
  using Grid = Dune::YaspGrid<dim>;
  Dune::FieldVector<double, dim> L = {1.0, 1.0};
  std::array<int, dim> s = {64, 64};
  std::bitset<dim> periodic(0);
  int overlap = 1;
  auto grid = std::make_shared<Grid>(L, s, periodic, overlap, helper.getCommunicator());

  using GV = Grid::LeafGridView;
  GV gv = grid->leafGridView();

  // 2. GFS
  using DF = Grid::ctype;
  using RF = double;
  using FEM = Dune::PDELab::QkLocalFiniteElementMap<GV, DF, RF, 1>;
  FEM fem(gv);
  using CON = Dune::PDELab::ConformingDirichletConstraints;
  using VBE = Dune::PDELab::ISTL::VectorBackend<Dune::PDELab::ISTL::Blocking::none>;
  using GFS = Dune::PDELab::GridFunctionSpace<GV, FEM, CON, VBE>;
  GFS gfs(gv, fem);
  gfs.name("Vh");

  // 3. Communication (Non-overlapping view, but includes ghosts if grid has them)
  auto novlp_comm = make_communication(gfs);

  // 4. Local Matrix Assembly
  using ProblemType = Problem<GV, RF>;
  ProblemType problem;
  using LOP = Dune::PDELab::ConvectionDiffusionDG<ProblemType, FEM>;
  LOP lop(problem);

  using CC = Dune::PDELab::EmptyTransformation;
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, LOP, MBE, RF, RF, RF, CC, CC>;
  GO go(gfs, gfs, lop, {9});

  using Matrix = GO::Jacobian;
  using Vector = GO::Domain;
  Matrix As(go);
  Vector x(gfs, 0.0);
  go.jacobian(x, As);

  // 5. Extend Overlap (Algebraically)
  // We extend by 1 layer. Since grid already has 1 layer, this might extend to 2 layers effectively?
  // Or if novlp_comm includes ghosts, and we extend by 1, we get 1 more layer.
  int extension_layers = 1;
  const auto& nAs = Dune::PDELab::Backend::native(As);
  auto [ovlp_comm, ext_mask] = make_overlapping_communication(*novlp_comm, nAs, extension_layers);

  // 6. Build Matrix on Extended Domain
  // We need to communicate the matrix entries to the new extended communicator.
  // Create VariableSizeCommunicator for the extended comm
  typename std::decay_t<decltype(*ovlp_comm)>::AllSet allset;
  Dune::Interface interface_ext;
  interface_ext.build(ovlp_comm->remoteIndices(), allset, allset);
  Dune::VariableSizeCommunicator varcomm(interface_ext);

  using NativeMat = Dune::PDELab::Backend::Native<Matrix>;
  CreateMatrixDataHandle cmdh(nAs, ovlp_comm->indexSet());
  varcomm.forward(cmdh);
  auto A_dir = std::make_shared<NativeMat>(cmdh.getOverlappingMatrix());

  // 7. Identify Boundary on Extended Domain
  IdentifyBoundaryDataHandle ibdh(*A_dir, ovlp_comm->indexSet());
  varcomm.forward(ibdh);

  const auto& boundary_mask = ibdh.get_boundary_mask();

  // 8. Visualize
  std::cout << "Preparing visualization..." << std::endl;

  using NativeVec = Dune::PDELab::Backend::Native<Vector>;
  NativeVec boundary_vec(ovlp_comm->indexSet().size());
  for (size_t i = 0; i < boundary_mask.size(); ++i) boundary_vec[i] = boundary_mask[i] ? 1.0 : 0.0;

  Dune::SubsamplingVTKWriter<GV> vtkwriter(gv, Dune::refinementIntervals(1));

  auto write_overlapping_vector = [&](const NativeVec& vec, const std::string& name) {
    // Convert to std::vector for communication to avoid BlockVector issues
    std::vector<double> vec_std(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) vec_std[i] = vec[i][0];

    // Zero out on all ranks except debug rank (say rank 0)
    if (helper.rank() != 0) std::fill(vec_std.begin(), vec_std.end(), 0.0);

    ovlp_comm->addOwnerCopyToAll(vec_std, vec_std);

    // Allocate on heap to ensure lifetime matches vtkwriter
    auto vis_vec_gfs = std::make_shared<Vector>(gfs, 0.0);
    auto& vis_vec_native = Dune::PDELab::Backend::native(*vis_vec_gfs);

    for (std::size_t i = 0; i < vis_vec_native.size(); ++i)
      if (i < vec_std.size()) vis_vec_native[i] = vec_std[i];

    using DGF = Dune::PDELab::DiscreteGridFunction<GFS, Vector>;
    auto dgf = std::make_shared<DGF>(Dune::stackobject_to_shared_ptr(gfs), vis_vec_gfs);
    vtkwriter.addVertexData(std::make_shared<Dune::PDELab::VTKGridFunctionAdapter<DGF>>(dgf, name));
  };

  write_overlapping_vector(boundary_vec, "boundary_mask");
  vtkwriter.write("debug_overlap", Dune::VTK::appendedraw);

  std::cout << "Done." << std::endl;

  return 0;
}
