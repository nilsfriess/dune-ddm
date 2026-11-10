// Helmholtz equation in 2D on the BP 2004 velocity benchmark model with an impedance condition on the whole boundary:
//
//   -div(grad u) - k^2 V^2 u = f in the model,   grad u . n - i k V u = 0 on the boundary,
//
// with k = 2 pi frequency and V = 1 / c, where c is the velocity of the model. Lengths are in km, velocities in km/s and
// the frequency in Hz, so k V = 2 pi frequency / c is the local wavenumber in 1/km. The impedance condition is the first
// order absorbing condition, with it the problem is well posed for every frequency. The source is the Gaussian
// f = exp(-|x - x_s|^2 / w^2) whose width w is a quarter of the wavelength at x_s, so it acts like a point source.
// Conforming P2 elements on a simplex grid.
//
// With problem.free_surface = true, the top of the model is a free surface (pressure release) instead, u = 0 there, as
// in the real benchmark. The impedance condition stays on the other sides. The surface reflects the waves back into the
// model, so the water layer acts as a waveguide and the problem is harder to solve.
//
// The velocity model is read from problem.velocity, an .ini file written by data/segy_to_bin.py from
// vel_z6.25m_x12.5m_exact.segy of the BP 2004 benchmark. The default is data/bp2004_vel.ini in the source tree. The
// model covers [0, 67.4375] x [0, 11.94375] km with the surface at the top. The source x_s is read from problem.source,
// by default it is in the water 0.5 km below the surface in the middle of the model. data/marmousi_vp.ini describes the
// original Marmousi model (9.2 x 3.004 km), its data is downloaded by data/download_models.sh.
//
// problem.extent (in km, default: the size of the model) restricts the domain to the upper left part of the model: the
// domain is [0, extent[0]] x [0, extent[1]], and its top is the surface of the model.
//
// The grid should resolve the shortest wavelength c_min / frequency (c_min = 1.429 km/s) with about 5 elements, i.e. it
// should have about 240 * frequency x 42 * frequency cells. The default is frequency 1 Hz on 240 x 42 cells.
//
// As in ex4, the eigenproblems of MsGFEM use the k-weighted energy form int grad(u) grad(v) + k^2 V^2 u v.
//
// Example: mpirun -np 4 ./ex5 -problem.frequency 1 -grid.cells "240 42" -prec.type tlschwarz -prec.coarse_space_type msgfem

#include "ddm/debug/vec_sink.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"
#include "examples/assembly.hh"
#include "examples/pdelab_communication.hh"
#include "examples/velocity_reader.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/shared_ptr.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/utility/parmetisgridpartitioner.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/pdelab.hh>
#include <iostream>
#include <memory>
#include <numbers>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
// The coefficients of the equation. Only the dofs on a free surface are constrained (u = 0 there).
class BPParameters : public Dune::PDELab::DirichletConstraintsParameters {
public:
  BPParameters(const Dune::ParameterTree& config, ddm::examples::VelocityField velocity_)
      : velocity(std::move(velocity_))
      , frequency(config.get("problem.frequency", 1.))
      , k(2 * std::numbers::pi * frequency)
      , model_size{velocity.nx * velocity.dx / 1000, velocity.nz * velocity.dz / 1000}
      , upper(config.get("problem.extent", model_size))
      , source(config.get("problem.source", Dune::FieldVector<double, 2>{upper[0] / 2, upper[1] - 0.5}))
      , source_width(c(source) / frequency / 4)
      , free_surface(config.get("problem.free_surface", false))
  {
    for (int i = 0; i < 2; ++i)
      DDM_CHECK(upper[i] > 0 && upper[i] <= model_size[i] * (1 + 1e-12), "ex5: problem.extent {} must be positive and at most the model size {} in every direction", upper[i],
                model_size[i]);
  }

  template <typename I>
  bool isDirichlet(const I& intersection, const Dune::FieldVector<typename I::ctype, I::mydimension>& coord) const
  {
    return free_surface && on_top(intersection.geometry().global(coord));
  }

  // Whether x (in km) is on the top of the model, the surface
  bool on_top(const Dune::FieldVector<double, 2>& x) const { return std::abs(x[1] - upper[1]) < 1e-9 * upper[1]; }

  // Velocity in km/s at x (in km), the velocity field is in m/s and m. The top of the domain is the top of the model.
  double c(const Dune::FieldVector<double, 2>& x) const { return velocity(1000 * (x + Dune::FieldVector<double, 2>{0, model_size[1] - upper[1]})) / 1000; }

  double V(const Dune::FieldVector<double, 2>& x) const { return 1 / c(x); }

  double f(const Dune::FieldVector<double, 2>& x) const { return std::exp(-(x - source).two_norm2() / (source_width * source_width)); }

  ddm::examples::VelocityField velocity;
  double frequency;
  double k;
  // Size of the whole velocity model
  Dune::FieldVector<double, 2> model_size;
  // Upper right corner of the domain, the lower left one is the origin
  Dune::FieldVector<double, 2> upper;
  Dune::FieldVector<double, 2> source;
  double source_width;
  // Whether the top is a free surface (u = 0) instead of an impedance boundary
  bool free_surface;
};

// The residual terms alpha_volume() and alpha_boundary() multiply the element matrix with the coefficients (the
// operator is linear and its data is in the lambda terms)
class ConformingHelmholtzLocalOperator : public Dune::PDELab::JacobianBasedAlphaVolume<ConformingHelmholtzLocalOperator>,
                                         public Dune::PDELab::JacobianBasedAlphaBoundary<ConformingHelmholtzLocalOperator>,
                                         public Dune::PDELab::FullVolumePattern,
                                         public Dune::PDELab::LocalOperatorDefaultFlags {
public:
  enum { doAlphaVolume = true };
  enum { doLambdaVolume = true };
  enum { doAlphaBoundary = true };

  // With energy = true, the operator is the k-weighted energy form int grad(u) grad(v) + k^2 V^2 u v instead: the sign
  // of the k^2 term is flipped and there is no impedance term. This form is Hermitian positive definite, its matrix is
  // attached to the matrix as its symmetric part for the eigenproblems of MsGFEM.
  ConformingHelmholtzLocalOperator(const BPParameters& params, bool energy = false)
      : params_(params)
      , energy_(energy)
  {
  }

  // Element matrix of int grad(u) grad(v) - k^2 V^2 u v (+ k^2 V^2 u v for the energy form). The operator is linear, so
  // x is not used.
  template <typename EG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_volume(const EG& eg, const LFSU& lfsu, const X&, const LFSV& lfsv, M& mat) const
  {
    using Basis = typename LFSU::Traits::FiniteElementType::Traits::LocalBasisType;
    using RF = typename Basis::Traits::RangeFieldType;
    constexpr int dim = EG::Entity::dimension;
    const auto sign = energy_ ? -1. : 1.;

    const auto geo = eg.geometry();
    const auto& basis = lfsu.finiteElement().localBasis();
    std::vector<typename Basis::Traits::RangeType> phi(lfsu.size());
    std::vector<typename Basis::Traits::JacobianType> gradphihat(lfsu.size());
    std::vector<Dune::FieldVector<RF, dim>> gradphi(lfsu.size());
    for (const auto& ip : Dune::PDELab::quadratureRule(geo, 2 * basis.order())) {
      basis.evaluateFunction(ip.position(), phi);
      basis.evaluateJacobian(ip.position(), gradphihat);

      // Transform the gradients from the reference element to the element
      const auto S = geo.jacobianInverseTransposed(ip.position());
      for (std::size_t i = 0; i < lfsu.size(); ++i) S.mv(gradphihat[i][0], gradphi[i]);

      const auto V = params_.V(geo.global(ip.position()));
      const auto k2v2 = sign * params_.k * params_.k * V * V;
      const auto factor = ip.weight() * geo.integrationElement(ip.position());
      for (std::size_t i = 0; i < lfsv.size(); ++i)
        for (std::size_t j = 0; j < lfsu.size(); ++j) mat.accumulate(lfsv, i, lfsu, j, ((gradphi[j] * gradphi[i]) - k2v2 * phi[j][0] * phi[i][0]) * factor);
    }
  }

  // Impedance term -i k int V u v on a boundary face (none for the energy form and on a free surface)
  template <typename IG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_boundary(const IG& ig, const LFSU& lfsu, const X&, const LFSV& lfsv, M& mat) const
  {
    if (energy_ || params_.isDirichlet(ig.intersection(), ig.geometry().local(ig.geometry().center()))) return;

    using Basis = typename LFSU::Traits::FiniteElementType::Traits::LocalBasisType;

    const auto geo = ig.geometry();
    const auto geo_in_inside = ig.geometryInInside();
    const auto& basis = lfsu.finiteElement().localBasis();
    std::vector<typename Basis::Traits::RangeType> phi(lfsu.size());
    for (const auto& ip : Dune::PDELab::quadratureRule(geo, 2 * basis.order())) {
      basis.evaluateFunction(geo_in_inside.global(ip.position()), phi);
      const std::complex<double> ikv(0, params_.k * params_.V(geo.global(ip.position())));
      const auto factor = ip.weight() * geo.integrationElement(ip.position());
      for (std::size_t i = 0; i < lfsv.size(); ++i)
        for (std::size_t j = 0; j < lfsu.size(); ++j) mat.accumulate(lfsv, i, lfsu, j, -ikv * phi[j][0] * phi[i][0] * factor);
    }
  }

  // Right-hand side -int f v (PDELab's sign convention: the residual is alpha + lambda)
  template <typename EG, typename LFSV, typename R>
  void lambda_volume(const EG& eg, const LFSV& lfsv, R& r) const
  {
    using Basis = typename LFSV::Traits::FiniteElementType::Traits::LocalBasisType;

    const auto geo = eg.geometry();
    const auto& basis = lfsv.finiteElement().localBasis();
    std::vector<typename Basis::Traits::RangeType> phi(lfsv.size());
    for (const auto& ip : Dune::PDELab::quadratureRule(geo, 2 * basis.order())) {
      basis.evaluateFunction(ip.position(), phi);
      const auto f = params_.f(geo.global(ip.position()));
      const auto factor = ip.weight() * geo.integrationElement(ip.position());
      for (std::size_t i = 0; i < lfsv.size(); ++i) r.accumulate(lfsv, i, -f * phi[i][0] * factor);
    }
  }

private:
  const BPParameters& params_;
  bool energy_;
};

// PDELab constrains the dofs of the Dirichlet faces of the elements of this rank. A vertex on such a face can also belong
// to an element of another rank that touches the face only at this vertex, and that rank does not constrain it. Every
// rank holding a constrained dof must constrain it (see assemble_jacobian()), so this constrains every dof that some
// rank constrains. Collective.
template <class GFS, class CC>
void make_constraints_consistent(const GFS& gfs, CC& cc)
{
  using Dune::PDELab::Backend::native;
  using V = Dune::PDELab::Backend::Vector<GFS, int>;
  V constrained(gfs, 0);
  for (const auto& [ci, row] : cc) native(constrained)[ci[0]][0] = 1;
  Dune::PDELab::MaxDataHandle<GFS, V> handle(gfs, constrained);
  gfs.gridView().communicate(handle, Dune::InteriorBorder_InteriorBorder_Interface, Dune::ForwardCommunication);

  // An empty row is a Dirichlet constraint, the container indices come from the local function spaces
  using LFS = Dune::PDELab::LocalFunctionSpace<GFS>;
  LFS lfs(gfs);
  Dune::PDELab::LFSIndexCache<LFS> cache(lfs);
  for (const auto& e : elements(gfs.gridView(), Dune::Partitions::interior)) {
    lfs.bind(e);
    cache.update();
    for (std::size_t i = 0; i < cache.size(); ++i)
      if (native(constrained)[cache.containerIndex(i)[0]][0] == 1) cc[cache.containerIndex(i)];
  }
}

// The modulus (angle = false) or the angle in (-pi, pi] (angle = true) of the complex function re + i im for the
// VTKWriter. Both are computed at the points where the writer evaluates the function, from the values of re and im
// there. Interpolating them from the dofs would be wrong: the modulus is not a P2 function, and the angle jumps by 2 pi.
template <class DGF>
class PolarVTKFunction : public Dune::VTKFunction<typename DGF::Traits::GridViewType> {
  using GV = typename DGF::Traits::GridViewType;
  using Entity = typename GV::template Codim<0>::Entity;

public:
  PolarVTKFunction(std::shared_ptr<const DGF> re, std::shared_ptr<const DGF> im, bool angle, std::string name)
      : re_(std::move(re))
      , im_(std::move(im))
      , angle_(angle)
      , name_(std::move(name))
  {
  }

  int ncomps() const override { return 1; }

  double evaluate(int, const Entity& e, const Dune::FieldVector<typename GV::ctype, GV::dimension>& xi) const override
  {
    typename DGF::Traits::RangeType re;
    typename DGF::Traits::RangeType im;
    re_->evaluate(e, xi, re);
    im_->evaluate(e, xi, im);
    const std::complex<double> z(re[0], im[0]);
    return angle_ ? std::arg(z) : std::abs(z);
  }

  std::string name() const override { return name_; }

private:
  std::shared_ptr<const DGF> re_;
  std::shared_ptr<const DGF> im_;
  bool angle_;
  std::string name_;
};

void run(const Dune::ParameterTree& config, const Dune::MPIHelper& helper)
{
  BPParameters params(config, ddm::examples::read_velocity_field(config.get("problem.velocity", std::string(DDM_DATA_DIR "/bp2004_vel.ini"))));

  // Create the grid: the rectangle covered by the velocity model
  constexpr int dim = 2;
  const auto cells = config.get("grid.cells", std::array<unsigned int, dim>{240, 42});
  auto grid = Dune::StructuredGridFactory<Dune::UGGrid<dim>>::createSimplexGrid({0., 0.}, params.upper, cells);
  const auto gv = grid->leafGridView();
  using GV = std::remove_cvref_t<decltype(gv)>;
  auto part = Dune::ParMetisGridPartitioner<GV>::partition(gv, helper);
  grid->loadBalance(part, 0);
  const auto refine = config.get("grid.refine", 0);
  grid->globalRefine(refine);

  if (gv.comm().rank() == 0) {
    const auto c_min = *std::ranges::min_element(params.velocity.data) / 1000;
    const auto h = std::max(params.upper[0] / cells[0], params.upper[1] / cells[1]) / (1 << refine);
    std::cout << "Frequency " << params.frequency << " Hz, " << c_min / params.frequency / h << " elements per shortest wavelength, source at "
              << params.source << " (velocity " << params.c(params.source) << " km/s, width " << params.source_width << " km)"
              << (params.free_surface ? ", free surface at the top" : "") << "\n";
  }

  // Define the types that PDELab needs to assemble the linear system and rhs
  using RF = std::complex<double>;
  constexpr int degree = 2;
  using ES = Dune::PDELab::OverlappingEntitySet<GV>;
  // The basis functions are real (dune-localfunctions' Lagrange basis does not support complex range types), only the
  // coefficients are complex
  using FEM = Dune::PDELab::PkLocalFiniteElementMap<ES, double, double, degree>;
  using GFS = Dune::PDELab::GridFunctionSpace<ES, FEM, Dune::PDELab::ConformingDirichletConstraints, Dune::PDELab::ISTL::VectorBackend<>>;

  ES es(gv);
  FEM fem(es);
  GFS gfs(es, fem);
  gfs.name("solution");

  using CC = typename GFS::template ConstraintsContainer<RF>::Type;
  CC cc;
  Dune::PDELab::constraints(params, gfs, cc);
  make_constraints_consistent(gfs, cc);
  ConformingHelmholtzLocalOperator lop(params);
  ConformingHelmholtzLocalOperator lop_energy(params, true);

  // Only the grid loops of the grid operators are used, PDELab assembles no matrix (the matrix backend is part of the
  // type)
  using MBE = Dune::PDELab::ISTL::BCRSMatrixBackend<>;
  using GO = Dune::PDELab::GridOperator<GFS, GFS, ConformingHelmholtzLocalOperator, MBE, RF, RF, RF, CC, CC>;
  GO go(gfs, cc, gfs, cc, lop, MBE(19));
  GO go_energy(gfs, cc, gfs, cc, lop_energy, MBE(19));

  // As in PDELab, we solve for a correction: with the Jacobian A and the residual r at x0 = 0, A d = r gives the
  // solution x0 - d. The energy matrix is attached to A.
  GO::Traits::Domain x0(gfs, 0.0);
  auto comm = make_communication(gfs);
  auto A = ddm::examples::assemble_jacobian(config, go, x0, comm);
  A->attach_symmetric_matrix(ddm::examples::assemble_jacobian(config, go_energy, x0, comm));
  auto r = ddm::examples::assemble_residual(go, x0, *A);

  // Define the VTKWriter already here, because the preconditioner might write something to the debug sink. VTK has no
  // complex numbers, so every vector is written as its real and imaginary part, each subsampled so that the P2 functions
  // show. With polar = true, also its modulus (the amplitude of the wave) and its angle (the phase, its jumps are the
  // wavefronts) are written. The debug vectors are written without them.
  using X = Dune::PDELab::Backend::Vector<GFS, double>;
  using DGF = Dune::PDELab::DiscreteGridFunction<GFS, X>;
  Dune::SubsamplingVTKWriter<GV> writer(gv, Dune::refinementIntervals(degree));
  const auto add_to_writer = [&](std::string_view name, const ddm::Vec<RF>& v, bool polar = false) {
    auto re = std::make_shared<X>(gfs, 0.0);
    auto im = std::make_shared<X>(gfs, 0.0);
    {
      const auto view = v.host_view(ddm::read);
      auto& native_re = Dune::PDELab::Backend::native(*re);
      auto& native_im = Dune::PDELab::Backend::native(*im);
      for (std::size_t i = 0; i < view.size(); ++i) {
        native_re[i][0] = view[i].real();
        native_im[i][0] = view[i].imag();
      }
    }
    auto dgf_re = std::make_shared<const DGF>(Dune::stackobject_to_shared_ptr(gfs), re);
    auto dgf_im = std::make_shared<const DGF>(Dune::stackobject_to_shared_ptr(gfs), im);
    for (const auto& [dgf, suffix] : {std::pair{dgf_re, "_re"}, std::pair{dgf_im, "_im"}})
      writer.addVertexData(std::make_shared<Dune::PDELab::VTKGridFunctionAdapter<DGF>>(dgf, std::string(name) + suffix));
    if (polar) {
      writer.addVertexData(std::make_shared<PolarVTKFunction<DGF>>(dgf_re, dgf_im, false, std::string(name) + "_abs"));
      writer.addVertexData(std::make_shared<PolarVTKFunction<DGF>>(dgf_re, dgf_im, true, std::string(name) + "_angle"));
    }
  };
  ddm::debug::VecSink<RF> sink(config.get("debug_rank", 0), add_to_writer);

  // Now the problem is fully assembled and we can setup the preconditioner and solver and solve it
  auto P = ddm::create_prec<RF>(config.sub("prec"), A);
  auto solver = ddm::create_solver<RF>(config.sub("solver"), A, P);
  auto d = A->create_domain_vector();
  solver->solve(r, d);
  auto x = A->create_domain_vector();
  ddm::examples::copy(x0, x);
  x -= d;

  // VTK output of the solution and of the velocity at the element centres
  add_to_writer(gfs.name(), x, true);
  std::vector<double> velocity(gv.size(0));
  for (const auto& e : elements(gv)) velocity[gv.indexSet().index(e)] = params.c(e.geometry().center());
  // addCellData() with all arguments, the short form is ambiguous for SubsamplingVTKWriter
  writer.addCellData(velocity, "velocity", 1, Dune::VTK::Precision::float32);
  std::vector<int> ranks(gv.size(0), gv.comm().rank());
  writer.addCellData(ranks, "Rank", 1, Dune::VTK::Precision::int32);
  writer.write("ex5");
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
