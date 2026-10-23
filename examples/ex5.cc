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
// The velocity model is read from problem.velocity, an .ini file written by data/segy_to_bin.py from
// vel_z6.25m_x12.5m_exact.segy of the BP 2004 benchmark. The default is data/bp2004_vel.ini in the source tree. The
// model covers [0, 67.4375] x [0, 11.94375] km with the surface at the top. The source x_s is read from problem.source,
// by default it is in the water 0.5 km below the surface in the middle of the model.
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
// The coefficients of the equation. There are no Dirichlet conditions, so no dof is constrained.
class BPParameters : public Dune::PDELab::DirichletConstraintsParameters {
public:
  BPParameters(const Dune::ParameterTree& config, ddm::examples::VelocityField velocity_)
      : velocity(std::move(velocity_))
      , frequency(config.get("problem.frequency", 1.))
      , k(2 * std::numbers::pi * frequency)
      , upper{velocity.nx * velocity.dx / 1000, velocity.nz * velocity.dz / 1000}
      , source(config.get("problem.source", Dune::FieldVector<double, 2>{upper[0] / 2, upper[1] - 0.5}))
      , source_width(c(source) / frequency / 4)
  {
  }

  template <typename I>
  bool isDirichlet(const I&, const Dune::FieldVector<typename I::ctype, I::mydimension>&) const
  {
    return false;
  }

  // Velocity in km/s at x (in km), the velocity field is in m/s and m
  double c(const Dune::FieldVector<double, 2>& x) const { return velocity(1000 * x) / 1000; }

  double V(const Dune::FieldVector<double, 2>& x) const { return 1 / c(x); }

  double f(const Dune::FieldVector<double, 2>& x) const { return std::exp(-(x - source).two_norm2() / (source_width * source_width)); }

  ddm::examples::VelocityField velocity;
  double frequency;
  double k;
  // Upper right corner of the model, the lower left one is the origin
  Dune::FieldVector<double, 2> upper;
  Dune::FieldVector<double, 2> source;
  double source_width;
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

  // Impedance term -i k int V u v on a boundary face (none for the energy form)
  template <typename IG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_boundary(const IG& ig, const LFSU& lfsu, const X&, const LFSV& lfsv, M& mat) const
  {
    if (energy_) return;

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
              << params.source << " (velocity " << params.c(params.source) << " km/s, width " << params.source_width << " km)\n";
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
  // show.
  using X = Dune::PDELab::Backend::Vector<GFS, double>;
  using DGF = Dune::PDELab::DiscreteGridFunction<GFS, X>;
  Dune::SubsamplingVTKWriter<GV> writer(gv, Dune::refinementIntervals(degree));
  const auto add_to_writer = [&](std::string_view name, const ddm::Vec<RF>& v) {
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
    for (const auto& [u, suffix] : {std::pair{re, "_re"}, std::pair{im, "_im"}}) {
      auto dgf = std::make_shared<DGF>(Dune::stackobject_to_shared_ptr(gfs), u);
      writer.addVertexData(std::make_shared<Dune::PDELab::VTKGridFunctionAdapter<DGF>>(dgf, std::string(name) + suffix));
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
  add_to_writer(gfs.name(), x);
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
