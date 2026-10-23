// Helmholtz equation in 2D with an impedance condition on the whole boundary:
//
//   -div(A grad u) - k^2 V^2 u = f on the unit square,   A grad u . n - i k beta u = g on the boundary,
//
// with A = V = beta = 1 and f = 0. Conforming P2 elements on a simplex grid. The wavenumber k is read from problem.k.
// The boundary data g is chosen such that the exact solution is the plane wave u = exp(i k d . x) with d = (0.6, 0.8)
// (it solves the homogeneous equation since |d| = 1), as in the frequency scaling test of arXiv:2409.06533. The L2 error
// of the discrete solution is printed at the end.
//
// As in that paper, the eigenproblems of MsGFEM use the k-weighted energy form int A grad(u) grad(v) + k^2 V^2 u v. Its
// matrix is assembled with a second grid operator and attached to the matrix (see Mat::attach_symmetric_matrix());
// harmonicity is still defined by the Helmholtz matrix.
//
// Example: mpirun -np 4 ./ex4 -grid.cells "32 32" -problem.k 10 -prec.type tlschwarz -prec.coarse_space_type msgfem

#include "ddm/debug/vec_sink.hh"
#include "ddm/mat/mat.hh"
#include "ddm/prec/prec.hh"
#include "ddm/solver/solver.hh"
#include "ddm/vec/vec.hh"
#include "examples/assembly.hh"
#include "examples/pdelab_communication.hh"

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
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
// The coefficients of the equation. There are no Dirichlet conditions, so no dof is constrained.
class ConstantParameters : public Dune::PDELab::DirichletConstraintsParameters {
public:
  explicit ConstantParameters(double k_)
      : k(k_)
  {
  }

  template <typename I>
  bool isDirichlet(const I&, const Dune::FieldVector<typename I::ctype, I::mydimension>&) const
  {
    return false;
  }

  double k;
  double A = 1;
  double V = 1;
  double beta = 1;
  double f = 0;
};

// Direction of the plane wave in the exact solution
const Dune::FieldVector<double, 2> direction{0.6, 0.8};

// Exact solution for A = V = beta = 1 and f = 0
std::complex<double> exact_solution(const ConstantParameters& params, const Dune::FieldVector<double, 2>& x)
{
  return std::exp(std::complex<double>(0, params.k * (direction * x)));
}

// Boundary data g = grad(u) . n - i k u of the exact solution at x, n is the unit outer normal
std::complex<double> impedance_data(const ConstantParameters& params, const Dune::FieldVector<double, 2>& x, const Dune::FieldVector<double, 2>& n)
{
  const std::complex<double> ik(0, params.k);
  const auto grad_u_n = ik * (direction * n) * std::exp(ik * (direction * x));
  return grad_u_n - ik * exact_solution(params, x);
}

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
  enum { doLambdaBoundary = true };

  // With energy = true, the operator is the k-weighted energy form int A grad(u) grad(v) + k^2 V^2 u v instead: the sign
  // of the k^2 term is flipped and there is no impedance term. This form is Hermitian positive definite, its matrix is
  // attached to the matrix as its symmetric part for the eigenproblems of MsGFEM.
  ConformingHelmholtzLocalOperator(const ConstantParameters& params, bool energy = false)
      : params_(params)
      , energy_(energy)
  {
  }

  // Element matrix of int A grad(u) grad(v) - k^2 V^2 u v (+ k^2 V^2 u v for the energy form). The operator is linear,
  // so x is not used.
  template <typename EG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_volume(const EG& eg, const LFSU& lfsu, const X&, const LFSV& lfsv, M& mat) const
  {
    using Basis = typename LFSU::Traits::FiniteElementType::Traits::LocalBasisType;
    using RF = typename Basis::Traits::RangeFieldType;
    constexpr int dim = EG::Entity::dimension;
    const auto k2v2 = (energy_ ? -1. : 1.) * params_.k * params_.k * params_.V * params_.V;

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

      const auto factor = ip.weight() * geo.integrationElement(ip.position());
      for (std::size_t i = 0; i < lfsv.size(); ++i)
        for (std::size_t j = 0; j < lfsu.size(); ++j) mat.accumulate(lfsv, i, lfsu, j, (params_.A * (gradphi[j] * gradphi[i]) - k2v2 * phi[j][0] * phi[i][0]) * factor);
    }
  }

  // Impedance term -i k beta int u v on a boundary face (none for the energy form)
  template <typename IG, typename LFSU, typename X, typename LFSV, typename M>
  void jacobian_boundary(const IG& ig, const LFSU& lfsu, const X&, const LFSV& lfsv, M& mat) const
  {
    if (energy_) return;

    using Basis = typename LFSU::Traits::FiniteElementType::Traits::LocalBasisType;
    const std::complex<double> ikbeta(0, params_.k * params_.beta);

    const auto geo = ig.geometry();
    const auto geo_in_inside = ig.geometryInInside();
    const auto& basis = lfsu.finiteElement().localBasis();
    std::vector<typename Basis::Traits::RangeType> phi(lfsu.size());
    for (const auto& ip : Dune::PDELab::quadratureRule(geo, 2 * basis.order())) {
      basis.evaluateFunction(geo_in_inside.global(ip.position()), phi);
      const auto factor = ip.weight() * geo.integrationElement(ip.position());
      for (std::size_t i = 0; i < lfsv.size(); ++i)
        for (std::size_t j = 0; j < lfsu.size(); ++j) mat.accumulate(lfsv, i, lfsu, j, -ikbeta * phi[j][0] * phi[i][0] * factor);
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
      const auto factor = ip.weight() * geo.integrationElement(ip.position());
      for (std::size_t i = 0; i < lfsv.size(); ++i) r.accumulate(lfsv, i, -params_.f * phi[i][0] * factor);
    }
  }

  // Impedance data -int g v on a boundary face
  template <typename IG, typename LFSV, typename R>
  void lambda_boundary(const IG& ig, const LFSV& lfsv, R& r) const
  {
    using Basis = typename LFSV::Traits::FiniteElementType::Traits::LocalBasisType;

    const auto geo = ig.geometry();
    const auto geo_in_inside = ig.geometryInInside();
    const auto& basis = lfsv.finiteElement().localBasis();
    std::vector<typename Basis::Traits::RangeType> phi(lfsv.size());
    for (const auto& ip : Dune::PDELab::quadratureRule(geo, 2 * basis.order())) {
      basis.evaluateFunction(geo_in_inside.global(ip.position()), phi);
      const auto factor = ip.weight() * geo.integrationElement(ip.position());
      const auto g = impedance_data(params_, geo.global(ip.position()), ig.unitOuterNormal(ip.position()));
      for (std::size_t i = 0; i < lfsv.size(); ++i) r.accumulate(lfsv, i, -g * phi[i][0] * factor);
    }
  }

private:
  const ConstantParameters& params_;
  bool energy_;
};

// L2 error of the discrete solution x against exact_solution(). Every element is integrated on the rank where it is
// interior; x is consistent, so the rank has the values of all dofs of these elements.
template <class GFS, class RF>
double l2_error(const GFS& gfs, const ConstantParameters& params, const ddm::Vec<RF>& x)
{
  const auto& gv = gfs.gridView();

  using LFS = Dune::PDELab::LocalFunctionSpace<GFS>;
  using Basis = typename LFS::Traits::FiniteElementType::Traits::LocalBasisType;
  LFS lfs(gfs);
  Dune::PDELab::LFSIndexCache<LFS> cache(lfs);
  std::vector<typename Basis::Traits::RangeType> phi;

  double error = 0;
  {
    const auto view = x.host_view(ddm::read);
    for (const auto& e : elements(gv, Dune::Partitions::interior)) {
      lfs.bind(e);
      cache.update();
      const auto geo = e.geometry();
      const auto& basis = lfs.finiteElement().localBasis();
      // Higher order than in the assembly, so that the quadrature error is small compared to the discretisation error
      for (const auto& ip : Dune::PDELab::quadratureRule(geo, 2 * basis.order() + 2)) {
        basis.evaluateFunction(ip.position(), phi);
        RF uh = 0;
        for (std::size_t i = 0; i < lfs.size(); ++i) uh += view[cache.containerIndex(i)[0]] * phi[i][0];
        error += std::norm(uh - exact_solution(params, geo.global(ip.position()))) * ip.weight() * geo.integrationElement(ip.position());
      }
    }
  }
  return std::sqrt(gv.comm().sum(error));
}

void run(const Dune::ParameterTree& config, const Dune::MPIHelper& helper)
{
  // Create the grid: unit square
  constexpr int dim = 2;
  const auto cells = config.get("grid.cells", std::array<unsigned int, dim>{32, 32});
  auto grid = Dune::StructuredGridFactory<Dune::UGGrid<dim>>::createSimplexGrid({0., 0.}, {1., 1.}, cells);
  const auto gv = grid->leafGridView();
  using GV = std::remove_cvref_t<decltype(gv)>;
  auto part = Dune::ParMetisGridPartitioner<GV>::partition(gv, helper);
  grid->loadBalance(part, 0);
  grid->globalRefine(config.get("grid.refine", 0));

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

  ConstantParameters params(config.get("problem.k", 1.));
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

  const auto error = l2_error(gfs, params, x);
  if (gv.comm().rank() == 0) std::cout << "L2 error: " << error << "\n";

  // VTK output of the solution
  add_to_writer(gfs.name(), x);
  // addCellData() with all arguments, the short form is ambiguous for SubsamplingVTKWriter
  std::vector<int> ranks(gv.size(0), gv.comm().rank());
  writer.addCellData(ranks, "Rank", 1, Dune::VTK::Precision::int32);
  writer.write("ex4");
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
