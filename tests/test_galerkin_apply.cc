/** @file
 *
 *  Checks GalerkinPreconditioner::apply() on a single rank against a hand-computed reference:
 *  with a tridiagonal matrix and one template vector R of ones, the coarse system is 1x1 and
 *  apply() must return x = (R.d / (R.A.R)) * R.
 *
 *  Runs for the host (ISTL) backend and for the SYCL backend.
 */

#include "dune/ddm/backend/host/backend.hh"
#include "dune/ddm/backend/sycl/backend.hh"
#include "dune/ddm/communication.hh"
#include "dune/ddm/galerkin_preconditioner.hh"
#include "dune/ddm/logger.hh"
#include "dune/ddm/multivector.hh"
#include "dune/ddm/sycl/mat.hh"
#include "tests/test_utils.hh"

#include <cmath>
#include <cstddef>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

using Bcrs = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
using HostVector = Dune::BlockVector<Dune::FieldVector<double, 1>>;

constexpr std::size_t n = 12;

Bcrs build_matrix()
{
  Bcrs A;
  A.setBuildMode(Bcrs::implicit);
  A.setImplicitBuildModeParameters(3, 0.5);
  A.setSize(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    A.entry(i, i) = 2.0;
    if (i > 0) A.entry(i, i - 1) = -1.0;
    if (i + 1 < n) A.entry(i, i + 1) = -1.0;
  }
  A.compress();
  return A;
}

ddm::Communication make_single_rank_communication()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<ddm::CommunicationNodes> roots(n);
  for (std::size_t i = 0; i < n; ++i) roots[i] = {rank, static_cast<std::int64_t>(i)};
  return ddm::Communication(MPI_COMM_WORLD, roots);
}

/// The reference: x = (R.d / R.A.R) * R with R = ones
/// This doesn't use the matrix; it applies the operator matrix-free.
/// If the test matrix ever changes, this must changes as well.
template <class Vector>
Vector reference_apply(const Vector& d)
{
  double Rd = 0.0;
  for (std::size_t i = 0; i < n; ++i) Rd += d[i];

  double AR = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double row = 2.0;
    if (i > 0) row -= 1.0;
    if (i + 1 < n) row -= 1.0;
    AR += row;
  }

  HostVector x(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = Rd / AR;
  return x;
}

template <class X, class Ref>
void check_solution(Dune::TestSuite& t, const X& x, const Ref& ref, const char* label)
{
  for (std::size_t i = 0; i < n; ++i)
    t.check(std::abs(x[i] - ref[i]) <= 1e-11 * (1.0 + std::abs(ref[i])), "apply result matches the reference") << label << ": i=" << i << " got=" << x[i] << " ref=" << ref[i];
}

void run_host_test(Dune::TestSuite& t)
{
  const Bcrs A = build_matrix();

  ddm::MultiVector<double, ddm::backend::HostBackend> R({}, n, 1);
  for (std::size_t i = 0; i < n; ++i) R.col(0u)[i] = 1.0;

  HostVector d(n);
  for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<double>(i % 3) - 1.0;

  auto comm = std::make_shared<ddm::Communication>(make_single_rank_communication());
  ddm::GalerkinPreconditioner<HostVector, ddm::Communication> prec(A, std::move(R), comm);

  HostVector x(n);
  prec.apply(x, d);
  check_solution(t, x, reference_apply(d), "host");
}

void run_sycl_test(Dune::TestSuite& t, sycl::queue& q)
{
  const Bcrs A_host = build_matrix();
  const ddm::Sycl::Mat<double> A = ddm::Sycl::Mat<double>::from_bcrs(q, A_host);

  ddm::MultiVector<double, ddm::backend::SyclBackend> R(q, n, 1);
  {
    std::vector<double> ones(n, 1.0);
    q.memcpy(R.col(0u), ones.data(), n * sizeof(double));
    q.wait();
  }

  std::vector<double> d_host(n);
  for (std::size_t i = 0; i < n; ++i) d_host[i] = static_cast<double>(i % 3) - 1.0;
  ddm::Sycl::Vec<double> d = ddm::Sycl::Vec<double>::from_host_vector(q, d_host);

  auto comm = std::make_shared<ddm::Communication>(make_single_rank_communication());
  ddm::GalerkinPreconditioner<ddm::Sycl::Vec<double>, ddm::Communication> prec(A, std::move(R), comm);

  ddm::Sycl::Vec<double> x(q, n);
  prec.apply(x, d);

  const auto x_host = x.to_host_vector();
  HostVector d_bv(n);
  for (std::size_t i = 0; i < n; ++i) d_bv[i] = d_host[i];
  check_solution(t, x_host, reference_apply(d_bv), "sycl");
}

} // namespace

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  setup_loggers(helper.rank(), argc, argv);

  int failed = 0;
  failed |= ddmtest::runParallelTest("test_galerkin_apply host", [&](Dune::TestSuite& t) { run_host_test(t); });

  sycl::queue q{sycl::property::queue::in_order{}};
  failed |= ddmtest::runParallelTest("test_galerkin_apply sycl", [&](Dune::TestSuite& t) { run_sycl_test(t, q); });

  return failed;
}
 
