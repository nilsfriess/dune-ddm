/** @file
 *
 *  Checks galerkin_product() against a straightforward host reference.
 *
 *  A global tridiagonal matrix whose index set is split into one contiguous block of
 *  n_owned indices per rank; every rank additionally holds a copy of each neighbouring
 *  rank's boundary index (overlap of one). The template vectors of a rank are supported
 *  on the rank's owned indices only, so the local matrices need complete rows only there
 *  and the zero-extension via exchange_all_holders() provides exactly what is missing.
 *  The ranks contribute different numbers of template vectors, so the per-rank blocks of
 *  the coarse matrix are ragged.
 *
 *  Rank 0 assembles the coarse matrix; its rows must equal the rows of the densely
 *  computed R A R^T, with the coarse columns ordered by rank. All other ranks must
 *  receive an empty matrix.
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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <dune/common/fmatrix.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

using Bcrs = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;

constexpr std::size_t n_owned = 12; ///< owned indices per rank; the global index set is their concatenation

/// Entry @p gid of template vector @p k; distinct values so that no coarse entry cancels
double template_entry(std::size_t gid, int k) { return static_cast<double>((gid + 1) * (k + 1)); }

/// Number of template vectors a rank contributes (all >= 1, ragged across ranks)
int num_t_of_rank(int rank) { return 2 + rank % 2; }

/// The gids this rank holds, in local order: its owned block first, then one copy of each
/// neighbouring rank's boundary gid (left neighbour first, if any).
std::vector<std::size_t> local_gids(int rank, int size)
{
  std::vector<std::size_t> gids;
  for (std::size_t i = 0; i < n_owned; ++i) gids.push_back(static_cast<std::size_t>(rank) * n_owned + i);
  if (rank > 0) gids.push_back(static_cast<std::size_t>(rank) * n_owned - 1);
  if (rank < size - 1) gids.push_back(static_cast<std::size_t>(rank + 1) * n_owned);
  return gids;
}

bool is_owned(std::size_t gid, int rank) { return gid >= static_cast<std::size_t>(rank) * n_owned && gid < static_cast<std::size_t>(rank + 1) * n_owned; }

ddm::Communication make_communication(const std::vector<std::size_t>& gids, int rank)
{
  std::vector<ddm::CommunicationNodes> roots(gids.size());
  for (std::size_t j = 0; j < gids.size(); ++j) {
    // Owned indices are owned by this rank, copies by the neighbouring rank they came from
    const int owner = is_owned(gids[j], rank) ? rank : (gids[j] < static_cast<std::size_t>(rank) * n_owned ? rank - 1 : rank + 1);
    roots[j] = {owner, static_cast<std::int64_t>(gids[j])};
  }
  return ddm::Communication(MPI_COMM_WORLD, roots);
}

/// The local piece of the global tridiagonal (2 on the diagonal, -1 next to it): complete
/// rows wherever the columns exist on this rank, which is all the assembly needs because
/// the template vectors vanish on the copied indices.
Bcrs build_local_matrix(const std::vector<std::size_t>& gids, std::size_t global_n)
{
  Bcrs A;
  A.setBuildMode(Bcrs::implicit);
  A.setImplicitBuildModeParameters(3, 0.5);
  A.setSize(gids.size(), gids.size());

  const auto local_of = [&](std::size_t g) { return static_cast<std::size_t>(std::find(gids.begin(), gids.end(), g) - gids.begin()); };

  for (std::size_t j = 0; j < gids.size(); ++j) {
    const std::size_t g = gids[j];
    A.entry(j, j) = 2.0;
    if (g > 0 && local_of(g - 1) < gids.size()) A.entry(j, local_of(g - 1)) = -1.0;
    if (g + 1 < global_n && local_of(g + 1) < gids.size()) A.entry(j, local_of(g + 1)) = -1.0;
  }
  A.compress();
  return A;
}

/// This rank's template vectors as dense host columns (length n_local), supported on the
/// owned indices and zero on the copies.
std::vector<std::vector<double>> local_restriction(const std::vector<std::size_t>& gids, int rank)
{
  const int num_t = num_t_of_rank(rank);
  std::vector<std::vector<double>> Rcols(num_t, std::vector<double>(gids.size(), 0.0));
  for (int k = 0; k < num_t; ++k)
    for (std::size_t j = 0; j < gids.size(); ++j)
      if (is_owned(gids[j], rank)) Rcols[k][j] = template_entry(gids[j], k);
  return Rcols;
}

/// G_ref[k][g] = R_k . (A R_g) for the global problem, computed densely on the host.
/// Coarse columns are ordered by rank (rank r's vectors first), like galerkin_product().
std::vector<std::vector<double>> reference_product(int size)
{
  const std::size_t global_n = static_cast<std::size_t>(size) * n_owned;

  std::vector<std::vector<double>> cols;
  for (int r = 0; r < size; ++r)
    for (int k = 0; k < num_t_of_rank(r); ++k) {
      std::vector<double> c(global_n, 0.0);
      for (std::size_t i = static_cast<std::size_t>(r) * n_owned; i < static_cast<std::size_t>(r + 1) * n_owned; ++i) c[i] = template_entry(i, k);
      cols.push_back(std::move(c));
    }

  std::vector<std::vector<double>> AR;
  for (const auto& c : cols) {
    std::vector<double> y(global_n, 0.0);
    for (std::size_t i = 0; i < global_n; ++i) {
      y[i] = 2.0 * c[i];
      if (i > 0) y[i] -= c[i - 1];
      if (i + 1 < global_n) y[i] += -1.0 * c[i + 1];
    }
    AR.push_back(std::move(y));
  }

  const std::size_t m = cols.size();
  std::vector<std::vector<double>> G(m, std::vector<double>(m, 0.0));
  for (std::size_t k = 0; k < m; ++k)
    for (std::size_t g = 0; g < m; ++g) {
      double sum = 0.0;
      for (std::size_t i = 0; i < global_n; ++i) sum += cols[k][i] * AR[g][i];
      G[k][g] = sum;
    }
  return G;
}

/// Rank 0's assembled coarse matrix must equal the reference: every rank contributes its own
/// rows via gatherMatrixFromRowsFlat(), and since coarse rows are ordered by rank each rank's
/// block lands exactly at its coarse offset. Every other rank must receive an empty matrix.
void check_coarse_matrix(Dune::TestSuite& t, const Bcrs& G, const std::vector<std::vector<double>>& G_ref, int rank)
{
  if (rank != 0) {
    t.check(G.N() == 0 && G.M() == 0, "non-root ranks get an empty matrix") << "got " << G.N() << "x" << G.M();
    return;
  }

  t.check(G.N() == G_ref.size(), "coarse matrix has the right number of rows") << "N=" << G.N() << ", expected " << G_ref.size();
  t.check(G.M() == G_ref.size(), "coarse matrix has the right number of columns") << "M=" << G.M() << ", expected " << G_ref.size();

  for (std::size_t k = 0; k < G.N(); ++k) {
    std::vector<double> row(G.M(), 0.0);
    std::vector<char> present(G.M(), false);
    for (auto ci = G[k].begin(); ci != G[k].end(); ++ci) {
      row[ci.index()] = (*ci)[0][0];
      present[ci.index()] = true;
    }
    for (std::size_t g = 0; g < G.M(); ++g) {
      const double ref = G_ref[k][g];
      if (!present[g]) t.check(std::abs(ref) <= 1e-13, "entry missing from the sparsity pattern is zero in the reference") << "k=" << k << " g=" << g << " ref=" << ref;
      else t.check(std::abs(row[g] - ref) <= 1e-11 * (1.0 + std::abs(ref)), "coarse entry matches the reference") << "k=" << k << " g=" << g << " got=" << row[g] << " ref=" << ref;
    }
  }
}

void run_host_test(Dune::TestSuite& t)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto gids = local_gids(rank, size);
  const Bcrs A = build_local_matrix(gids, static_cast<std::size_t>(size) * n_owned);
  const auto Rcols = local_restriction(gids, rank);

  ddm::MultiVector<double, ddm::backend::HostBackend> R({}, gids.size(), Rcols.size());
  for (std::size_t k = 0; k < Rcols.size(); ++k) std::copy(Rcols[k].begin(), Rcols[k].end(), R.col(k));

  auto comm = make_communication(gids, rank);
  const Bcrs G = ddm::galerkin_product(A, R, comm);

  check_coarse_matrix(t, G, reference_product(size), rank);
}

void run_sycl_test(Dune::TestSuite& t, sycl::queue& q)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto gids = local_gids(rank, size);
  const Bcrs A_host = build_local_matrix(gids, static_cast<std::size_t>(size) * n_owned);
  const ddm::Sycl::Mat<double> A = ddm::Sycl::Mat<double>::from_bcrs(q, A_host);
  const auto Rcols = local_restriction(gids, rank);

  ddm::MultiVector<double, ddm::backend::SyclBackend> R(q, gids.size(), Rcols.size());
  for (std::size_t k = 0; k < Rcols.size(); ++k) q.memcpy(R.col(k), Rcols[k].data(), Rcols[k].size() * sizeof(double));
  q.wait();

  auto comm = make_communication(gids, rank);
  const Bcrs G = ddm::galerkin_product(A, R, comm);

  check_coarse_matrix(t, G, reference_product(size), rank);
}

} // namespace

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  setup_loggers(helper.rank(), argc, argv);

  int failed = 0;
  failed |= ddmtest::runParallelTest("test_galerkin_product host", [&](Dune::TestSuite& t) { run_host_test(t); });

  sycl::queue q{sycl::property::queue::in_order{}};
  failed |= ddmtest::runParallelTest("test_galerkin_product sycl", [&](Dune::TestSuite& t) { run_sycl_test(t, q); });

  return failed;
}
