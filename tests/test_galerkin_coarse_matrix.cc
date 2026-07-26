#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_utils.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/ddm/galerkin_preconditioner.hh>
#include <dune/ddm/helpers.hh>
#include <dune/ddm/overlap_extension.hh>
#include <dune/ddm/pou.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <format>
#include <limits>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

/** @file
 *
 *  Parallel tests for GalerkinPreconditioner: the coarse matrix it assembles must equal the global
 *  Galerkin product \f$R A R^T\f$, and apply() must evaluate \f$R^T (R A R^T)^{-1} R\f$.
 *
 *  Both are checked against a dense reference computed on rank 0 from the *global* matrix and the
 *  globally assembled (zero-extended) template vectors, so the reference shares no code with the
 *  distributed assembly it verifies.
 *
 *  <b>Why the distributed assembly can be exact.</b> Rank q evaluates
 *  \f$\Phi_k^q \cdot A^{ovlp}_q V\f$, where \f$V\f$ is a neighbour's template vector transferred
 *  onto q's index set: it is filled at the indices p and q share and left at zero elsewhere. That
 *  agrees with the global product exactly when, for every \f$i\f$ in the support of
 *  \f$\Phi_k^q\f$ and every graph neighbour \f$j\f$ of \f$i\f$, both \f$j\f$ lies in q's index set
 *  and \f$A^{ovlp}_q[i][j] = A[i][j]\f$. Every template vector below is built from a partition of
 *  unity, so its support stops before the outer boundary layer of the overlapping subdomain and
 *  both conditions hold. This is also what makes the entries between two ranks that share no index
 *  genuinely zero, which is what the assembly assumes.
 *
 *  Exactness is the point, not a convenience: it lets the test compare against the mathematical
 *  definition of the coarse space rather than against a second implementation of the algorithm.
 */

using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;
using Comm = Dune::OwnerOverlapCopyCommunication<std::size_t>;
using Prec = GalerkinPreconditioner<Vector, Comm>;

namespace {

/// Relative tolerance for all comparisons. The reference sums in a different order than the
/// distributed assembly, so agreement is up to round-off, not bit for bit.
constexpr double tolerance = 1e-9;

/// The defect apply() is tested on. Depends on the global index only, so the vector it defines is
/// consistent across ranks, which is how apply() interprets its argument.
double defectValue(std::size_t global_index) { return 1.0 + std::sin(0.017 * static_cast<double>(global_index)); }

// ---------------------------------------------------------------------------------------------
// The distributed fine problem
// ---------------------------------------------------------------------------------------------

struct Problem {
  std::size_t nx = 0;             ///< Grid points per direction; the DOF at (ix, iy) has global index iy * nx + ix.
  std::size_t N = 0;              ///< Global number of DOFs.
  Matrix A_global;                ///< The full matrix, the reference for every check. Assembled on every rank; it only depends on nx.
  std::shared_ptr<Matrix> localA; ///< Non-overlapping part of the matrix.
  std::shared_ptr<Comm> novlp_comm;
  std::shared_ptr<Matrix> A_ovlp; ///< Overlapping part of the matrix, what the preconditioner works on.
  std::shared_ptr<Comm> ovlp_comm;
  std::vector<bool> dirichlet; ///< Global mask of the DOFs whose row was replaced by an identity row.
  int overlap = 0;
  bool active = false; ///< True if *every* rank received a part of the matrix (collectively determined).
};

/** @brief Assembles the fine problem on rank 0 and distributes it.
 *
 *  The matrix is a 2d Laplacian with Neumann boundary conditions plus a shift on the diagonal. The
 *  shift matters: the pure Neumann Laplacian is singular, the constant vector lies in the range of
 *  every \f$R^T\f$ built below, and so \f$R A R^T\f$ would be singular too — which both the
 *  preconditioner's own coarse solve and the reference solve need it not to be.
 *
 *  With @p with_dirichlet, the left edge of the grid gets identity rows, the form in which
 *  detect_dirichlet_dofs() recognises Dirichlet DOFs. Only the row is cleared, not the column, so
 *  the matrix is then nonsymmetric.
 *
 *  Collective. Every rank returns the same @c active flag, so callers may branch on it.
 */
Problem makeProblem(std::size_t nx, int overlap, bool with_dirichlet, int rank, int size)
{
  Problem p;
  p.nx = nx;
  p.N = nx * nx;
  p.overlap = overlap;

  // The rule depends on the global index only, so every rank can evaluate it without communication.
  p.dirichlet.assign(p.N, false);
  if (with_dirichlet)
    for (std::size_t g = 0; g < p.N; ++g) p.dirichlet[g] = (g % nx == 0);

  // Assembled on every rank: it only depends on nx, and every rank needs it to check its own part of
  // the overlapping matrix. distributeMatrixFrom0 still gets it from rank 0 only, as it requires.
  p.A_global = ddmtest::buildLaplacian2d<Matrix>(nx, nx);

  for (auto ri = p.A_global.begin(); ri != p.A_global.end(); ++ri) (*ri)[ri.index()] += 1.0;

  for (auto ri = p.A_global.begin(); ri != p.A_global.end(); ++ri) {
    if (!p.dirichlet[ri.index()]) continue;
    for (auto ci = ri->begin(); ci != ri->end(); ++ci) *ci = (ci.index() == ri.index()) ? 1.0 : 0.0;
  }

  const Matrix nothing;
  auto dist = distributeMatrixFrom0<Matrix, std::size_t>(rank == 0 ? p.A_global : nothing, MPI_COMM_WORLD, size);
  p.novlp_comm = dist.comm;
  p.localA = dist.matrix;

  // Reduce the flag so that the early return below is taken on all ranks or on none, which keeps
  // the collectives that follow aligned.
  p.active = ddmtest::allRanks(dist.active);
  if (!p.active) return p;

  // Dune's redistributeMatrix() leaves the non-owner rows of the local matrix as identity rows
  // (CommMatrixRow::setOverlapRowsToDirichlet), its convention for the overlapping solver category.
  // Those placeholders must not reach the overlap extension: it would either add them to (Additive)
  // or assign them over (Consistent) the real values their owner holds, and either way the entries
  // on the subdomain interfaces come out wrong. Clearing them makes the local matrix a proper
  // additive splitting -- the owner holds the complete row, every other rank nothing -- which is
  // what MatrixRepresentation::Additive expects. checkOverlappingMatrix() verifies the outcome.
  std::vector<bool> owned(p.localA->N(), false);
  for (const auto& idx : p.novlp_comm->indexSet())
    if (idx.local().attribute() == Dune::OwnerOverlapCopyAttributeSet::owner) owned[idx.local().local()] = true;

  for (auto ri = p.localA->begin(); ri != p.localA->end(); ++ri)
    if (!owned[ri.index()])
      for (auto ci = ri->begin(); ci != ri->end(); ++ci) *ci = 0.0;

  auto ovlp = create_overlapping_matrix(*p.novlp_comm, *p.localA, overlap, MatrixRepresentation::Additive);
  p.ovlp_comm = ovlp.comm;
  p.A_ovlp = ovlp.matrix;

  return p;
}

// ---------------------------------------------------------------------------------------------
// The coarse space
// ---------------------------------------------------------------------------------------------

/// How the template vectors are laid out. All variants are built from a partition of unity, see the
/// exactness argument in the file comment.
enum class BasisKind : std::uint8_t {
  Trivial,  ///< One vector per rank: the indicator of the owned, non-overlapping subdomain.
  Standard, ///< One vector per rank: the standard partition of unity, which reaches into the overlap.
  Ragged,   ///< 1 + rank % 3 vectors per rank, so the ranks disagree on how many they contribute.
};

const char* basisName(BasisKind kind)
{
  switch (kind) {
    case BasisKind::Trivial: return "trivial POU";
    case BasisKind::Standard: return "standard POU";
    case BasisKind::Ragged: return "ragged counts";
  }
  return "";
}

/** @brief The rank-local template vectors of the coarse space.
 *
 *  The first vector is the partition of unity itself; the further ones scale it by the grid
 *  coordinates. That keeps their support inside the subdomain while making them linearly
 *  independent, so the coarse matrix stays invertible.
 */
std::vector<Vector> buildBasis(BasisKind kind, const Problem& p, int rank)
{
  const auto pou_type = kind == BasisKind::Standard ? PartitionOfUnityType::Standard : PartitionOfUnityType::Trivial;
  const PartitionOfUnity pou(*p.A_ovlp, *p.ovlp_comm, pou_type, 0, p.overlap);

  const std::size_t num_t = kind == BasisKind::Ragged ? 1 + static_cast<std::size_t>(rank % 3) : 1;
  std::vector<Vector> ts(num_t, Vector(pou.size()));
  for (auto& v : ts) v = 0.0;

  const auto scale = static_cast<double>(p.nx);
  for (const auto& idx : p.ovlp_comm->indexSet()) {
    const auto l = idx.local().local();
    const std::size_t ix = idx.global() % p.nx;
    const std::size_t iy = idx.global() / p.nx;

    // k = 0 is the partition of unity itself, k = 1 and k = 2 scale it by the grid coordinates.
    for (std::size_t k = 0; k < num_t; ++k) {
      const double factor = k == 0 ? 1.0 : static_cast<double>(k == 1 ? ix : iy) / scale;
      ts[k][l] = pou[l][0] * factor;
    }
  }

  return ts;
}

/** @brief Zero-extends the local vectors to the global index space and gathers them on rank 0.
 *
 *  The rows of all ranks are concatenated in rank order, which is exactly the coarse numbering
 *  GalerkinPreconditioner assigns to the template vectors. The result is therefore the restriction
 *  matrix \f$R\f$ of the coarse space. Empty on every rank but 0.
 *
 *  Vectors shorter than the index set are accepted and read at their leading entries, so this also
 *  works for vectors living on the non-overlapping index set: the overlap extension appends its new
 *  indices, it does not renumber the existing ones.
 *
 *  Collective.
 */
Dune::BCRSMatrix<double> gatherGlobalRows(const std::vector<Vector>& vecs, const Comm& comm, std::size_t N)
{
  std::vector<Dune::BlockVector<double>> rows(vecs.size(), Dune::BlockVector<double>(N));
  for (auto& row : rows) row = 0.0;

  for (const auto& idx : comm.indexSet()) {
    const auto l = idx.local().local();
    for (std::size_t k = 0; k < vecs.size(); ++k)
      if (l < vecs[k].N()) rows[k][idx.global()] = vecs[k][l];
  }

  return gatherMatrixFromRows(rows, MPI_COMM_WORLD, 0);
}

/** @brief The global template vectors as dense vectors, on rank 0.
 *
 *  @p dirichlet entries are dropped, because GalerkinPreconditioner zeroes the template vectors at
 *  the Dirichlet DOFs it detects in the matrix before using them.
 */
std::vector<Vector> globalBasis(const Dune::BCRSMatrix<double>& R, std::size_t N, const std::vector<bool>& dirichlet)
{
  std::vector<Vector> phi(R.N(), Vector(N));
  for (std::size_t i = 0; i < R.N(); ++i) {
    phi[i] = 0.0;
    for (auto ci = R[i].begin(); ci != R[i].end(); ++ci)
      if (!dirichlet[ci.index()]) phi[i][ci.index()] = *ci;
  }
  return phi;
}

/// The reference coarse matrix R A R^T, computed densely from the global matrix on rank 0.
Dune::DynamicMatrix<double> referenceCoarseMatrix(const std::vector<Vector>& phi, const Matrix& A)
{
  const auto m = phi.size();
  Dune::DynamicMatrix<double> A0(m, m, 0.0);

  Vector y(A.N());
  for (std::size_t j = 0; j < m; ++j) {
    A.mv(phi[j], y);
    for (std::size_t i = 0; i < m; ++i) A0[i][j] = phi[i] * y;
  }
  return A0;
}

// ---------------------------------------------------------------------------------------------
// Comparisons, all on rank 0
// ---------------------------------------------------------------------------------------------

/// Entry (@p i, @p j) of a sparse matrix, 0 where the pattern has no entry.
double at(const Prec::CoarseMatrix& A, std::size_t i, std::size_t j)
{
  const auto ci = A[i].find(j);
  return ci == A[i].end() ? 0.0 : (*ci)[0][0];
}

/** @brief Describes the first entry of @p A0 that differs from @p ref, empty if all match.
 *
 *  Iterating the dense reference rather than the sparse pattern means a spurious nonzero in @p A0
 *  is reported too, not just a wrong or missing value.
 */
std::string firstBadEntry(const Prec::CoarseMatrix& A0, const Dune::DynamicMatrix<double>& ref, double tol)
{
  for (std::size_t i = 0; i < ref.N(); ++i)
    for (std::size_t j = 0; j < ref.M(); ++j) {
      const double got = at(A0, i, j);
      if (std::abs(got - ref[i][j]) > tol) return std::format("entry ({},{}): expected {}, got {}", i, j, ref[i][j], got);
    }
  return {};
}

/// The largest asymmetry of @p A0, i.e. max |A0[i][j] - A0[j][i]|.
double asymmetry(const Prec::CoarseMatrix& A0)
{
  double worst = 0.0;
  for (std::size_t i = 0; i < A0.N(); ++i)
    for (std::size_t j = 0; j < A0.M(); ++j) worst = std::max(worst, std::abs(at(A0, i, j) - at(A0, j, i)));
  return worst;
}

// ---------------------------------------------------------------------------------------------
// The checks
// ---------------------------------------------------------------------------------------------

/** @brief The overlapping matrix must reproduce the global matrix wherever it stores an entry.
 *
 *  This is the precondition the exactness of the distributed Galerkin assembly rests on, see the
 *  file comment, and it is easy to violate — the local matrix handed out by distributeMatrixFrom0()
 *  carries identity placeholder rows at its non-owner indices, and those inflate the interface
 *  entries if they reach the overlap extension. Checking it separately means such a regression is
 *  reported where it originates instead of surfacing as an unexplained coarse matrix mismatch.
 *
 *  The pattern is allowed to be incomplete only where it has to be: a row may omit a neighbour that
 *  lies outside the local index set, but not one that lies inside it.
 *
 *  Collective.
 */
void checkOverlappingMatrix(Dune::TestSuite& t, const Problem& p)
{
  constexpr auto not_local = std::numeric_limits<std::size_t>::max();

  std::vector<std::size_t> l2g(p.A_ovlp->N(), 0);
  std::vector<std::size_t> g2l(p.N, not_local);
  for (const auto& idx : p.ovlp_comm->indexSet()) {
    l2g[idx.local().local()] = idx.global();
    g2l[idx.global()] = idx.local().local();
  }

  double worst = 0.0;
  double missing = 0.0;
  for (auto ri = p.A_ovlp->begin(); ri != p.A_ovlp->end(); ++ri) {
    const auto gi = l2g[ri.index()];

    for (auto ci = ri->begin(); ci != ri->end(); ++ci) {
      const auto ref = p.A_global[gi].find(l2g[ci.index()]);
      const double want = ref == p.A_global[gi].end() ? 0.0 : (*ref)[0][0];
      worst = std::max(worst, std::abs((*ci)[0][0] - want));
    }

    for (auto ci = p.A_global[gi].begin(); ci != p.A_global[gi].end(); ++ci) {
      const auto lj = g2l[ci.index()];
      if (lj == not_local) continue; // legitimately outside the subdomain
      if (ri->find(lj) == ri->end()) missing += 1.0;
    }
  }

  std::array<double, 2> local = {{worst, missing}};
  std::array<double, 2> global = {{0.0, 0.0}};
  MPI_Allreduce(local.data(), global.data(), 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  t.check(global[0] == 0.0, "overlapping matrix equals the global matrix") << "largest deviation " << global[0];
  t.check(global[1] == 0.0, "overlapping matrix keeps every neighbour it knows") << static_cast<std::size_t>(global[1]) << " entries missing from the pattern on one rank";
}

/** @brief Builds the coarse space for @p kind and checks the coarse matrix and the correction.
 *
 *  @p expect_symmetric says whether the fine matrix is symmetric, and hence whether the coarse
 *  matrix has to come out symmetric as well. That is a check the reference comparison does not
 *  subsume: it fails when the products between two different ranks are assembled inconsistently,
 *  independently of whether the reference happens to be reproduced.
 *
 *  Collective; every check is recorded rather than thrown so that all ranks reach all collectives.
 */
void checkCoarseSpace(Dune::TestSuite& t, const Problem& p, BasisKind kind, bool expect_symmetric)
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const std::string prefix = std::string(basisName(kind)) + (expect_symmetric ? "" : " with Dirichlet rows");

  const auto ts = buildBasis(kind, p, rank);
  const auto R = gatherGlobalRows(ts, *p.ovlp_comm, p.N);

  Dune::ParameterTree ptree;
  ptree["galerkin.type"] = "umfpack";

  Prec prec(*p.A_ovlp, ts, p.ovlp_comm, ptree);
  const auto& A0 = prec.get_coarse_matrix();

  // The coarse size, so that every rank can judge the shape it is supposed to see.
  int num_t = static_cast<int>(ts.size());
  int total = 0;
  MPI_Allreduce(&num_t, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const auto m = static_cast<std::size_t>(total);

  std::vector<Vector> phi;
  Dune::DynamicMatrix<double> ref;
  bool shape_ok = false;

  if (rank == 0) {
    phi = globalBasis(R, p.N, p.dirichlet);
    ref = referenceCoarseMatrix(phi, p.A_global);
    const double scale = std::max(1.0, ref.infinity_norm());

    shape_ok = A0.N() == m and A0.M() == m;
    t.check(shape_ok, prefix + ": coarse matrix shape") << "expected " << m << "x" << m << ", got " << A0.N() << "x" << A0.M();

    if (shape_ok) {
      const auto bad = firstBadEntry(A0, ref, tolerance * scale);
      t.check(bad.empty(), prefix + ": coarse matrix equals R A R^T") << bad;

      if (expect_symmetric) {
        const double asym = asymmetry(A0);
        t.check(asym <= tolerance * scale, prefix + ": coarse matrix is symmetric") << "max |A0[i][j] - A0[j][i]| = " << asym;
      }
    }
  }
  else {
    t.check(A0.N() == 0 and A0.M() == 0, prefix + ": coarse matrix is empty off rank 0") << "got " << A0.N() << "x" << A0.M();
  }

  // ---- The coarse correction -----------------------------------------------------------------
  //
  // apply() takes and returns vectors on the non-overlapping index set. The defect is consistent by
  // construction (it depends on the global index only), which is how apply() reads it, and the
  // result comes back consistent as well, so it can be compared at every local index.

  Vector d(p.localA->N());
  Vector x(p.localA->N());
  d = 0.0;
  x = 0.0;
  for (const auto& idx : p.ovlp_comm->indexSet())
    if (idx.local().local() < d.N()) d[idx.local().local()] = defectValue(idx.global());

  prec.apply(x, d);

  // apply() reuses two member vectors as scratch space, so a second call has to give the same
  // answer as the first.
  Vector x_again(p.localA->N());
  x_again = 0.0;
  prec.apply(x_again, d);

  std::vector<double> u_ref(p.N, 0.0);
  if (rank == 0 and shape_ok) {
    Dune::DynamicVector<double> rhs(m, 0.0);
    Dune::DynamicVector<double> c(m, 0.0);

    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t g = 0; g < p.N; ++g) rhs[i] += phi[i][g][0] * defectValue(g);

    ref.solve(c, rhs);

    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t g = 0; g < p.N; ++g) u_ref[g] += c[i] * phi[i][g][0];
  }
  MPI_Bcast(u_ref.data(), static_cast<int>(p.N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double max_err = 0.0;
  double repeat_err = 0.0;
  for (const auto& idx : p.ovlp_comm->indexSet()) {
    const auto l = idx.local().local();
    if (l >= x.N()) continue;
    max_err = std::max(max_err, std::abs(x[l][0] - u_ref[idx.global()]));
    repeat_err = std::max(repeat_err, std::abs(x[l][0] - x_again[l][0]));
  }

  double u_scale = 0.0;
  for (const double v : u_ref) u_scale = std::max(u_scale, std::abs(v));

  std::array<double, 2> local_errs = {{max_err, repeat_err}};
  std::array<double, 2> global_errs = {{0.0, 0.0}};
  MPI_Allreduce(local_errs.data(), global_errs.data(), 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  t.check(global_errs[0] <= tolerance * std::max(1.0, u_scale), prefix + ": apply() computes R^T (R A R^T)^-1 R d")
      << "max deviation " << global_errs[0] << ", reference magnitude " << u_scale;

  // Not bit for bit: the order in which the prolongation sums the contributions of the neighbouring
  // subdomains is up to MPI, which is worth a last digit. Stale scratch state would be visible at a
  // wholly different magnitude.
  t.check(global_errs[1] <= 1e-14 * std::max(1.0, u_scale), prefix + ": apply() is repeatable") << "two calls differ by " << global_errs[1];
}

} // namespace

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  const int rank = helper.rank();
  const int size = helper.size();

  return ddmtest::runParallelTest("test_galerkin_coarse_matrix", [&](Dune::TestSuite& t) {
    // Grid large enough that every rank owns a two-dimensional chunk, so that the coordinate-scaled
    // template vectors of the ragged case stay linearly independent.
    const auto nx = static_cast<std::size_t>(12 * std::ceil(std::sqrt(static_cast<double>(size))));
    const int overlap = 2;

    const auto neumann = makeProblem(nx, overlap, false, rank, size);
    t.check(neumann.active, "every rank received part of the matrix") << "rank " << rank << " did not; choose a larger problem size";

    if (neumann.active) {
      checkOverlappingMatrix(t, neumann);

      // One template vector per rank, supported on the owned subdomain only: the coarse coupling
      // between two ranks then comes purely from the matrix entries across their interface.
      checkCoarseSpace(t, neumann, BasisKind::Trivial, true);

      // One template vector per rank whose support reaches into the overlap, so that the products
      // between two ranks involve DOFs both of them know.
      checkCoarseSpace(t, neumann, BasisKind::Standard, true);

      // Different numbers of template vectors per rank, which is what the padding to max_num_t, the
      // per-rank offsets and the ragged gather of the coarse rows exist for.
      checkCoarseSpace(t, neumann, BasisKind::Ragged, true);
    }

    // Identity rows in the matrix: the template vectors have to be zeroed there before the Galerkin
    // product is formed, otherwise the coarse correction injects values at the Dirichlet DOFs.
    const auto dirichlet = makeProblem(nx, overlap, true, rank, size);
    t.check(dirichlet.active, "every rank received part of the Dirichlet matrix") << "rank " << rank << " did not";

    if (dirichlet.active) {
      checkOverlappingMatrix(t, dirichlet);
      checkCoarseSpace(t, dirichlet, BasisKind::Ragged, false);
    }
  });
}
