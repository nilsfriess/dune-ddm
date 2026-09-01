#pragma once

#include <algorithm>
#include <cstddef>
#include <dune/common/exceptions.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/grid/common/datahandleif.hh>
#include <dune/grid/common/gridenums.hh>
#include <dune/grid/utility/globalindexset.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <iostream>
#include <mpi.h>
#include <vector>

/** @file
 *
 *  Shared utilities for the dune-ddm tests, on top of Dune::TestSuite.
 *
 *  Use TestSuite::check() rather than require() inside collective regions. check() records the
 *  failure and carries on, so every rank still reaches every collective call; require() throws,
 *  which unwinds one rank out of a collective and hangs the rest.
 */

namespace ddmtest {

/// Exit code CTest interprets as "test skipped".
inline constexpr int skip_exit_code = 77;

/** @brief Collective AND over all ranks.
 *
 *  Use for conditions that are only meaningful globally, so that every rank passes the same
 *  verdict to its TestSuite instead of some ranks failing and others not.
 */
inline bool allRanks(bool condition, MPI_Comm comm = MPI_COMM_WORLD)
{
  int local = condition ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
  return global != 0;
}

/** @brief Collective. Reduces the per-rank results and returns the same exit code on every rank.
 *
 *  Dune::TestSuite counts failures per rank, so without this a failure on rank 7 alone leaves
 *  rank 0 reporting success. Rank 0 additionally names the failing ranks.
 */
inline int mpiExit(const Dune::TestSuite& suite, const std::string& display_name, MPI_Comm comm = MPI_COMM_WORLD)
{
  const int ok = suite.report() ? 1 : 0; // report() prints this rank's summary if it failed

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::vector<int> all(rank == 0 ? static_cast<std::size_t>(size) : 0);
  MPI_Gather(&ok, 1, MPI_INT, all.data(), 1, MPI_INT, 0, comm);

  int num_failed = 0;
  if (rank == 0) {
    std::vector<int> failed;
    for (int i = 0; i < size; ++i)
      if (all[i] == 0) failed.push_back(i);

    num_failed = static_cast<int>(failed.size());
    if (num_failed > 0) {
      std::cout << "TEST FAILED (" << display_name << ") on " << num_failed << "/" << size << " rank(s):";
      for (const int r : failed) std::cout << ' ' << r;
      std::cout << std::endl;
    }
  }
  MPI_Bcast(&num_failed, 1, MPI_INT, 0, comm);

  return num_failed > 0 ? 1 : 0;
}

inline int mpiExit(const Dune::TestSuite& suite, MPI_Comm comm = MPI_COMM_WORLD) { return mpiExit(suite, suite.name(), comm); }

/** @brief Runs @p body with a TestSuite and reduces the result across @p comm.
 *
 *  An exception escaping @p body becomes a failed check, so the collective reduction in mpiExit()
 *  still runs. This only rescues throws outside collective regions — a rank that throws while the
 *  others are blocked in an MPI call cannot be rescued, which is why checks inside collective
 *  regions must be non-fatal.
 */
template <class Body>
int runParallelTest(const std::string& name, Body&& body, MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // The rank goes into the suite name so that TestSuite's own per-rank output is attributable.
  Dune::TestSuite suite(name + " rank " + std::to_string(rank));

  try {
    body(suite);
  }
  catch (const Dune::Exception& e) {
    suite.check(false, "uncaught Dune::Exception") << e.what();
  }
  catch (const std::exception& e) {
    suite.check(false, "uncaught std::exception") << e.what();
  }

  return mpiExit(suite, name, comm);
}

/** @brief 2d Laplacian with Neumann boundary conditions on an @p nx x @p ny grid.
 *
 *  Collective-free; call it on the rank that should hold the matrix.
 */
template <class Matrix>
Matrix buildLaplacian2d(std::size_t nx, std::size_t ny)
{
  const std::size_t n = nx * ny;

  Matrix A;
  A.setBuildMode(Matrix::implicit);
  A.setImplicitBuildModeParameters(5, 0.2); // centre + 4 neighbours
  A.setSize(n, n);

  for (std::size_t iy = 0; iy < ny; ++iy)
    for (std::size_t ix = 0; ix < nx; ++ix) {
      const std::size_t i = iy * nx + ix;
      double degree = 0.0;
      if (ix > 0) {
        A.entry(i, i - 1) = -1.0;
        degree += 1.0;
      }
      if (ix + 1 < nx) {
        A.entry(i, i + 1) = -1.0;
        degree += 1.0;
      }
      if (iy > 0) {
        A.entry(i, i - nx) = -1.0;
        degree += 1.0;
      }
      if (iy + 1 < ny) {
        A.entry(i, i + nx) = -1.0;
        degree += 1.0;
      }
      A.entry(i, i) = degree;
    }
  A.compress();
  return A;
}

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

/* @brief Creates a communication object to communicate data that lives on the vertices of a parallel grid
 *
 * This function takes a grid view, then creates a unique global index for each vertex, and also
 * determines a owner for each vertex (vertices that exist on multiple ranks, the rank with the
 * smallest rank number is determined as the owner).
 *
 * If @p patch_mask is non-empty, only vertices for which the mask is true enter the index set (the
 * resulting communication then lives on the subdomain "patch" defined by the mask rather than on
 * the whole grid view). In that case the local indices are renumbered contiguously 0..n-1 in
 * ascending grid view index order, so that they match the contiguous renumbering the caller has
 * to apply to its matrix rows / vector entries. Ownership is still determined on the whole view,
 * which is safe as long as the owner of every kept vertex keeps it in its own patch — true
 * whenever the mask only drops vertices more than one element layer away from the rank's
 * subdomain (the owner of a vertex is always within one element layer of it). The remote indices
 * are rebuilt on the filtered index set, so ranks sharing no patch vertex simply end up with
 * empty remote lists.
 */
template <class GridView>
std::shared_ptr<Dune::OwnerOverlapCopyCommunication<int, int>> create_communication_for_grid(const GridView& gv, const std::vector<bool>& patch_mask = {})
{
  if (not patch_mask.empty() and patch_mask.size() != static_cast<std::size_t>(gv.size(GridView::dimension)))
    DUNE_THROW(Dune::InvalidStateException, "patch_mask has size " << patch_mask.size() << " but the grid view has " << gv.size(GridView::dimension) << " vertices");
  const bool filter = not patch_mask.empty();
  constexpr int dim = GridView::dimension;

  // Renumber the patch vertices contiguously in ascending grid view index order. The caller's
  // matrix/vector entries must follow exactly this numbering.
  std::vector<int> local_of_view;
  if (filter) {
    std::vector<int> patch_vertices;
    for (const auto& v : vertices(gv))
      if (patch_mask[gv.indexSet().index(v)]) patch_vertices.push_back(gv.indexSet().index(v));
    std::sort(patch_vertices.begin(), patch_vertices.end());

    local_of_view.assign(gv.size(dim), -1);
    for (std::size_t i = 0; i < patch_vertices.size(); ++i) local_of_view[patch_vertices[i]] = static_cast<int>(i);
  }

  using Communication = Dune::OwnerOverlapCopyCommunication<int, int>;
  auto comm = std::make_shared<Communication>();

  const int rank = gv.comm().rank();

  // Get a globally unique numbering of all vertices
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

  using Attribute = Dune::OwnerOverlapCopyAttributeSet;
  auto& parallel_idxs = comm->indexSet();
  parallel_idxs.beginResize();
  for (const auto& v : vertices(gv)) {
    auto gidx = gidxs.index(v);
    auto lidx = gv.indexSet().index(v);
    if (filter and not patch_mask[lidx]) continue;
    auto owned = owner[lidx] == rank;
    const int local = filter ? local_of_view[lidx] : lidx;
    // TODO: The last parameter of the local index structure is set to true here.
    // It is named `isPublic` and it seems to only influence the setup of the remote
    // index set below slightly (in the sense that setting this flag to `true` is
    // always correct, setting it to true only on those indices that are actually
    // communicated might slightly speed up the setup). For later operations it
    // doesn't seem to be used, so let's just leave it like that.
    parallel_idxs.add(gidx, {static_cast<std::size_t>(local), owned ? Attribute::owner : Attribute::copy, true});
  }
  parallel_idxs.endResize();
  comm->remoteIndices().template rebuild<true>();

  return comm;
}
} // namespace ddmtest
