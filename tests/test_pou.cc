#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_utils.hh"

#include <array>
#include <cmath>
#include <cstddef>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/ddm/helpers.hh>           // distributeMatrixFrom0
#include <dune/ddm/overlap_extension.hh> // create_overlapping_matrix
#include <dune/ddm/pou.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

/** @file
 *
 *  Parallel test for the partition-of-unity constructions in pou.hh: each one must sum to 1
 *  across all subdomains sharing a DOF.
 */

using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;

int main(int argc, char** argv)
{
  const auto& helper = Dune::MPIHelper::instance(argc, argv);
  const int rank = helper.rank();
  const int size = helper.size();

  // A single subdomain has no overlap and the whole problem is "core", which is a degenerate case
  // for the overlap-extension machinery; this test is meaningful only in parallel. The early exit
  // precedes every collective call, so it is safe to return directly.
  if (size < 2) {
    std::cerr << "This test must be run with at least 2 MPI ranks.\n";
    return ddmtest::skip_exit_code;
  }

  return ddmtest::runParallelTest("test_pou", [&](Dune::TestSuite& t) {
    // Assemble 2d Laplacian with pure Neumann boundary conditions
    const auto n = static_cast<std::size_t>(16 * std::ceil(std::sqrt(static_cast<double>(size))));
    const Matrix A = rank == 0 ? ddmtest::buildLaplacian2d<Matrix>(n, n) : Matrix{};

    // Partition and distribute the matrix
    auto [novlp_comm, localA, active] = distributeMatrixFrom0<Matrix, std::size_t>(A, MPI_COMM_WORLD, size);
    const bool all_active = ddmtest::allRanks(active);
    t.check(all_active, "every rank received part of the matrix") << "rank " << rank << (active ? " did" : " did NOT") << "; choose a larger problem size";
    if (!all_active) return;

    // Turn non-overlapping matrix into an overlapping one
    const int overlap = 4;
    auto [ovlp_comm, A_ovlp, boundary] = create_overlapping_matrix(*novlp_comm, *localA, overlap, MatrixRepresentation::Consistent);

    struct Case {
      PartitionOfUnityType type;
      const char* name;
    };
    const std::array<Case, 3> cases = {{
        {PartitionOfUnityType::Trivial, "trivial"},
        {PartitionOfUnityType::Standard, "standard"},
        {PartitionOfUnityType::Distance, "distance"},
    }};
    const std::array<int, 4> shrinks = {{0, 1, 2, 3}}; // shrink must be < overlap

    for (const int shrink : shrinks)
      for (const auto& c : cases) {
        PartitionOfUnity pou(*A_ovlp, *ovlp_comm, c.type, shrink, overlap);

        // Sum the partition across all subdomains sharing each DOF.
        Vector sum(pou.size());
        for (std::size_t i = 0; i < pou.size(); ++i) sum[i] = pou[i];
        ovlp_comm->addOwnerCopyToAll(sum, sum);

        // Owner mask so that DOFs shared by several subdomains are counted only once.
        std::vector<bool> owned(pou.size(), false);
        for (const auto& idx : ovlp_comm->indexSet())
          if (idx.local().attribute() == Dune::OwnerOverlapCopyAttributeSet::owner) owned[idx.local().local()] = true;

        // How many subdomains share each DOF: 2 => pairwise interface, >= 3 => cross point. Lets
        // us see whether failures are confined to cross points.
        Vector count(pou.size());
        for (std::size_t i = 0; i < pou.size(); ++i) count[i] = 1.0;
        ovlp_comm->addOwnerCopyToAll(count, count);

        double max_err = 0.0;
        std::array<unsigned long long, 6> bad_by_count = {{0, 0, 0, 0, 0, 0}}; // index = min(count, 5)
        unsigned long long n_bad = 0;
        for (std::size_t i = 0; i < sum.size(); ++i) {
          if (!owned[i]) continue;
          const double err = std::abs(sum[i][0] - 1.0);
          max_err = std::max(max_err, err);
          if (err > 1e-10) {
            ++n_bad;
            ++bad_by_count[std::min<std::size_t>(static_cast<std::size_t>(std::lround(count[i][0])), 5)];
          }
        }

        double global_err = 0.0;
        unsigned long long global_bad = 0;
        std::array<unsigned long long, 6> global_bad_by_count = {{0, 0, 0, 0, 0, 0}};
        MPI_Allreduce(&max_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&n_bad, &global_bad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(bad_by_count.data(), global_bad_by_count.data(), 6, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        const auto name = std::string("POU '") + c.name + "' (shrink=" + std::to_string(shrink) + ") sums to 1";
        t.check(global_err <= 1e-10, name) << "max deviation " << global_err << " over " << global_bad << " DOFs; by subdomain count: 2=" << global_bad_by_count[2]
                                           << " 3=" << global_bad_by_count[3] << " 4=" << global_bad_by_count[4] << " >=5=" << global_bad_by_count[5];
      }
  });
}
