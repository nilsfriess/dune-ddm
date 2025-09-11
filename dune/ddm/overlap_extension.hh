#pragma once

#include "datahandles.hh"
#include "helpers.hh"
#include "logger.hh"

#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/mpitraits.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <limits>
#include <mpi.h>
#include <unordered_set>

template <class RemoteIndices, class Mat>

class ExtendedRemoteIndices {
  using GlobalIndex = typename RemoteIndices::GlobalIndex;
  using LocalIndex = typename RemoteIndices::LocalIndex;

  AttributeSet all_att{Attribute::owner, Attribute::copy};

public:
  using ParallelIndexSet = typename RemoteIndices::ParallelIndexSet;

  ExtendedRemoteIndices(const RemoteIndices &remoteids, const Mat &A, int overlap) : overlap(overlap)
  {
    assert(remoteids.sourceIndexSet().size() == A.N() && "Index set must match size of matrix");

    index_set_sizes.reserve(overlap + 1);
    index_set_sizes.push_back(A.N());

    // Create communicator data structures
    comm_if.build(remoteids, all_att, all_att);
    varcomm = std::make_unique<Dune::VariableSizeCommunicator<>>(comm_if);

    // Initialise the local-to-global and global-to-local maps
    ltg.resize(remoteids.sourceIndexSet().size());
    for (const auto &it : remoteids.sourceIndexSet()) {
      ltg[it.local().local()] = it.global();
      gis.insert(it.global());
    }

    // Initialise the "boundary distance map"
    IdentifyBoundaryDataHandle ibdh(A, remoteids.sourceIndexSet());
    varcomm->forward(ibdh);
    const auto &boundary_mask = ibdh.get_boundary_mask();

    boundary_distance.resize(remoteids.sourceIndexSet().size(), std::numeric_limits<int>::max() - 1);
    for (std::size_t i = 0; i < boundary_distance.size(); ++i) {
      if (boundary_mask[i]) {
        boundary_distance[i] = 0;
      }
    }

    for (int round = 0; round < overlap + 2; ++round) {
      for (std::size_t i = 0; i < boundary_distance.size(); ++i) {
        for (auto cit = A[i].begin(); cit != A[i].end(); ++cit) {
          auto nb_dist_plus_one = boundary_distance[cit.index()] + 1;
          if (nb_dist_plus_one < boundary_distance[i]) {
            boundary_distance[i] = nb_dist_plus_one;
          }
        }
      }
    }

    // Register events for logging
    extend_event = Logger::get().registerOrGetEvent("OverlapExtension", "extend overlap");

    // Actually extend the overlap
    extend_overlap(remoteids, A);

    // When we're done, we create a boundary mask on the overlapping index set
    ovlp_boundary_mask.resize(size(), false);
    auto n_indices_last_added = index_set_sizes[index_set_sizes.size() - 1] - index_set_sizes[index_set_sizes.size() - 2];
    for (std::size_t i = size() - n_indices_last_added; i < size(); ++i) {
      ovlp_boundary_mask[i] = true;
    }
  }

  const RemoteIndices &get_remote_indices() const { return *ext_indices.first; }
  const ParallelIndexSet &get_parallel_index_set() const { return *ext_indices.second; }
  RemoteParallelIndices<RemoteIndices> get_remote_par_indices() { return ext_indices; }
  int get_overlap() const { return overlap; }
  std::size_t size() const { return get_parallel_index_set().size(); }
  const std::vector<bool> &get_overlapping_boundary_mask() const { return ovlp_boundary_mask; }

  Dune::VariableSizeCommunicator<> &get_overlapping_communicator() const { return *varcomm; }

  const std::vector<std::size_t> &get_index_set_sizes() const { return index_set_sizes; }

  Mat create_overlapping_matrix(const Mat &A) const
  {
    auto *create_matrix_event = Logger::get().registerOrGetEvent("OverlapExtension", "create Matrix");
    auto *add_matrix_event = Logger::get().registerOrGetEvent("OverlapExtension", "add Matrix");

    Logger::get().startEvent(create_matrix_event);
    CreateMatrixDataHandle cmdh(A, get_parallel_index_set(), ltg, gis);
    varcomm->forward(cmdh);
    auto Aovlp = cmdh.getOverlappingMatrix();
    Logger::get().endEvent(create_matrix_event);

    Logger::get().startEvent(add_matrix_event);
    AddMatrixDataHandle amdh(A, Aovlp, get_parallel_index_set());
    varcomm->forward(amdh);
    Logger::get().endEvent(add_matrix_event);

    return Aovlp;
  }

  /** @brief Updates an overlapping matrix using a given non-overlapping matrix.
   *
   *  This function takes a non-overlapping matrix \p A and scatters its values
   *  into the overlapping matrix \p Aovlp (which would usually be obtained by
   *  create_overlapping_matrix(). This can be used in nonlinear problems, for
   *  instance, to avoid re-creating the matrix from scratch everytime.
   */
  Mat update_overlapping_matrix(const Mat &A, Mat &Aovlp) const
  {
    auto *add_matrix_event = Logger::get().registerOrGetEvent("OverlapExtension", "add Matrix");

    Logger::get().startEvent(add_matrix_event);
    Aovlp = 0;
    AddMatrixDataHandle amdh(A, Aovlp, get_parallel_index_set());
    varcomm->forward(amdh);
    Logger::get().endEvent(add_matrix_event);

    return Aovlp;
  }

private:
  // Creates a new parallel index set from the given index set, using the same local/global indices but the "public" state of the local indices
  // will be modified according to the functor isPublic(int local_index) -> bool. Note that indices that were public in the old index set will
  // also be set as public in the returned one.
  template <class ParallelIndexSet, class IsPublic>
  ParallelIndexSet modify_parindexset_public_state(const ParallelIndexSet &indexSet, IsPublic &&isPublic)
  {
    ParallelIndexSet newIndexSet;
    newIndexSet.beginResize();
    for (const auto &idx : indexSet) {
      newIndexSet.add(idx.global(), {idx.local().local(), idx.local().attribute(), idx.local().isPublic() or isPublic(idx.local().local())});
    }
    newIndexSet.endResize();
    return newIndexSet;
  }

  void extend_overlap(const RemoteIndices &remoteids, const Mat &A)
  {
    Logger::ScopedLog sl{extend_event};

    MPI_Comm comm = remoteids.communicator();
    int rank{};
    MPI_Comm_rank(comm, &rank);

    // For each index, we find out which other ranks also know that index
    Dune::BufferedCommunicator buffcomm;
    buffcomm.build<RankTuple>(comm_if);

    RankTuple rt;
    rt.rank = rank;
    rt.rankmap.resize(index_set_sizes[0]);
    buffcomm.forward<RankDataHandle>(rt);

    // Store all neighbours in one set
    std::set<int> nbs_set;
    for (const auto &ranks : rt.rankmap) {
      nbs_set.insert(ranks.begin(), ranks.end());
    }

    // Next, create a new parallel index set. This will be extended in each overlap extension round
    auto ext_pidxs = modify_parindexset_public_state(remoteids.sourceIndexSet(), [&](int li) { return boundary_distance[li] <= overlap; });

    // Create the remote indices
    std::vector<int> nbs(nbs_set.begin(), nbs_set.end());
    auto ext_rids = std::make_shared<RemoteIndices>(ext_pidxs, ext_pidxs, comm, nbs);
    ext_rids->template rebuild<false>();

    // Rebuild the communication data structures
    comm_if.free();
    comm_if.build(*ext_rids, all_att, all_att);
    varcomm = std::make_unique<Dune::VariableSizeCommunicator<>>(comm_if, 10 * 1024 * 1024); // This will reserve 10*1024*1024*64 bits \approx 80megabytes per rank

    IndexsetExtensionMatrixGraphDataHandle extdh(rank, A, gis);
    UpdateRankInfoDataHandle uprdh(rank);

    std::vector<MPI_Request> reqs;
    std::vector<std::size_t> sendcount;
    std::vector<std::size_t> recvcount;

    std::vector<std::vector<int>> new_nbs_data;
    std::vector<std::vector<int>> new_nbs_data_recv;

    for (int round = 0; round < overlap; ++round) {
      extdh.set_index_set(ltg);
      extdh.rankmap = rt.rankmap;
      extdh.updated_rankmap = rt.rankmap;
      varcomm->forward(extdh);

      index_set_sizes.push_back(ltg.size());

      uprdh.set_rankmap(extdh.updated_rankmap);
      varcomm->forward(uprdh);

      std::map<int, std::set<int>> new_nbs;
      for (const auto &ranks : extdh.updated_rankmap) {
        for (const auto &p : ranks) {
          for (const auto &q : ranks) {
            if (p != q and nbs_set.contains(p)) {
              new_nbs[p].insert(q);
            }
          }
        }
      }

      for (const auto &p : nbs_set) {
        if (not new_nbs.contains(p)) {
          new_nbs[p].insert(p);
        }
      }

      reqs.resize(2 * new_nbs.size());
      sendcount.resize(new_nbs.size());
      recvcount.resize(new_nbs.size());
      std::size_t i = 0; // Counter for reqs
      std::size_t j = 0; // Counter for send/recvcount
      for (const auto &[p, pnbs] : new_nbs) {
        sendcount[j] = pnbs.size();
        MPI_Irecv(&(recvcount[j]), 1, Dune::MPITraits<std::size_t>::getType(), p, 1, comm, &(reqs[i++]));
        MPI_Isend(&(sendcount[j++]), 1, Dune::MPITraits<std::size_t>::getType(), p, 1, comm, &(reqs[i++]));
      }
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

      new_nbs_data.resize(new_nbs.size());
      new_nbs_data_recv.resize(new_nbs.size());
      i = 0;
      j = 0;
      for (const auto &[p, pnbs] : new_nbs) {
        new_nbs_data[j].assign(pnbs.begin(), pnbs.end());
        new_nbs_data_recv[j].resize(recvcount[j]);

        MPI_Irecv(new_nbs_data_recv[j].data(), static_cast<int>(recvcount[j]), MPI_INT, p, 2, comm, &(reqs[i++]));
        MPI_Isend(new_nbs_data[j].data(), static_cast<int>(sendcount[j]), MPI_INT, p, 2, comm, &(reqs[i++]));
        j++;
      }
      MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

      j = 0;
      for (const auto &[p, pnbs] : new_nbs) {
        (void)p; // Silence warning
        for (const auto &q : new_nbs_data_recv[j++]) {
          if (q != rank) {
            nbs_set.insert(q);
          }
        }
      }

      ext_pidxs.beginResize();
      for (std::size_t i = index_set_sizes[round]; i < index_set_sizes[round + 1]; ++i) {
        ext_pidxs.add(ltg[i], {i, Attribute::copy, true});
      }
      ext_pidxs.endResize();

      ext_rids->setNeighbours(nbs_set);
      ext_rids->template rebuild<false>();

      comm_if.free();
      comm_if.build(*ext_rids, all_att, all_att);

      rt.rankmap.clear();
      rt.rankmap.resize(index_set_sizes[round + 1]);

      buffcomm.free();
      buffcomm.build<RankTuple>(comm_if);
      buffcomm.forward<RankDataHandle>(rt);
    }

    ext_indices = makeRemoteParallelIndices(ext_rids);
  }

  int overlap;
  std::vector<std::size_t> index_set_sizes;

  Dune::Interface comm_if;
  std::unique_ptr<Dune::VariableSizeCommunicator<>> varcomm;

  std::vector<GlobalIndex> ltg;        // Local to global map (modified during overlap extension)
  std::unordered_set<GlobalIndex> gis; // Global indices we know

  std::vector<int> boundary_distance; // Distance of dofs to the boundary

  std::vector<bool> ovlp_boundary_mask; // A mask for the dofs at the overlapping subdomain boundary

  Logger::Event *extend_event;

  RemoteParallelIndices<RemoteIndices> ext_indices;
};
