#pragma once

#include "datahandles.hh"
#include "helpers.hh"
#include "logger.hh"

#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>

// Creates a new parallel index set from the given index set, using the same local/global indices but the "public" state of the local indices
// will be modified according to the functor isPublic(int local_index) -> bool.
template <class ParallelIndexSet, class IsPublic>
ParallelIndexSet modifyIndexSetPublicState(const ParallelIndexSet &indexSet, const IsPublic &isPublic)
{
  ParallelIndexSet newIndexSet;
  newIndexSet.beginResize();
  for (const auto &idx : indexSet) {
    newIndexSet.add(idx.global(), {idx.local().local(), idx.local().attribute(), isPublic(idx.local().local())});
  }
  newIndexSet.endResize();
  return newIndexSet;
}

template <class ParallelIndexSet, class RemoteIndices, class Mat>
void extendOverlapOnce(ParallelIndexSet &paridxs, RemoteIndices &remoteids, std::vector<int> &neighbours, const Mat &A)
{
  const AttributeSet allAttributes{Attribute::owner, Attribute::copy};

  MPI_Comm comm = remoteids.communicator();
  int rank{};
  MPI_Comm_rank(comm, &rank);

  Dune::Interface interface;
  interface.build(remoteids, allAttributes, allAttributes);
  Dune::VariableSizeCommunicator communicator(interface);

  // For each public index, we store a set of MPI ranks that also know this index, which is populated ine the following data handle.
  std::map<int, std::set<int>> neighbours_for_index;
  RankDataHandle rdh(rank, neighbours_for_index);
  communicator.forward(rdh);

  // Next, create a map that stores (for all local public indices) the (local) indices connected to it (i.e., the new overlap indices).
  // We include the index itself which is needed to setup the next map.
  std::map<int, std::set<int>> connected_indices;
  for (const auto &idxpair : paridxs) {
    auto li = idxpair.local();
    if (li.isPublic() && li < A.N()) // If index is public and within our original index set
    {
      // TODO: Skip indices that we already sent (can be identified as having a lower distance to the boundary than li).
      for (auto cit = A[li].begin(); cit != A[li].end(); ++cit) {
        connected_indices[li].insert(cit.index());
      }
    }
  }

  // For all indices determined above, collect all ranks (that we know of) that will know this index after overlap increase.
  // Note that we might learn about new ranks that own a certain index that we will send off only after already having sent
  // some information. Therefore, a second communication will be necessary to propagate this information.
  std::map<int, std::set<int>> connected_ranks;
  for (const auto &[_, ovlp_lis] : connected_indices) {
    for (auto li : ovlp_lis) {
      if (connected_ranks.contains(li)) {
        continue; // We already handled that index
      }

      for (auto cit = A[li].begin(); cit != A[li].end(); ++cit) {
        if (neighbours_for_index.count(cit.index()) > 0) {
          connected_ranks[li].insert(neighbours_for_index[cit.index()].begin(), neighbours_for_index[cit.index()].end());
        }
      }
    }
  }

  // Now send the data collected above.
  DataHandle dh(A, paridxs, connected_indices, connected_ranks, rank);
  communicator.forward(dh);

  // For all indices we just sent, check if we received messages about additional ranks that now know this index.
  Dune::GlobalLookupIndexSet glis(paridxs);
  std::map<int, std::set<int>> additional_neighbours;
  for (const auto &[gi, nbs] : dh.neighbours_for_gidx) {
    if (not paridxs.exists(gi)) {
      continue; // We already care about indices that we already knew
    }

    const auto li = paridxs[gi].local().local();
    if (connected_ranks.count(li) > 0) {
      // We now know that we received some rank numbers for this index, and we knew some before.
      // We add the difference to a new set that can be sent later.
      std::set<int> new_neighbours;
      new_neighbours.insert(nbs.begin(), nbs.end());
      new_neighbours.insert(connected_ranks[li].begin(), connected_ranks[li].end());
      new_neighbours.erase(rank);
      if (new_neighbours.size() > 0) {
        additional_neighbours[li] = std::move(new_neighbours);
      }
    }
  }

  // From this information, we build a map  int -> { int }, that stores all neighbours that we now know of for a specific rank.
  std::map<int, std::set<int>> combined_new_neighbours;
  for (auto nb : neighbours) {
    for (const auto &[_, nbs] : additional_neighbours) {
      if (nbs.contains(nb)) {
        combined_new_neighbours[nb].insert(nbs.begin(), nbs.end());
      }
    }
  }

  std::map<int, std::vector<int>> new_neighbours;
  for (const auto &[orig_nb, nbs] : combined_new_neighbours) {
    new_neighbours[orig_nb].assign(nbs.begin(), nbs.end());
  }

  std::vector<MPI_Request> requests;
  requests.reserve(neighbours.size());
  for (const auto &orig_nb : neighbours) {
    MPI_Isend(new_neighbours[orig_nb].data(), static_cast<int>(new_neighbours[orig_nb].size()), MPI_INT, orig_nb, 0, comm, &requests.emplace_back());
  }

  std::map<int, std::vector<int>> other_neighbours;
  for (const auto &orig_nb : neighbours) {
    MPI_Status status;
    int count{};

    MPI_Probe(orig_nb, 0, comm, &status);
    MPI_Get_count(&status, MPI_INT, &count);
    other_neighbours[orig_nb].resize(count);

    MPI_Recv(other_neighbours[orig_nb].data(), count, MPI_INT, orig_nb, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

  // Combine all neighbours into a single set
  std::set<int> final_neighbours;
  final_neighbours.insert(neighbours.begin(), neighbours.end()); // The ones we knew before

  for (const auto &[_, nbs] : dh.neighbours_for_gidx) { // The ones we learned about during the first communication
    final_neighbours.insert(nbs.begin(), nbs.end());
  }

  for (auto &[_, nbs] : other_neighbours) { // The ones we learned about during the second communication
    final_neighbours.insert(nbs.begin(), nbs.end());
  }

  final_neighbours.erase(rank);

  neighbours.resize(0);
  neighbours.reserve(final_neighbours.size());
  for (auto it = final_neighbours.begin(); it != final_neighbours.end();) {
    neighbours.push_back(final_neighbours.extract(it++).value());
  }

  // Finally, update the parallel index set
  auto size_before = paridxs.size();
  int cnt = 0;
  paridxs.beginResize();
  for (auto gi : dh.gis) {
    if (paridxs.exists(gi)) {
      continue;
    }
    paridxs.add(gi, {size_before + cnt, Attribute::copy, true});
    cnt++;
  }
  paridxs.endResize();

  // No need to rebuild the remote indices here, we will modify
  // the parallel index set in the extendOverlap loop again anyway,
  // so just rebuild there.
}

// TODO: This function assumes that only the local indices at the subdomain boundary are marked public, this can be avoided.
template <class RemoteIndices, class Mat>
RemoteParallelIndices<RemoteIndices> extendOverlap(const RemoteIndices &remoteids, const Mat &A, int overlap)
{
  static auto *overlap_family = Logger::get().registerFamily("OverlapExtension");
  static auto *extend_event = Logger::get().registerEvent(overlap_family, "extend");
  Logger::ScopedLog sl(extend_event);

  MPI_Comm comm = remoteids.communicator();

  auto paridxs = remoteids.sourceIndexSet();
  auto nextIndexSet = paridxs;
  std::vector<int> neighbours(remoteids.getNeighbours().begin(), remoteids.getNeighbours().end());
  auto nextRemoteIds = std::make_shared<RemoteIndices>(paridxs, paridxs, comm, neighbours);
  nextRemoteIds->template rebuild<false>();

  const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  Dune::Interface interface;
  interface.build(remoteids, allAttributes, allAttributes);
  Dune::VariableSizeCommunicator communicator(interface);
  IdentifyBoundaryDataHandle ibdh(A, paridxs);
  communicator.forward(ibdh);
  std::vector<bool> boundaryMask(ibdh.extractBoundaryMask());

  std::vector<int> boundary_dst(paridxs.size(), std::numeric_limits<int>::max() - 1);
  for (const auto &idxpair : paridxs) {
    auto li = idxpair.local();
    if (boundaryMask[li]) {
      boundary_dst[li] = 0;
    }
  }

  for (int round = 0; round <= overlap + 3; ++round) {
    for (int i = 0; i < boundary_dst.size(); ++i) {
      for (auto cIt = A[i].begin(); cIt != A[i].end(); ++cIt) {
        boundary_dst[i] = std::min(boundary_dst[i], boundary_dst[cIt.index()] + 1); // Increase distance from boundary by one
      }
    }
  }

  for (int round = 0; round < overlap; ++round) {
    auto max_local_idx = nextIndexSet.size();
    extendOverlapOnce(nextIndexSet, *nextRemoteIds, neighbours, A);

    // We mark those indices as public, that (i) just were added during the last round of overlap generation,
    // or (ii) have to be communicated during the next round of overlap generation. Only public indices are
    // considered in the data handles, so this way we save communication overhead.
    nextIndexSet = modifyIndexSetPublicState(nextIndexSet, [&](int li) {
      if (li < boundary_dst.size() && boundary_dst[li] <= round + 2) {
        return true;
      }

      if (li >= max_local_idx) {
        return true;
      }

      return false;
    });

    nextRemoteIds->setIndexSets(nextIndexSet, nextIndexSet, comm, neighbours);
    nextRemoteIds->template rebuild<false>();
  }

  Dune::GlobalLookupIndexSet glis(paridxs);

  // After we're done, we have to mark all indices in the overlap region and all indices that were added as public.
  nextIndexSet = modifyIndexSetPublicState(nextIndexSet, [&](int li) {
    if (li < boundary_dst.size() && boundary_dst[li] <= overlap + 3) {
      return true;
    }

    if (li >= paridxs.size()) {
      return true;
    }

    if (glis.pair(li)->local().isPublic()) {
      return true;
    }

    return false;
  });

  nextRemoteIds->setIndexSets(nextIndexSet, nextIndexSet, comm, neighbours);

  // Wrap both the RemoteIndices and the corresponding ParallelIndexSet in one object, because
  // the RemoteIndices only store a pointer to the ParallelIndexSet (which is a temporary)
  return makeRemoteParallelIndices(nextRemoteIds);
}

template <class Matrix, class RemoteIndices>
Matrix createOverlappingMatrix(const Matrix &A, const RemoteIndices &remoteids)
{
  static auto *overlap_family = Logger::get().registerOrGetFamily("OverlapExtension");
  static auto *matrix_event = Logger::get().registerEvent(overlap_family, "create Matrix");
  Logger::ScopedLog sl(matrix_event);

  const AttributeSet allAttributes{Attribute::owner, Attribute::copy};
  Dune::Interface interface;
  interface.build(remoteids, allAttributes, allAttributes);
  Dune::VariableSizeCommunicator communicator(interface);

  CreateMatrixDataHandle cmdh(A, remoteids.sourceIndexSet());
  communicator.forward(cmdh);
  auto Aovlp = cmdh.getOverlappingMatrix();

  AddMatrixDataHandle amdh(A, Aovlp, remoteids.sourceIndexSet());
  communicator.forward(amdh);

  // Check for rows with zeros everywhere, except on the diagonal. Set the diagonal entry to one.
  // TODO: Check if this is actually correct or find a better way to prevent that this happens.
  //       Should this be considered a bug in PDELab?
  for (auto rIt = Aovlp.begin(); rIt != Aovlp.end(); ++rIt) {
    bool dirichlet_row = true;
    for (auto cIt = rIt->begin(); cIt != rIt->end(); ++cIt) {

      if (cIt.index() == rIt.index()) {
        dirichlet_row &= (*cIt != 0.0);
      }
      else {
        dirichlet_row &= (*cIt == 0.0);
      }
    }

    if (dirichlet_row) {
      for (auto cIt = rIt->begin(); cIt != rIt->end(); ++cIt) {
        *cIt = (cIt.index() == rIt.index()) ? 1.0 : 0.0;
      }
    }
  }

  return Aovlp;
}
