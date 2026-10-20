#pragma once

#include "ddm/check.hh"
#include "ddm/logger.hh"
#include "ddm/mat/mat.hh"
#include "ddm/overlap.hh"
#include "ddm/region.hh"

#include <algorithm>
#include <cstdint>
#include <dune/common/parametertree.hh>
#include <memory>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
namespace ddm {
template <class T>
class NeumannAssembler {
  struct Triple {
    Index row;
    Index col;
    T value;

    friend bool operator<(const Triple& t1, const Triple& t2)
    {
      return std::tie(t1.row, t1.col) < std::tie(t2.row, t2.col);
    }
  };

  // What we send per matrix entry: the row is implicit (see the constructor), the column is a position in send_idx
  struct Entry {
    Index col;
    T value;
  };

public:
  /** @brief Receive our neighbours' contributions to the Neumann matrix on a region of our overlapping subdomain
   *
   *  If the mat \p A was assembled element-by-element and value recording was enabled
   *  during assembly, then this exchanges all information that is necessary to build
   *  the Neumann matrix on the region \p region of the subdomain described by \p ovlp.
   *  Every rank tells its peers which of the indices they share lie in its region, and
   *  the peers only send their element blocks with all dofs inside of it. The matrix is
   *  not built in this constructor; to build the matrix, use matrix(). Collective.
   */
  NeumannAssembler(std::shared_ptr<const Mat<T>> A, std::shared_ptr<const Overlap> ovlp, Region region)
      : A_(std::move(A))
      , ovlp_(std::move(ovlp))
      , region_(std::move(region))
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Neumann", "exchange")};
    DDM_CHECK(A_ != nullptr, "NeumannAssembler: matrix is nullptr");
    const auto* element_blocks = A_->element_blocks();
    DDM_CHECK(element_blocks != nullptr, "NeumannAssembler: Matrix was not assembled with -record_values enabled but this is required for Neumann matrix assembly");
    DDM_CHECK(A_->rows() == ovlp_->n_original, "NeumannAssembler: matrix has {} rows, but the overlap was created for {} indices", A_->rows(), ovlp_->n_original);
    DDM_CHECK(region_.subdomain_size() == std::ssize(ovlp_->layer), "NeumannAssembler: the region was created for {} indices, but the overlap has {}", region_.subdomain_size(), ovlp_->layer.size());

    const auto n_original = ovlp_->n_original; // The number of indices that were in the original index set

    const auto& pattern = ovlp_->comm->communication_pattern();
    const auto& shared = pattern.reduction_indices();
    MPI_Comm mpi_comm = pattern.communicator();

    // First tell every peer which of the indices we share with it lie in our region: one list with a single flag per
    // index in send_idx. The flags we receive from a peer are in the order of our recv_idx.
    std::unordered_map<int, detail::FlatLists<std::uint8_t>> flags_out;
    for (const auto& [peer, idx] : shared) {
      auto& o = flags_out[peer];
      o.counts.assign(idx.send_idx.size(), 1);
      for (auto i : idx.send_idx) o.data.push_back(region_.region_index(i) >= 0);
    }
    const auto flags_in = detail::exchange_flat_lists(mpi_comm, shared, flags_out);

    // send_idx[peer] is the list of local indices that we share with peer. The `pos` vector holds the
    // inverse: pos[i] = k if send_idx[peer][k] == i, and -1 if i is not shared with that peer or not in
    // its region. in_peer_region[i] is the flag of the peer for i. Both are filled for one peer at a time
    // and reset afterwards.
    std::vector<Index> pos(n_original, -1);
    std::vector<std::uint8_t> in_peer_region(n_original, 0);
    std::vector<Triple> triples; // reused for all peers, clear() keeps the capacity
    std::unordered_map<int, detail::FlatLists<Entry>> out;
    for (const auto& [peer, idx] : shared) {
      if (const auto it = flags_in.find(peer); it != flags_in.end())
        for (std::size_t k = 0; k < idx.recv_idx.size(); ++k)
          if (idx.recv_idx[k] < n_original) in_peer_region[idx.recv_idx[k]] = it->second.data[k];

      for (std::size_t k = 0; k < idx.send_idx.size(); ++k) {
        // send_idx contains all the indices in the overlap region we share with peer.
        // We're only interested in those that we handled during assembly, which can
        // be checked by comparing them with n_original, because indices that were added
        // later (during overlap extension) will have larger local index.
        if (idx.send_idx[k] < n_original && in_peer_region[idx.send_idx[k]]) pos[idx.send_idx[k]] = k;
      }

      // Now loop over the element blocks and check if all dofs d satisfy pos[d] >= 0. In that case
      // that element contribution needs to be sent to peer.
      triples.clear();
      for (std::size_t i = 0; i < element_blocks->size(); ++i) {
        const auto dofs = element_blocks->dofs(i);

        // Check if all dofs are inside the overlap region with peer
        bool send = true;
        for (unsigned int m = 0; m < dofs.size(); ++m) send = send && pos[dofs[m]] >= 0;
        if (!send) continue;

        const auto values = element_blocks->values(i);
        for (unsigned int m = 0; m < dofs.size(); ++m)
          for (unsigned int n = 0; n < dofs.size(); ++n) triples.emplace_back(pos[dofs[m]], pos[dofs[n]], values[m + n * dofs.size()]);
      }

      for (auto i : idx.send_idx)
        if (i < n_original) pos[i] = -1;
      for (auto i : idx.recv_idx)
        if (i < n_original) in_peer_region[i] = 0;

      // Now we pack the data that we send away more tightly: First, we later don't need to send the rows
      // alongside the columns and the values; the receiver knows for which of their local indices it receives
      // data and that local index is exactly the row. Second, we can compress the Triples by sorting them by
      // (row, col) and then accumulating consecutive entries that have the same row and column. Lastly, there
      // is no need to introduce global indexing anywhere: We send the columns as indices within our send_idx
      // array; by contract of the Communication class, the corresponding recv_idx on the peer side is ordered
      // in the same way. In other words, if they use the received indices to index into their recv_idx, they
      // obtain the correct local index in their numbering.
      sort_and_merge(triples);

      // One list per index in send_idx, also for the ones without entries: the peer expects as many lists as
      // it has indices in recv_idx. The triples are sorted by row, so the lists come out in send_idx order.
      auto& o = out[peer];
      o.counts.assign(idx.send_idx.size(), 0);
      o.data.reserve(triples.size());
      for (const auto& t : triples) {
        ++o.counts[t.row];
        o.data.push_back({t.col, t.value});
      }
    }

    // List k from peer is the row recv_idx[k], and the columns are positions in recv_idx as well. The peers only sent
    // blocks inside of our region, so all of them have a number in it
    for (const auto& [peer, in] : detail::exchange_flat_lists(mpi_comm, shared, out)) {
      const auto& recv_idx = shared.at(peer).recv_idx;
      auto it = in.data.begin();
      for (std::size_t k = 0; k < in.counts.size(); ++k)
        for (int c = 0; c < in.counts[k]; ++c, ++it) {
          const auto row = region_.region_index(recv_idx[k]);
          const auto col = region_.region_index(recv_idx[it->col]);
          DDM_ASSERT(row >= 0 && col >= 0, "NeumannAssembler: peer {} sent the entry ({}, {}) outside of the region", peer, recv_idx[k], recv_idx[it->col]);
          received_.push_back({row, col, it->value});
        }
    }

    // Several peers can contribute to the same entry
    sort_and_merge(received_);
  }

  /** @brief Get the Neumann matrix on the region passed to the constructor
   *
   *  The matrix lives on the region and is numbered by it: it is the sum of our and our neighbours' element blocks with
   *  all dofs in the region. For a region smaller than the subdomain (e.g. a ring), this gives Neumann conditions on
   *  its boundary towards the rest of the subdomain as well.
   *
   *  The local matrix is created with \p config (see create_local_mat_like(): without a type, it has the type of
   *  A's local matrix). Dirichlet conditions are not applied, the element blocks are recorded before them. Not
   *  collective, all communication happens in the constructor.
   **/
  std::shared_ptr<LocalMat<T>> matrix(const Dune::ParameterTree& config) const
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Neumann", "matrix")};

    // All our blocks are inside our overlapping subdomain (they only contain original indices), and received_
    // contains exactly the neighbours' blocks inside of the region. So with our blocks inside of the region, their
    // sum is the Neumann matrix.
    const auto* blocks = A_->element_blocks();
    const Index n = region_.size();

    // The dofs of block b in the numbering of the region, false if one of them is outside of it
    std::vector<Index> dofs;
    const auto region_dofs = [&](std::size_t b) {
      dofs.clear();
      for (auto i : blocks->dofs(b)) {
        const auto k = region_.region_index(i);
        if (k < 0) return false;
        dofs.push_back(k);
      }
      return true;
    };

    Pattern pattern(n, n);
    for (std::size_t b = 0; b < blocks->size(); ++b) {
      if (!region_dofs(b)) continue;
      for (auto i : dofs)
        for (auto j : dofs) pattern.add(i, j);
    }
    for (const auto& t : received_) pattern.add(t.row, t.col);
    pattern.finalize();

    auto N = create_local_mat_like<T>(config, pattern, A_->local());
    for (std::size_t b = 0; b < blocks->size(); ++b)
      if (region_dofs(b)) N->add_values(dofs, dofs, blocks->values(b));

    // received_ is sorted by row, so it can be added one row at a time
    std::vector<Index> cols;
    std::vector<T> values;
    for (std::size_t k = 0; k < received_.size();) {
      const auto row = received_[k].row;
      cols.clear();
      values.clear();
      for (; k < received_.size() && received_[k].row == row; ++k) {
        cols.push_back(received_[k].col);
        values.push_back(received_[k].value);
      }
      N->add_values(std::span(&row, 1), cols, values);
    }

    N->assemble();
    return N;
  }

private:
  // Sorts the triples by (row, col) and sums up the values of triples with the same row and column
  static void sort_and_merge(std::vector<Triple>& triples)
  {
    std::sort(triples.begin(), triples.end());
    std::size_t n = 0;
    for (const auto& t : triples)
      if (n > 0 and triples[n - 1].row == t.row and triples[n - 1].col == t.col) triples[n - 1].value += t.value;
      else triples[n++] = t;
    triples.resize(n);
  }

  std::shared_ptr<const Mat<T>> A_; ///< the matrix whose element blocks the Neumann matrices are built from
  std::shared_ptr<const Overlap> ovlp_;
  Region region_;                ///< the region the Neumann matrix is built on
  std::vector<Triple> received_; ///< the neighbours' contributions, in the numbering of region_, sorted by (row, col)
};
} // namespace ddm
