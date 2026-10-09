#pragma once

#include "ddm/check.hh"
#include "ddm/mat/mat.hh"
#include "ddm/overlap.hh"
#include "dune/ddm/logger.hh"

#include <algorithm>
#include <dune/common/parametertree.hh>
#include <memory>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
namespace ddm {
class Region {
public:
  static Region subdomain(const Overlap& ovlp) // TODO: Remove argument?
  {
    (void)ovlp;

    Region r;
    return r;
  }

  static Region ring(const Overlap& ovlp, int extend = 0)
  {
    Region r;
    r.region_to_subdomain_.reserve(2 * (ovlp.layer.size() - ovlp.n_original));

    return r;
  }

  // True if the region is the whole overlapping subdomain
  bool is_subdomain() const { return region_to_subdomain_.empty(); }

  Index subdomain_index(Index region_index) const
  {
    if (region_to_subdomain_.empty()) return region_index;
    else TODO("Region::subdomain_index");
  }

private:
  std::vector<Index> region_to_subdomain_;
};

template <class T>
class NeumannAssembler {
  struct Triple {
    Index row;
    Index col;
    T value;

    friend bool operator<(const Triple& t1, const Triple& t2) { return std::tie(t1.row, t1.col) < std::tie(t2.row, t2.col); }
  };

  // What we send per matrix entry: the row is implicit (see the constructor), the column is a position in send_idx
  struct Entry {
    Index col;
    T value;
  };

public:
  /** @brief Receive our neighbours' contributions to our overlapping subdomain
   *
   *  If the mat \p A was assembled element-by-element and value recording was enabled
   *  during assembly, then this exchanges all information that is necessary to build
   *  the Neumann matrix on the subdomain described by \p ovlp. The matrix is not built
   *  in this constructor; to build the matrix, use matrix(). Collective.
   */
  NeumannAssembler(std::shared_ptr<const Mat<T>> A, std::shared_ptr<const Overlap> ovlp)
      : A_(std::move(A))
      , ovlp_(std::move(ovlp))
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Neumann", "exchange")};
    DDM_CHECK(A_ != nullptr, "NeumannAssembler: matrix is nullptr");
    const auto* element_blocks = A_->element_blocks();
    DDM_CHECK(element_blocks != nullptr, "NeumannAssembler: Matrix was not assembled with -record_values enabled but this is required for Neumann matrix assembly");
    DDM_CHECK(A_->rows() == ovlp_->n_original, "NeumannAssembler: matrix has {} rows, but the overlap was created for {} indices", A_->rows(), ovlp_->n_original);

    const auto n_original = ovlp_->n_original; // The number of indices that were in the original index set

    const auto& pattern = ovlp_->comm->communication_pattern();
    const auto& shared = pattern.reduction_indices();
    MPI_Comm mpi_comm = pattern.communicator();

    // send_idx[peer] is the list of local indices that we share with peer. The `pos` vector holds the
    // inverse: pos[i] = k if send_idx[peer][k] == i, and -1 if i is not shared with that peer. It is
    // filled for one peer at a time and reset afterwards.
    std::vector<Index> pos(n_original, -1);
    std::vector<Triple> triples; // reused for all peers, clear() keeps the capacity
    std::unordered_map<int, detail::FlatLists<Entry>> out;
    for (const auto& [peer, idx] : shared) {
      for (std::size_t k = 0; k < idx.send_idx.size(); ++k) {
        // send_idx contains all the indices in the overlap region we share with peer.
        // We're only interested in those that we handled during assembly, which can
        // be checked by comparing them with n_original, because indices that were added
        // later (during overlap extension) will have larger local index.
        if (idx.send_idx[k] < n_original) pos[idx.send_idx[k]] = k;
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
          for (unsigned int n = 0; n < dofs.size(); ++n) triples.emplace_back(pos[dofs[m]], pos[dofs[n]], values[m * dofs.size() + n]);
      }

      for (auto i : idx.send_idx)
        if (i < n_original) pos[i] = -1;

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

    // List k from peer is the row recv_idx[k], and the columns are positions in recv_idx as well
    for (const auto& [peer, in] : detail::exchange_flat_lists(mpi_comm, shared, out)) {
      const auto& recv_idx = shared.at(peer).recv_idx;
      auto it = in.data.begin();
      for (std::size_t k = 0; k < in.counts.size(); ++k)
        for (int c = 0; c < in.counts[k]; ++c, ++it) received_.push_back({recv_idx[k], recv_idx[it->col], it->value});
    }

    // Several peers can contribute to the same entry
    sort_and_merge(received_);
  }

  /** @brief Get the Neumann matrix on the provided region
   *
   *  The region encodes two things: on which part of the subdomain the matrix is assembled
   *  and if that region is the region on which the matrix "lives" or if it lives on a larger
   *  region. This is important, e.g., in the GenEO coarse space: the right-hand side matrix
   *  there is only assembled in the overlap region but the size of the matrix still has to
   *  match the left-hand side matrix (which lives on the whole subdomain). For the ring coarse
   *  spaces, however, the matrix really only lives on that ring.
   *
   *  The local matrix is created with \p config (see create_local_mat_like(): without a type, it has the type of
   *  A's local matrix). Dirichlet conditions are not applied, the element blocks are recorded before them. Not
   *  collective, all communication happens in the constructor.
   **/
  std::shared_ptr<LocalMat<T>> matrix(const Dune::ParameterTree& config, const Region& r) const
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Neumann", "matrix")};
    // TODO: Sub-regions: only own blocks with all dofs in the region, and numbering/size according to the region
    if (!r.is_subdomain()) TODO("NeumannAssembler::matrix: regions other than the whole subdomain");

    // All our blocks are inside our overlapping subdomain (they only contain original indices), and received_
    // contains exactly the neighbours' blocks inside of it, so their sum is the Neumann matrix.
    const auto* blocks = A_->element_blocks();
    const Index n = std::ssize(ovlp_->layer);

    Pattern pattern(n, n);
    for (std::size_t b = 0; b < blocks->size(); ++b) {
      const auto dofs = blocks->dofs(b);
      for (auto i : dofs)
        for (auto j : dofs) pattern.add(i, j);
    }
    for (const auto& t : received_) pattern.add(t.row, t.col);
    pattern.finalize();

    auto N = create_local_mat_like<T>(config, pattern, A_->local());
    for (std::size_t b = 0; b < blocks->size(); ++b) {
      const auto dofs = blocks->dofs(b);
      N->add_values(dofs, dofs, blocks->values(b));
    }

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
  std::vector<Triple> received_; ///< the neighbours' contributions, in the local numbering of ovlp_, sorted by (row, col)
};
} // namespace ddm
