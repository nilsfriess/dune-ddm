#pragma once

#include "ddm/check.hh"
#include "ddm/index.hh"
#include "ddm/logger.hh"
#include "ddm/mat/assembly.hh"
#include "ddm/mat/mat.hh"
#include "ddm/overlap.hh"
#include "ddm/region.hh"

#include <cstddef>
#include <dune/common/parametertree.hh>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ddm {
/** Builds Neumann matrices on regions of our overlapping subdomain, e.g. the whole subdomain, a ring along its boundary
 *  or a local subdomain.
 *
 *  The Neumann matrix of a region is the sum of the element blocks of all ranks whose dofs all lie in the region. This
 *  needs a matrix A that was assembled element by element with value recording enabled, so that every rank has the
 *  blocks of its own elements. The constructor collects the blocks of our neighbours that lie inside of our overlapping
 *  subdomain, once. They are kept as separate blocks (not summed up), so that the Neumann matrix of any region can then
 *  be built without communication, see matrix().
 */
template <class T>
class NeumannAssembler {
public:
  /** Receives the element blocks of our neighbours with all dofs inside of our overlapping subdomain (described by
   *  ovlp), i.e. the blocks whose dofs the neighbour all shares with us. Collective.
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

    // Our blocks only contain original indices. A block goes to a peer if the peer shares all of its dofs with us. The
    // dofs are sent as positions in send_idx: by contract of the Communication class, the peer's recv_idx is ordered
    // in the same way, so indexing its recv_idx with them gives the dofs in the peer's numbering.
    //
    // exchange_flat_lists() sends one list per index in send_idx. A block is sent with the list of its first dof: the
    // dof list is the number of dofs followed by their positions, the value list the values of the block. The receiver
    // only needs the blocks, not the index whose list they came with.
    //
    // pos[i] = k if send_idx[k] == i for the current peer, -1 if i is not shared with it. It is filled for one peer at
    // a time and reset afterwards.
    std::vector<Index> pos(n_original, -1);
    std::unordered_map<int, detail::FlatLists<Index>> dofs_out;
    std::unordered_map<int, detail::FlatLists<T>> values_out;
    for (const auto& [peer, idx] : shared) {
      for (std::size_t k = 0; k < idx.send_idx.size(); ++k)
        if (idx.send_idx[k] < n_original) pos[idx.send_idx[k]] = static_cast<Index>(k);

      std::vector<std::vector<Index>> dof_lists(idx.send_idx.size());
      std::vector<std::vector<T>> value_lists(idx.send_idx.size());
      for (std::size_t b = 0; b < element_blocks->size(); ++b) {
        const auto dofs = element_blocks->dofs(b);
        bool send = !dofs.empty();
        for (auto d : dofs) send = send && pos[d] >= 0;
        if (!send) continue;

        auto& dof_list = dof_lists[pos[dofs[0]]];
        dof_list.push_back(std::ssize(dofs));
        for (auto d : dofs) dof_list.push_back(pos[d]);
        const auto values = element_blocks->values(b);
        value_lists[pos[dofs[0]]].insert(value_lists[pos[dofs[0]]].end(), values.begin(), values.end());
      }

      for (auto i : idx.send_idx)
        if (i < n_original) pos[i] = -1;

      // One list per index in send_idx, also for the ones without blocks: the peer expects as many lists as it has
      // indices in recv_idx
      auto& d = dofs_out[peer];
      auto& v = values_out[peer];
      for (std::size_t k = 0; k < idx.send_idx.size(); ++k) {
        d.counts.push_back(static_cast<int>(dof_lists[k].size()));
        d.data.insert(d.data.end(), dof_lists[k].begin(), dof_lists[k].end());
        v.counts.push_back(static_cast<int>(value_lists[k].size()));
        v.data.insert(v.data.end(), value_lists[k].begin(), value_lists[k].end());
      }
    }

    // The lists of a peer, concatenated, are its blocks one after the other, so they can be read without the counts
    const auto dofs_in = detail::exchange_flat_lists(mpi_comm, shared, dofs_out);
    const auto values_in = detail::exchange_flat_lists(mpi_comm, shared, values_out);
    std::vector<Index> dofs;
    for (const auto& [peer, in] : dofs_in) {
      const auto& recv_idx = shared.at(peer).recv_idx;
      const auto& values = values_in.at(peer).data;
      std::size_t v = 0;
      for (std::size_t k = 0; k < in.data.size();) {
        const auto n = static_cast<std::size_t>(in.data[k++]);
        dofs.clear();
        for (std::size_t m = 0; m < n; ++m) dofs.push_back(recv_idx[in.data[k++]]);
        DDM_ASSERT(v + n * n <= values.size(), "NeumannAssembler: peer {} sent fewer values than its blocks need", peer);
        received_.add(dofs, std::span(values.data() + v, n * n));
        v += n * n;
      }
      DDM_ASSERT(v == values.size(), "NeumannAssembler: peer {} sent {} values, but its blocks need {}", peer, values.size(), v);
    }
  }

  /** @brief Get the Neumann matrix on a region of our overlapping subdomain
   *
   *  The matrix lives on the region and is numbered by it: it is the sum of our and our neighbours' element blocks with
   *  all dofs in the region. For a region smaller than the subdomain (e.g. a ring), this gives Neumann conditions on
   *  its boundary towards the rest of the subdomain as well.
   *
   *  The local matrix is created with \p config (see create_local_mat_like(): without a type, it has the type of
   *  A's local matrix). Dirichlet conditions are not applied, the element blocks are recorded before them. Not
   *  collective, all communication happens in the constructor.
   **/
  std::shared_ptr<LocalMat<T>> matrix(const Dune::ParameterTree& config, const Region& region) const
  {
    Logger::ScopedLog sl{Logger::get().registerOrGetEvent("Neumann", "matrix")};
    DDM_CHECK(region.subdomain_size() == std::ssize(ovlp_->layer), "NeumannAssembler: the region was created for {} indices, but the overlap has {}", region.subdomain_size(), ovlp_->layer.size());

    // Calls f(dofs, values) for every block (ours and the received ones) with all dofs in the region, with its dofs
    // in the numbering of the region. Our blocks only contain original indices, which have the same number in the
    // overlapping index set, so both kinds of blocks are numbered by the overlapping index set.
    std::vector<Index> dofs;
    const auto for_each_block_in_region = [&](auto&& f) {
      for (const auto* blocks : {A_->element_blocks(), &received_}) {
        for (std::size_t b = 0; b < blocks->size(); ++b) {
          dofs.clear();
          bool inside = true;
          for (auto i : blocks->dofs(b)) {
            const auto k = region.region_index(i);
            inside = inside && k >= 0;
            if (!inside) break;
            dofs.push_back(k);
          }
          if (inside) f(std::span<const Index>(dofs), blocks->values(b));
        }
      }
    };

    Pattern pattern(region.size(), region.size());
    for_each_block_in_region([&](std::span<const Index> block_dofs, std::span<const T>) {
      for (auto i : block_dofs)
        for (auto j : block_dofs) pattern.add(i, j);
    });
    pattern.finalize();

    auto N = create_local_mat_like<T>(config, pattern, A_->local());
    for_each_block_in_region([&](std::span<const Index> block_dofs, std::span<const T> values) { N->add_values(block_dofs, block_dofs, values); });
    N->assemble();
    return N;
  }

private:
  std::shared_ptr<const Mat<T>> A_; ///< the matrix whose element blocks the Neumann matrices are built from
  std::shared_ptr<const Overlap> ovlp_;
  ElementBlocks<T> received_; ///< the neighbours' blocks inside of our overlapping subdomain, numbered by the overlapping index set
};
} // namespace ddm
