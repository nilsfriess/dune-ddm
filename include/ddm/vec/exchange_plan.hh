#pragma once

#include "ddm/communication_pattern.hh"
#include "ddm/vec/host_view.hh"

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ddm {
template <class T>
class MultiVec;

// The type of the reduction operation
enum class ReductionOperation : std::uint8_t {
  None,
  Addition,
};

// The backend-specific part of an exchange between ranks: holds the per-peer message buffers in the memory space of
// the vector and moves the values between the vector and these buffers. Created by MultiVec::make_exchange_plan() for
// a multivector with one column (the multivector of a Vec), the MPI communication itself is done by the caller.
template <class T>
class ExchangePlan {
public:
  using Buffers = std::unordered_map<int, std::span<T>>; // peer -> buffer

  virtual ~ExchangePlan() = default;
  ExchangePlan(const ExchangePlan&) = delete;
  ExchangePlan& operator=(const ExchangePlan&) = delete;

  // Gathers the values of v at the send indices into the send buffers. The buffers can be handed to MPI as soon as
  // this returns.
  virtual const Buffers& pack(MultiVec<T>& v) = 0;

  // The buffers the values from the peers are received into
  virtual const Buffers& recv_buffers() = 0;

  // Writes the received values to v at the recv indices, combined with the values in v according to op. The values
  // are in place when this returns.
  virtual void unpack(MultiVec<T>& v, ReductionOperation op) = 0;

protected:
  ExchangePlan() = default;
};

// Plan for vectors whose data can be accessed through host views, i.e. any vector. The buffers live in host memory.
template <class T>
class HostExchangePlan final : public ExchangePlan<T> {
  using Buffers = typename ExchangePlan<T>::Buffers;

public:
  explicit HostExchangePlan(const CommunicationPattern::IndexMap& idxs)
      : idxs_(idxs)
  {
    for (const auto& [peer, indices] : idxs_) {
      if (!indices.send_idx.empty()) send_storage_[peer].resize(indices.send_idx.size());
      if (!indices.recv_idx.empty()) recv_storage_[peer].resize(indices.recv_idx.size());
    }

    // The storage is not resized anymore, so the spans stay valid
    for (auto& [peer, buffer] : send_storage_) send_bufs_.emplace(peer, std::span<T>(buffer));
    for (auto& [peer, buffer] : recv_storage_) recv_bufs_.emplace(peer, std::span<T>(buffer));
  }

  const Buffers& pack(MultiVec<T>& v) override
  {
    const auto data = std::as_const(v).host_view(read);
    for (auto& [peer, buffer] : send_storage_) {
      const auto& send_idx = idxs_.at(peer).send_idx;
      for (std::size_t k = 0; k < send_idx.size(); ++k) buffer[k] = data[send_idx[k]];
    }
    return send_bufs_;
  }

  const Buffers& recv_buffers() override
  {
    return recv_bufs_;
  }

  void unpack(MultiVec<T>& v, ReductionOperation op) override
  {
    auto data = v.host_view(read_write);
    for (const auto& [peer, buffer] : recv_storage_) {
      const auto& recv_idx = idxs_.at(peer).recv_idx;
      for (std::size_t k = 0; k < recv_idx.size(); ++k) {
        switch (op) {
          case ReductionOperation::None: data[recv_idx[k]] = buffer[k]; break;
          case ReductionOperation::Addition: data[recv_idx[k]] += buffer[k]; break;
        }
      }
    }
  }

private:
  CommunicationPattern::IndexMap idxs_;

  std::unordered_map<int, std::vector<T>> send_storage_;
  std::unordered_map<int, std::vector<T>> recv_storage_;

  Buffers send_bufs_;
  Buffers recv_bufs_;
};
} // namespace ddm
