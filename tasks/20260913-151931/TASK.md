# Support non-GPU-aware MPI

- STATUS: CLOSED
- PRIORITY: 100
- TAGS: gpu

Currently we assume GPU-aware MPI in this call
```cpp
MPI_Isend(send_bufs[peer].data(), (int)indices.send_idx.size(), mpi_type, peer, 4, pcomm, &requests.emplace_back());
```
We should come up with some kind of abstraction in the backend that ensures a copy to host is only carried out if necessary.

Closed on 2026-11-26: the exchanges of a Communication go through the backend hook MultiVec::do_make_exchange_plan (host staging by default, GPU backends override it). The remaining host-view paths (neighbour basis exchange, coarse gather/scatter in galerkin.hh) are part of 20261126-092823.
