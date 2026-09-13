# Support non-GPU-aware MPI

- STATUS: OPEN
- PRIORITY: 100
- TAGS: gpu

Currently we assume GPU-aware MPI in this call
```cpp
MPI_Isend(send_bufs[peer].data(), (int)indices.send_idx.size(), mpi_type, peer, 4, pcomm, &requests.emplace_back());
```
We should come up with some kind of abstraction in the backend that ensures a copy to host is only carried out if necessary.
