# Avoid host views in the neighbour basis exchange and the coarse gather/scatter

- STATUS: OPEN
- PRIORITY: 70
- TAGS: gpu,backend,mpi

`include/ddm/impl/prec/galerkin.hh` moves data through host views in three places:
- `neighbour_basis()`: the send and receive buffers live on the backend of V, but the messages go through host views.
- Setting up the coarse problem: the coarse matrix values are gathered on rank 0 through a host view.
- `gather_coarse_defect()` and `scatter_coarse_solution()`: host views in every apply.

For the istl backend this costs nothing, but on a GPU backend it means a device-host copy per message. The exchanges of
a `Communication` already avoid this through the `MultiVec::do_make_exchange_plan` hook; these paths need the same
(device buffers with GPU-aware MPI, host staging otherwise).

Related: 20261126-092823 (GPU backend). The distributed coarse solve (20261126-092820) replaces the gather/scatter, so do
that first and only fix what remains.
