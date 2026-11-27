# Let Communication exchange lists, not only scalars

- STATUS: OPEN
- PRIORITY: 40
- TAGS: mpi

Building the overlap (`include/ddm/overlap.hh`, around the "Add the local rows of the other holders" step) needs to
exchange a list per shared index. The reduction plan of `Communication` has exactly the right index information, but
`Communication` can only exchange scalars, so the lists go through the separate helpers `detail::exchange_lists` and
`exchange_flat_lists` in `overlap.hh`.

Consider extending `Communication` to exchange lists (variable-length data per index), so these helpers go away.
