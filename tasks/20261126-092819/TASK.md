# Several subdomains per rank in the one-level Schwarz preconditioner

- STATUS: OPEN
- PRIORITY: 95
- TAGS: roadmap,schwarz

Design from the 2026-11-24 session. The MPI rank stays the unit of communication, and the local subdomains become
`Region`s of the overlap the rank already builds. Every setup step and every apply does one MPI exchange per rank; the
per-subdomain work is a rank-local loop (later batched on the GPU, 20261126-092823).

- API: `A->set_local_subdomains(std::vector<std::vector<Index>>)`, attached like `set_near_kernel()`. Default: one set
  with all original indices, which is today's behaviour. The sets must cover all original indices.
- `Subdomain` = { `Region region`, per index its layer (distance to O_s, 0..L), global id from an Exscan }. The layers
  come from a local BFS on the overlapping matrix: its rows are complete below the outermost layer, so the BFS gives
  the global N_L(O_s) without new communication.
- PoU (`pou.hh`): add each subdomain's weights into one rank vector, then the one existing reduce.
- `SchwarzPrec`: A_s = `submatrix(A_U, region_s)`, eliminated boundary = `layer_s == L`. Apply: one broadcast, then per
  subdomain gather -> mask -> solve -> D_s -> scatter-add, then one reduce. `MultiVec` needs an additive `unpack_rows`.
- Subdomain-level code must use `Subdomain::layer`, not `ovlp->layer` or `n_original`.
- Logging: `Schwarz/subdomain dofs` should report max and sum.

Immediate use: 32-128 subdomains on the laptop or workstation with meaningful timings instead of oversubscribing.
Next: 20261126-092822 (two-level), 20261126-092823 (GPU).
