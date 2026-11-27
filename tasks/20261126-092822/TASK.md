# Several subdomains per rank in the two-level Schwarz preconditioner

- STATUS: OPEN
- PRIORITY: 80
- TAGS: roadmap,schwarz,coarse

Second half of the 2026-11-24 design, after 20261126-092819.

- Neumann matrices (`neumann.hh`): a subdomain's Neumann matrix needs to know which elements lie entirely inside it, and
  the current merged-triple exchange loses element identity. Instead, send whole element blocks (dof positions + dense
  block) for every element inside the rank's overlap, once, and filter them per subdomain into an `ElementBlocks`
  numbered by the subdomain.
- Coarse spaces (GenEO, MS-GFEM, ring): one eigenproblem per local subdomain. Coarse dofs are numbered per subdomain
  (Exscan over the subdomain counts) instead of per rank in `galerkin.hh`.
- Algebraic coarse spaces (20261126-092824) can take the per-subdomain Neumann approximation from the rank's overlapping matrix,
  without the element-block exchange.

Needed for 20261126-092823 (one rank per GPU); helps 20261126-092826 (coarse levels on fewer ranks).
