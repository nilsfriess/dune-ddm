# Coarse matrix as a distributed additive Mat, then multilevel GenEO

- STATUS: OPEN
- PRIORITY: 50
- TAGS: roadmap,coarse,multilevel

- Because `Mat` is stored like MATIS, A_0 = sum_p Phi_p^T A_p^loc Phi_p, where Phi_p holds all coarse basis vectors on
  rank p's local dofs (`neighbour_basis()` already moves most of this). Every piece is SPSD, which is exactly the
  Neumann input GenEO needs on the next level (multilevel GenEO, Al Daas-Grigori-Jolivet-Tournier 2021). No algebraic
  splitting needed.
- Steps: (1) A_0 as a distributed additive `Mat` with its own `Communication`, on a subcommunicator; (2) two-level
  Schwarz recursively on it; the coarse levels run on fewer ranks (redistribution, or several subdomains per rank from
  20261126-092819/20261126-092822).
- The same container gives GDSW/AGDSW (shared coarse dofs, 2026-11-16 overview).
- Open research: MS-GFEM and Helmholtz on coarse levels.
- Before starting: write down a hypothesis and a deadline (RESEARCH_PLAN anti-scatter rule 3), and check 20261126-092820's results;
  a distributed direct coarse solve may be enough up to 1024 ranks.
