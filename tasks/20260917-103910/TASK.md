# Make galerkin_product return a sparse block matrix

- STATUS: CLOSED
- PRIORITY: 100
- TAGS:

No description.

Closed on 2026-11-26: galerkin_product was removed in dfcf2a9; coarse_matrix_rows() returns block rows, and rank 0 assembles a sparse coarse matrix. A distributed coarse matrix is 20261126-092820 / 20261126-092826.
