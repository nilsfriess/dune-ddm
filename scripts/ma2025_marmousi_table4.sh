#!/bin/bash
# Table 4 of Ma et al., "Two-level restricted additive Schwarz preconditioner based on multiscale spectral generalized
# FEM for heterogeneous Helmholtz problems", J. Sci. Comput. 105:99 (2025), arXiv:2409.06533, with ex5: GMRES
# iterations on the Marmousi model for the frequencies 5, 10 and 20 Hz and n_loc = 15, 20, 25 and 30 eigenvectors per
# subdomain, without (H/H* = 1) and with oversampling (H/H* = 0.86).
#
# Setup as in the paper: (0, 9 km) x (-3 km, 0), u = 0 on the top and impedance on the other sides, source at
# (6 km, 0), P2 with h = 7.5 m, M = 3 m^2 subdomains, overlap of 40 h / m (the oversampling has the same width), RAS
# with the coarse correction after the fine level, left-preconditioned GMRES with reduction 1e-6 and at most 100
# iterations (100 iterations means not converged, x in the paper).
# Differences: Dirichlet instead of impedance local solves, ParMETIS instead of uniform rectangular subdomains.
#
# Usage: scripts/ma2025_marmousi_table4.sh m   (m = 10, 20 or 40, runs 3 m^2 MPI ranks)
# EX5 and VEL set the executable and the velocity model (download it with data/download_models.sh). Writes one log per
# run and results.tsv into table4_m<m>/ in the current directory.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
EX5=$(realpath "${EX5:-$repo/build/examples/ex5}")
VEL=$(realpath "${VEL:-$repo/data/marmousi_vp.ini}")
m=$1
np=$((3 * m * m))
delta=$((40 / m)) # overlap in layers of elements

mkdir -p "table4_m$m"
cd "table4_m$m"
printf "method\tnu\toversampling\tnloc\titerations\n" > results.tsv
for method in saddle_point two_sided_ring; do
  args="-prec.coarse_space.method ${method%_ring}"
  [ "$method" = two_sided_ring ] && args="$args -prec.coarse_space.inner_product ring"
  for nu in 5 10 20; do
    for oversampling in 0 1; do
      for nloc in 15 20 25 30; do
        log="${method}_nu${nu}_os${oversampling}_n${nloc}.log"
        mpirun --oversubscribe --mca mpi_yield_when_idle 1 -np $np -x MKL_NUM_THREADS=1 -x OMP_NUM_THREADS=1 -x OPENBLAS_NUM_THREADS=1 -x LOG_LEVEL=info \
          "$EX5" -problem.velocity "$VEL" -problem.extent "9 3" -problem.source "6 3" -problem.free_surface true \
          -problem.frequency $nu -grid.cells "600 200" -grid.refine 1 \
          -prec.type tlschwarz -prec.coarse_space_type msgfem -prec.fine.variant restricted -prec.coarse_mode deflated \
          -prec.overlap $((delta * (1 + oversampling))) -prec.pou.layers $delta $args \
          -prec.coarse_space.eigensolver.shift 0 -prec.coarse_space.eigensolver.nev $nloc \
          -solver.verbosity 1 -solver.reduction 1e-6 -solver.maxit 100 -solver.restart 100 > "$log" 2>&1 || true
        its=$(grep -oP '^=== rate.*IT=\K[0-9]+' "$log" | tail -1 || true)
        printf "%s\t%s\t%s\t%s\t%s\n" "$method" "$nu" "$oversampling" "$nloc" "${its:-failed}" | tee -a results.tsv
      done
    done
  done
done
