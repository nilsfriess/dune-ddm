#!/bin/bash
# Pilot for the DD30 experiments: are the new MS-GFEM variants (two-sided with eta = omega and eta = ring) as robust as
# the saddle point baseline on ex2, while RAS and Nicolaides are not? Only the iteration counts matter here, so it runs
# on a workstation (timings are meaningless with oversubscribed cores).
#
# Sweeps kappa_dark (contrast) and velocity_scale (Peclet number) for all methods in examples/ini/methods/, with the
# options of examples/ini/ex2.ini. Writes one JSON and one log per run into OUT; finished runs are not repeated. The
# executable and the ini files are copied into OUT first and the runs use the copies, so that rebuilding or editing
# them while the script runs does not change the runs (and OUT documents what produced the results).
#
# Usage: scripts/dd30/pilot_2d.sh
# Environment (defaults in brackets): EX2 [build/examples/ex2], MPIRUN [mpirun], NP [16], OUT [pilot_2d],
#   KAPPAS [1 1e-2 1e-4 1e-6], VELOCITIES [1 10 100], METHODS [ras nicolaides msgfem_saddle_point msgfem msgfem_ring],
#   EXTRA [] (more options for all runs, e.g. "-grid.refine 2")
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
EX2=$(realpath "${EX2:-$repo/build/examples/ex2}")
MPIRUN=${MPIRUN:-mpirun}
NP=${NP:-16}
OUT=${OUT:-pilot_2d}
KAPPAS=${KAPPAS:-1 1e-2 1e-4 1e-6}
VELOCITIES=${VELOCITIES:-1 10 100}
METHODS=${METHODS:-ras nicolaides msgfem_saddle_point msgfem msgfem_ring}
EXTRA=${EXTRA:-}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"
[ -f "$OUT/ex2" ] || cp "$EX2" "$OUT/ex2"
[ -d "$OUT/ini" ] || cp -r "$repo/examples/ini" "$OUT/ini"
git -C "$repo" rev-parse HEAD > "$OUT/commit"
EX2=$(realpath "$OUT/ex2")
ini=$(realpath "$OUT/ini")
for kappa in $KAPPAS; do
  for vel in $VELOCITIES; do
    for method in $METHODS; do
      name="${method}_k${kappa}_v${vel}"
      [ -f "$OUT/$name.json" ] && continue
      $MPIRUN -np "$NP" "$EX2" -config "$ini/ex2.ini $ini/methods/$method.ini" \
        -grid.cells "128 128" -grid.refine 1 -problem.kappa_dark "$kappa" -problem.velocity_scale "$vel" \
        -timing.barriers false -output.json "$OUT/$name.json" $EXTRA > "$OUT/$name.log" 2>&1 || echo "$name failed, see $OUT/$name.log"
      echo "$name: $(grep -oP '^=== rate.*IT=\K[0-9]+' "$OUT/$name.log" | tail -1 || echo -)"
    done
  done
done
