#!/bin/bash
# Submits a list of runs of the examples as one batch job on Helix (bwForCluster), so that they share the wait in the
# queue. Run on a login node of Helix from anywhere.
#
# Usage: scripts/dd30/submit_helix.sh <out_dir> <ranks> <time limit> <runs file>
#   e.g. scripts/dd30/submit_helix.sh ~/dd30/pilot_3d 64 02:00:00 scripts/dd30/runs/pilot_3d.txt
#
# The runs file has one run per line (empty lines and lines starting with # are skipped):
#   <name> <example> <ini files, comma separated, relative to examples/ini> [options...]
#   e.g. ring_k1 ex6 ex6.ini,methods/msgfem_ring.ini -problem.kappa_light 1
# Every run writes <out_dir>/<name>.json (see output.json) and <out_dir>/<name>.log; runs whose JSON exists are skipped,
# so the same runs file can be submitted again to finish or repeat (in another out_dir) a list.
#
# Before submitting, the executables (build/examples/ex2, ex6), the ini files, the runs file and the commit are copied
# into <out_dir>, unless they are already there, and the job only uses these copies: rebuilding or editing the sources
# does not change a submitted job, and <out_dir> documents what produced the results.
#
# The job uses RANKS_PER_NODE ranks per node (default 16, more give little speedup because of the memory bandwidth),
# each bound to CORES_PER_RANK cores (default 64 / RANKS_PER_NODE, i.e. its own share of the caches), with whole nodes
# (--exclusive). Environment: RANKS_PER_NODE, RUN_TIMEOUT (time limit of a single run in minutes, default 60), DRY_RUN.
set -euo pipefail
[ $# -eq 4 ] || { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "$1")
ranks=$2
time_limit=$3
runs=$(realpath "$4")
per_node=${RANKS_PER_NODE:-16}
[ "$ranks" -lt "$per_node" ] && per_node=$ranks
nodes=$(( (ranks + per_node - 1) / per_node ))
cores=${CORES_PER_RANK:-$(( 64 / per_node ))}
partition=cpu-multi
[ "$nodes" -eq 1 ] && partition=cpu-single

mkdir -p "$out"
for ex in ex2 ex6; do [ -f "$out/$ex" ] || cp "$repo/build/examples/$ex" "$out/$ex"; done
[ -d "$out/ini" ] || cp -r "$repo/examples/ini" "$out/ini"
runs_copy="$out/runs_$(date +%Y%m%d-%H%M%S).txt"
cp "$runs" "$runs_copy"
[ -f "$out/commit" ] || git -C "$repo" rev-parse HEAD > "$out/commit"

args=(--job-name="dd30_$(basename "$out")" --partition="$partition" --nodes="$nodes" --ntasks="$ranks"
      --ntasks-per-node="$per_node" --cpus-per-task="$cores" --time="$time_limit" --exclusive --mem=0
      --output="$out/job_%j.out")
echo "sbatch ${args[*]} $repo/scripts/dd30/job_helix.sh $out $runs_copy"
if [ "${DRY_RUN:-0}" != 1 ]; then
  RUN_TIMEOUT=${RUN_TIMEOUT:-60} sbatch "${args[@]}" "$repo/scripts/dd30/job_helix.sh" "$out" "$runs_copy"
fi
