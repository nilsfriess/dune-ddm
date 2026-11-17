#!/bin/bash
# Submits a list of runs of the examples as one batch job on bwUniCluster 3.0 (KIT), like submit_helix.sh on Helix.
# Run on a login node of the cluster from anywhere.
#
# Usage: scripts/dd30/submit_kit.sh <out_dir> <ranks> <time limit> <runs file>
#   e.g. scripts/dd30/submit_kit.sh ~/dd30/kit_check_P64_llvm22 64 01:00:00 scripts/dd30/runs/kit_check_P64.txt
#
# The runs file is the same as for submit_helix.sh. Before submitting, the executables (build-<toolchain>/examples/ex2,
# ex6), the ini files, the runs file, the commit and env_kit.sh are copied into <out_dir>, unless they are already
# there, and the job only uses these copies. With DEPEND=<job id> (e.g. a build job) the job waits until that job
# succeeded; executables that do not exist yet are then copied when the job starts.
#
# Nodes of the partition cpu: 2 x 48-core AMD EPYC 9454 (Slurm counts 192 hyperthreads), 384 GB. The job uses
# RANKS_PER_NODE ranks per node (default 24), each bound to CORES_PER_RANK cores (default 96 / RANKS_PER_NODE), with
# whole nodes (--exclusive). At most 3840 hyperthreads (20 nodes) per user at a time.
# Environment: TOOLCHAIN (llvm22 | gcc14, default llvm22), PARTITION (default cpu; dev_cpu: at most 2 nodes and 30
# min), RANKS_PER_NODE, CORES_PER_RANK, RUN_TIMEOUT (time limit of a single run in minutes, default 60), DEPEND,
# DRY_RUN.
set -euo pipefail
[ $# -eq 4 ] || { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "$1")
ranks=$2
time_limit=$3
runs=$(realpath "$4")
toolchain=${TOOLCHAIN:-llvm22}
build="$repo/build-$toolchain"
per_node=${RANKS_PER_NODE:-24}
[ "$ranks" -lt "$per_node" ] && per_node=$ranks
nodes=$(( (ranks + per_node - 1) / per_node ))
cores=${CORES_PER_RANK:-$(( 96 / per_node ))}
partition=${PARTITION:-cpu}

mkdir -p "$out"
if [ -f "$out/toolchain" ] && [ "$(cat "$out/toolchain")" != "$toolchain" ]; then
  echo "$out was made with toolchain $(cat "$out/toolchain"), not $toolchain" >&2
  exit 1
fi
echo "$toolchain" > "$out/toolchain"
echo "$build" > "$out/build_dir"
[ -f "$out/env_kit.sh" ] || cp "$repo/scripts/dd30/env_kit.sh" "$out/env_kit.sh"
for ex in ex2 ex6; do
  if [ ! -f "$out/$ex" ]; then
    if [ -f "$build/examples/$ex" ]; then cp "$build/examples/$ex" "$out/$ex"
    elif [ -z "${DEPEND:-}" ]; then echo "no executable $build/examples/$ex" >&2; exit 1
    fi
  fi
done
[ -d "$out/ini" ] || cp -r "$repo/examples/ini" "$out/ini"
runs_copy="$out/runs_$(date +%Y%m%d-%H%M%S).txt"
cp "$runs" "$runs_copy"
[ -f "$out/commit" ] || git -C "$repo" rev-parse HEAD > "$out/commit"

# Slurm counts hyperthreads: a rank gets 2 * CORES_PER_RANK of them, i.e. CORES_PER_RANK whole cores
# --exclusive without --mem gets the whole memory of the nodes (2000 MB per hyperthread); --mem=0 is rejected here
args=(--job-name="dd30_$(basename "$out")" --partition="$partition" --nodes="$nodes" --ntasks="$ranks"
      --ntasks-per-node="$per_node" --cpus-per-task="$(( 2 * cores ))" --time="$time_limit" --exclusive
      --output="$out/job_%j.out")
[ -n "${DEPEND:-}" ] && args+=(--dependency="afterok:$DEPEND")
echo "sbatch ${args[*]} $repo/scripts/dd30/job_kit.sh $out $runs_copy"
if [ "${DRY_RUN:-0}" != 1 ]; then
  RUN_TIMEOUT=${RUN_TIMEOUT:-60} sbatch "${args[@]}" "$repo/scripts/dd30/job_kit.sh" "$out" "$runs_copy"
fi
