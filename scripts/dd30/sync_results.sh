#!/bin/bash
# Copies the DD30 results from the clusters (~/dd30, the out_dirs of submit_helix.sh and submit_kit.sh) to
# <results>/raw and rebuilds <results>/index.html and <results>/runs.csv (build_index.py). Without the executables (ex2,
# ex6 and their backups) and the build logs; safe to repeat, and to run while jobs are still writing. The out_dirs of
# both clusters land side by side in raw/, so their names must differ (the KIT ones start with kit_). A cluster that
# cannot be reached is skipped with a warning.
#
# Usage: scripts/dd30/sync_results.sh [results dir]   (default ~/Dokumente/PhD/Conferences/Bogota/Talk/results)
# Environment: HOSTS (default "helix kit")
set -uo pipefail
results=${1:-$HOME/Dokumente/PhD/Conferences/Bogota/Talk/results}
mkdir -p "$results/raw"
for host in ${HOSTS:-helix kit}; do
  timeout 600 rsync -a --exclude='/*/ex[0-9]' --exclude='/*/ex[0-9].*' --exclude='/build*' --exclude='/*.log' \
    --exclude='/gdb*' --exclude='/*.py' --exclude='/*.sh' -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' \
    "$host:dd30/" "$results/raw/" || echo "WARNING: could not sync from $host" >&2
done
python3 "$(dirname "$0")/build_index.py" "$results"
