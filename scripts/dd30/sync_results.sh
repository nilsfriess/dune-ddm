#!/bin/bash
# Copies the DD30 results from helix (~/dd30, the out_dirs of submit_helix.sh) to <results>/raw and rebuilds
# <results>/index.html and <results>/runs.csv (build_index.py). Without the executables (ex2, ex6 and their backups) and
# the build logs; safe to repeat, and to run while jobs are still writing.
#
# Usage: scripts/dd30/sync_results.sh [results dir]   (default ~/Dokumente/PhD/Conferences/Bogota/Talk/results)
set -euo pipefail
results=${1:-$HOME/Dokumente/PhD/Conferences/Bogota/Talk/results}
mkdir -p "$results/raw"
rsync -a --exclude='/*/ex[0-9]' --exclude='/*/ex[0-9].*' --exclude='/build*' --exclude='/*.log' --exclude='/gdb*' \
  --exclude='/*.py' --exclude='/*.sh' helix:dd30/ "$results/raw/"
python3 "$(dirname "$0")/build_index.py" "$results"
