#!/bin/bash
# Batch job of submit_helix.sh: runs the runs file $2 with the executables and ini files in the out_dir $1, one after
# the other, each on all ranks of the job. Not meant to be called directly.
#
# The options of a run are parsed like bash arguments (quotes group words, e.g. -grid.cells "24 24 24"). The runs file
# is read on its own file descriptor and srun gets no stdin, otherwise srun would forward the rest of the runs file to
# the ranks. If the first run fails without writing its JSON, the job stops: then something is broken for all runs.
set -uo pipefail
out=$1
runs=$2
module purge
module load compiler/gnu/14.1 mpi/openmpi/4.1 numlib/mkl/2022.2 lib/boost/1.80.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PMIX_MCA_psec=native PMIX_MCA_gds=hash
echo "Job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES nodes ($SLURM_JOB_NODELIST), $SLURM_NTASKS ranks with $SLURM_CPUS_PER_TASK cores each, $(date)"
first=true
while read -r -u 3 name example inis options; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [ -f "$out/$name.json" ] && { echo "$name: done before"; continue; }
  config=""
  IFS=, read -ra files <<< "$inis"
  for f in "${files[@]}"; do config+="$out/ini/$f "; done
  eval "opts=( $options )"
  start=$(date +%s)
  # The CPU masks of the binding are printed into the log (verbose)
  timeout "${RUN_TIMEOUT:-60}m" srun --mpi=pmix --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=verbose,cores \
    "$out/$example" -config "$config" "${opts[@]}" -output.json "$out/$name.json" < /dev/null > "$out/$name.log" 2>&1
  status=$?
  its=$(grep -oP '^=== rate.*IT=\K[0-9]+' "$out/$name.log" | tail -1)
  echo "$name: exit $status, ${its:-no} iterations, $(( $(date +%s) - start )) s"
  if $first && [ ! -f "$out/$name.json" ]; then
    echo "The first run failed, stopping (see $out/$name.log)"
    exit 1
  fi
  first=false
done 3< "$runs"
echo "Finished $(date)"
