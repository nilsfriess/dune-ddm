#!/bin/bash
# Batch job of submit_helix.sh: runs the runs file $2 with the executables and ini files in the out_dir $1, one after
# the other, each on all ranks of the job. Not meant to be called directly.
set -uo pipefail
out=$1
runs=$2
module purge
module load compiler/gnu/14.1 mpi/openmpi/4.1 numlib/mkl/2022.2 lib/boost/1.80.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PMIX_MCA_psec=native PMIX_MCA_gds=hash
echo "Job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES nodes ($SLURM_JOB_NODELIST), $SLURM_NTASKS ranks, $(date)"
while read -r name example inis options; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [ -f "$out/$name.json" ] && { echo "$name: done before"; continue; }
  config=""
  IFS=, read -ra files <<< "$inis"
  for f in "${files[@]}"; do config+="$out/ini/$f "; done
  start=$(date +%s)
  # shellcheck disable=SC2086
  timeout "${RUN_TIMEOUT:-60}m" srun --mpi=pmix --cpu-bind=cores "$out/$example" -config "$config" $options -output.json "$out/$name.json" \
    > "$out/$name.log" 2>&1
  status=$?
  its=$(grep -oP '^=== rate.*IT=\K[0-9]+' "$out/$name.log" | tail -1)
  echo "$name: exit $status, ${its:-no} iterations, $(( $(date +%s) - start )) s"
done < "$runs"
echo "Finished $(date)"
