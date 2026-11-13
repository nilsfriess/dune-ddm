#!/bin/bash
#SBATCH --job-name=build_ddm
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/%u/dd30/build_%j.out
# Configures (if needed) and builds dune-ddm in ~/dune-ddm-talk/build on Helix, as a batch job (sbatch build_helix.sh
# ex2 ex6) or on the login node with 4 jobs (nice -n 10 bash build_helix.sh ex2 ex6). All
# parallel builds, including the bundled dependencies during the configuration, use the CPUs of the job.
# Targets as arguments, e.g. sbatch build.sh ex2 ex6
set -euo pipefail
jobs=${SLURM_CPUS_PER_TASK:-4}
module purge
module load devel/cmake/3.31.6 compiler/gnu/14.1 mpi/openmpi/4.1 numlib/mkl/2022.2 lib/boost/1.80.0
cd ~/dune-ddm-talk
mkdir -p build && cd build
if [ ! -f CMakeCache.txt ]; then
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-march=native" -DCMAKE_CXX_FLAGS="-march=native" \
    -DMKL_DIR=$MKLROOT/lib/cmake/mkl -DMKL_MPI=openmpi -DDDM_DEPS_JOBS=$jobs
else
  cmake . > /dev/null # picks up new targets (e.g. a new example)
fi
make -j $jobs "$@"
