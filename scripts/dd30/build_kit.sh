#!/bin/bash
#SBATCH --job-name=build_ddm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --output=/home/hd/hd_hd/%u/dd30/build_%j.out
# Configures (if needed) and builds dune-ddm in ~/dune-ddm-talk/build-<toolchain> on bwUniCluster 3.0 (KIT), as a batch
# job on a compute node of the partition cpu (AMD EPYC 9454, like the login node, so -march=native fits both):
#
#   sbatch scripts/dd30/build_kit.sh llvm22 ex2 ex6
#   sbatch -p dev_cpu -t 00:30:00 --exclusive -c 192 --mem=376000 scripts/dd30/build_kit.sh gcc14 ex2 ex6  # whole node, sooner
#
# The toolchains are described in env_kit.sh. The login node only has one core, so build there only small changes
# (bash scripts/dd30/build_kit.sh llvm22 ex6). All parallel builds, including the bundled dependencies during the
# configuration, use one job per core of the job (Slurm counts hyperthreads: -c 64 are 32 cores). A build that hits
# the time limit continues where it stopped when submitted again.
set -euo pipefail
toolchain=$1
shift
jobs=$(( ${SLURM_CPUS_PER_TASK:-2} / 2 ))
repo=~/dune-ddm-talk
source "$repo/scripts/dd30/env_kit.sh" "$toolchain"
module list 2>&1
echo "Toolchain $toolchain: $DDM_CXX $DDM_FLAGS, on $(hostname), $jobs jobs, $(date)"
"$DDM_CXX" --version | head -1
# the bundled dependencies (configured during cmake) see the flags through the environment
export CFLAGS="$DDM_FLAGS" CXXFLAGS="$DDM_FLAGS"
mkdir -p "$repo/build-$toolchain" && cd "$repo/build-$toolchain"
if [ ! -f CMakeCache.txt ]; then
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$DDM_CC" -DCMAKE_CXX_COMPILER="$DDM_CXX" \
    -DCMAKE_C_FLAGS="-march=native $DDM_FLAGS" -DCMAKE_CXX_FLAGS="-march=native $DDM_FLAGS" \
    -DMKL_DIR="$MKLROOT/lib/cmake/mkl" -DMKL_MPI=openmpi -DDDM_DEPS_JOBS="$jobs"
else
  cmake . > /dev/null # picks up new targets (e.g. a new example)
fi
make -j "$jobs" "$@"
echo "Finished $(date)"
