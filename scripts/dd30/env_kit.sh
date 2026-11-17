# Environment of the DD30 builds and runs on bwUniCluster 3.0 (KIT), sourced by build_kit.sh and job_kit.sh with the
# toolchain as argument (the name of the build directory build-<toolchain> and of the out_dirs):
#
#   gcc14   GCC 14.2 (Helix has GCC 14.1), OpenMPI 5.0 and MKL 2025.0 for PARDISO (the default and only one)
#
# (Clang 22 was tried on 2026-11-17 and dropped: its executables are PIE by default, the gfortran objects of the bundled
# STRUMPACK are not, and it needs --gcc-install-dir for a libstdc++ with <print>.)
#
# Sets DDM_CC, DDM_CXX and DDM_FLAGS (flags that every C/C++ compile and link needs, also for the bundled dependencies).
toolchain=${1:-gcc14}
module purge
case $toolchain in
  gcc14)
    module load compiler/gnu/14.2 mpi/openmpi/5.0-gnu-14.2 numlib/mkl/2025.0
    DDM_CC=$(command -v gcc)
    DDM_CXX=$(command -v g++)
    DDM_FLAGS=""
    ;;
  *)
    echo "env_kit.sh: unknown toolchain '$toolchain' (gcc14)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
export DDM_CC DDM_CXX DDM_FLAGS
export FC=gfortran  # for the bundled dependencies
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
