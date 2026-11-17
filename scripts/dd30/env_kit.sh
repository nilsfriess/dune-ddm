# Environment of the DD30 builds and runs on bwUniCluster 3.0 (KIT), sourced by build_kit.sh and job_kit.sh with the
# toolchain as argument:
#
#   llvm22  clang 22.1 (the default)
#   gcc14   GCC 14.2 (as on Helix, which has GCC 14.1), for comparison
#
# Both use OpenMPI 5.0 built with GCC 14.2 (only its C interface is used, so the compiler of the MPI library does not
# matter; the OpenMPI of the LLVM 19.1 module would swap LLVM 22 for 19) and MKL 2025.0 for PARDISO. Clang 22 would take
# the C++ standard library of the system's gcc-toolset-13, which has no <print> (include/ddm/logger.hh): it is pointed to
# the one of GCC 14.2, which the OpenMPI module loads, so both toolchains use the same standard library.
#
# Sets DDM_CC, DDM_CXX and DDM_FLAGS (flags that every C/C++ compile and link needs, also for the bundled dependencies).
toolchain=${1:-llvm22}
module purge
case $toolchain in
  llvm22)
    module load compiler/llvm/22.1 mpi/openmpi/5.0-gnu-14.2 numlib/mkl/2025.0
    DDM_CC=$LLVM_BIN_DIR/clang
    DDM_CXX=$LLVM_BIN_DIR/clang++
    DDM_FLAGS="--gcc-install-dir=$(dirname "$(g++ -print-libgcc-file-name)")"
    ;;
  gcc14)
    module load compiler/gnu/14.2 mpi/openmpi/5.0-gnu-14.2 numlib/mkl/2025.0
    DDM_CC=$(command -v gcc)
    DDM_CXX=$(command -v g++)
    DDM_FLAGS=""
    ;;
  *)
    echo "env_kit.sh: unknown toolchain '$toolchain' (llvm22 | gcc14)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
export DDM_CC DDM_CXX DDM_FLAGS
export FC=gfortran  # for the bundled dependencies, in both toolchains
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
