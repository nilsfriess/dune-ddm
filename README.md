# dune-ddm — Domain decomposition methods in Dune
This library implements domain decomposition preconditioners using spectral coarse spaces for robust preconditioning of linear systems that arise from finite element discretisations of multiscale PDEs.

## Building

The library `dune-ddm` is based on [DUNE](https://www.dune-project.org/). It includes all necessary DUNE dependencies and many other dependencies as git submodules. To download and build all dependencies it is enough to run:

```sh
git clone --recursive <repo-url> dune-ddm
cd dune-ddm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
During CMake configuration, this will also build a few dependencies (METIS, ParMETIS, SuiteSparse, STRUMPACK, Eigen, and Spectra).

If you already cloned without `--recursive`, fetch the submodules with:

```sh
git submodule update --init --recursive
```
