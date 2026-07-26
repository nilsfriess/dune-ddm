# dune-ddm — Domain decomposition methods in Dune

## Building

`dune-ddm` is set up as a **superbuild**: the required DUNE modules are bundled
as git submodules under `extern/` and built in-tree, so no `dunecontrol` is
needed. The DUNE core modules and staging modules track `releases/2.11`;
`dune-pdelab` tracks `releases/2.10` (no 2.11 release exists yet).

```sh
git clone --recursive <repo-url> dune-ddm
cd dune-ddm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

If you already cloned without `--recursive`, fetch the submodules with:

```sh
git submodule update --init --recursive
```

### Build dependencies

You need a C++20 compiler, CMake ≥ 3.24, MPI, BLAS/LAPACK, OpenMP, Eigen3 and
SuiteSparse (UMFPACK, CHOLMOD). STRUMPACK is used if found. The third-party
libraries in `extern/` (LuaJIT, Spectra, TRL) are also git submodules and are
built automatically.

### Useful CMake options

| Option | Default | Description |
| --- | --- | --- |
| `DUNE_DDM_SUPERBUILD` | `ON` | Build the DUNE modules in `extern/` in-tree. Set to `OFF` to build against externally provided DUNE modules (e.g. via `dunecontrol` or a system installation). |
| `DUNE_ENABLE_PYTHONBINDINGS` | `OFF` | DUNE Python bindings (off by default in the superbuild to keep `cmake ..` friction-free). |
| `DUNE_DDM_BUILD_EXAMPLES` | `ON` | Build the examples in `examples/` (requires dune-pdelab). |

The examples are `EXCLUDE_FROM_ALL`; build a specific one with e.g.
`make pdelab_example`.
