# DUNE-DDM Agent Guidelines

## General

This is a DUNE module implementing domain decomposition methods for parallel PDE solvers. It depends on dune-common, dune-grid, dune-istl, dune-pdelab, and optionally dune-uggrid.

## Building and Running

Examples are built out-of-source. To compile and run:

```bash
cd /home/friessn/Projects/dune/dune2.10_ddm/debug-build-clang/dune-ddm/examples
make <example_name>
./example_name                           # sequential
mpirun -np 4 ./example_name              # parallel with 4 ranks
mpirun --oversubscribe -np 16 ./example  # if more ranks than cores
```

Configuration is via `.ini` files in the examples directory, not command line arguments.

## Debugging

**IMPORTANT**: When encountering segfaults or crashes, DO NOT guess at the cause. Immediately use debugging tools:

```bash
# GDB for backtrace
gdb -batch -ex run -ex bt ./example_name

# Valgrind for memory issues
valgrind ./example_name

# For MPI programs, run single rank first or use:
mpirun -np 1 xterm -e gdb ./example_name
```

The backtrace will show exactly where the crash occurs and save significant debugging time.

## Code Guidelines

*Mathematical correctness*: This is research code for a PhD in numerical analysis. One of the top goals of this code is therefore to be mathematically correct.

*Command line parameters*: Configuration is primarily via `.ini` files. Command line arguments can override ini settings using **single-dash** syntax with DUNE's ParameterTreeParser:

```bash
# Correct - single dash:
./poisson -overlap=2 -coarsespace.type=geneo -solver.verbose=1

# WRONG - double dashes don't work with ParameterTreeParser:
./poisson --overlap=2   # This will NOT set the overlap parameter!

# Nested parameters use dots:
./poisson -coarsespace.type=pou -coarsespace.threshold=0.1

# Exception: logging uses double-dash:
./poisson --log-level=debug
```

*Parallel index sets*: The parallel index set size (from `make_communication`) will typically be larger than the matrix size because it corresponds to overlapping subdomains that are algebraically extended from the initial non-overlapping decomposition. This size difference is expected and not an error.

*DG constraints*: For parallel DG, use `NoConstraints` instead of `P0ParallelGhostConstraints`. The ghost constraints can interfere with correct skeleton assembly at processor boundaries.

*NonOverlappingEntitySet limitation*: PDELab's `NonOverlappingBorderDOFExchanger` only handles codim > 0 entities (vertices/edges for CG), NOT codim-0 (elements for DG). Using `NonOverlappingEntitySet` with DG causes a segfault in `BorderIndexIdCache`. Use `AllEntitySet` instead for DG.

*A*1 = 0 test*: For pure Neumann problems with DG, the stiffness matrix should satisfy $A \cdot \mathbf{1} = 0$ (row sums are zero). This is a useful sanity check for correct parallel assembly. If this fails in parallel but works sequentially, it indicates missing flux contributions at processor boundaries.butions at processor boundaries.