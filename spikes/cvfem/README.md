# CVFEM

A control-volume finite element Navier–Stokes spike: HEX8 and TET4 upwind kernels, a steady
solver, a semi-structured geometric multigrid preconditioner, a CUDA port, and the benchmark
harness the numbers in `docs/` were measured with.

This is an index. Everything it points at was written elsewhere; nothing here is a summary
of results.

## Layout

| | |
|---|---|
| `src/core/` | portability shims, element colouring, FGMRES |
| `src/hex8/`, `src/tet4/` | element kernels and the memory layouts they run under |
| `src/generated/` | SymPy-emitted kernels — **machine-written, do not edit** (see `python/`) |
| `src/ss/` | semi-structured operator, Galerkin coarsening, transfers, Vanka smoother |
| `src/op/` | the `sfem::Op` face of the kernels |
| `src/cases/` | channel and manufactured-solution problem definitions |
| `drivers/` | one executable per file; the benchmarks and solvers |
| `tests/` | one ctest per file |
| `python/` | the SymPy kernel synthesis, plus the analysis and plotting scripts |
| `scripts/` | build, run and post-processing helpers |
| `jobs/` | Slurm batch jobs for Alps |
| `docs/` | the lab journal (`README_alps.md`) and the written-up results |
| `cuda/` | CUDA kernels, built only with `-DCVFEM_ENABLE_CUDA=ON` |
| `subpar/` | variants that were measured, lost, and were quarantined rather than deleted |
| `patches/`, `eval/`, `literature/` | smesh patches, evaluation data, references |

Every cvfem header is included by bare filename; each directory under `src/` is placed flat
on the include path, the way SFEM treats its own modules.

## Building

Against an installed SFEM:

```sh
cmake -S . -B build -DSFEM_DIR=<prefix>/lib/cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

On Alps, `scripts/alps_env.sh` wraps the uenv and the dependency paths:

```sh
source scripts/alps_env.sh
cvfem_configure && cvfem_build --target cvfem_hex8_ns_ssgmg
CVFEM_CPUS=72 cvfem_run env OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./build/cvfem_hex8_ns_ssgmg
```

Options, all `CVFEM_ENABLE_*`: `TRACE` (on), `SUBPAR`, `BLAS`, `CUDA`. They and the
performance flag set live in `cmake/CVFEMCMakeFunctions.cmake`; the `-fno-finite-math-only`
comment there is load-bearing and explains why `cvfem_guard_selftest` exists.

## Checking for performance regressions

The gate is the **packed matrix-free throughput** — the residual and Jacobian action of
`cvfem_hex8_ns_upwind_bench` at `--layout packed`, which reach >2000 MDOF/s on a Grace
socket. Those kernels are the fast path, they are what the numbers in `docs/` quote, and
they are what the compiler flags were tuned on. A slower driver is not a substitute:
`cvfem_ns_apply_bench` saturates near 267 MDOF/s and agreement there says close to nothing
about a kernel running nine times faster.

Run it for any change that could touch code generation, compiler flags, memory layout or
the build:

```sh
# The real gate: new binary against one built from the commit you are comparing to,
# both measured in one allocation with the order alternating. Bands at 5%.
sbatch --export=ALL,REF_BIN=/path/to/old/cvfem_hex8_ns_upwind_bench jobs/perf_regression.sbatch

# Coarse check against perf/baseline_grace.csv. Bands at 12%, because the same binary
# varies 5-11% between Grace nodes. Use it as a smoke test, not to clear a change.
sbatch jobs/perf_regression.sbatch
```

The gate carries the bare element kernel *and* three configurations with the Rhie-Chow
term and the boundary closure on -- the operator the solver actually runs, which is 53%
of the bare kernel's throughput, or 28% if the nodal pressure gradient is not cached.
`docs/CVFEM_Kernels.md` has the measurement.

`scripts/perf_regression.sh --help` explains the two modes, why three configurations are
recorded but not gated in baseline mode, and which two are too bimodal to measure at all.
Re-record the baseline (`RECORD=1`) only deliberately, and say in the commit message what
changed and on what evidence.

## Running something long

Use `scripts/cvrun.sh`, not a hand-rolled `driver | grep` pipeline. It keeps a complete
unfiltered log, line-buffers the filtered view so a running job is distinguishable from a
hung one, and timestamps start and end.

```sh
scripts/cvrun.sh poiseuille ./build/cvfem_hex8_ns_ssgmg OMP_NUM_THREADS=72 -- ...
```

## Regenerating the kernels

The headers in `src/generated/` are emitted by the scripts in `python/`; see
`python/README.md` for which script writes which header and how to check a regeneration
against what is committed.
