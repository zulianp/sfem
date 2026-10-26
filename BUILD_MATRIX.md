# Build matrix

Every configuration below is built with `SFEM_ENABLE_DEV_MODE=ON` (warnings are errors in C, C++
and CUDA, in SFEM's own sources), Release, testing ON, and passes its whole ctest.
Recorded 2026-10-25 on branch tight-api (Stage H of `TIGHT_API.md`).

## Platforms

| Platform | Compilers | MPI | Script |
|---|---|---|---|
| macOS 15.7.7, Apple silicon | Homebrew clang 20.1.8 | Open MPI 5.0.7 | `MODE=pairwise RUN_TESTS=1 scripts/testing/test_CPU_configs.sh` |
| Alps GH200 (Grace), uenv `prgenv-gnu/24.11:v2` | GCC 13.3.0 | Cray MPICH 8.1.30 | `test_CPU_configs.sh`, pairwise |
| Alps GH200 (H100), same uenv | GCC 13.3.0, nvcc 12.6.20, arch 90 | Cray MPICH 8.1.30 | `test_CUDA_configs.sh --pairwise`, memory model host |

The pairwise sweep covers every pair of options in all four ON/OFF combinations. METIS is
not in the Alps uenv, so the Alps CPU sweep has one option fewer than macOS.

## macOS (CPU)

| # | Options ON | Build | ctest |
|---|---|---|---|
| 0000 | none | ok | 59/59 |
| 0001 | amg mpi metis blas ryaml sccd ssdf | ok | 89/89 |
| 0002 | amg openmp mpi blas lapack ryaml ssdf resampling | ok | 80/80 |
| 0003 | amg openmp metis blas lapack sccd ssdf explicit_vectorization | ok | 80/80 |
| 0004 | openmp mpi metis blas resampling explicit_vectorization | ok | 63/63 |
| 0005 | amg openmp mpi metis lapack ryaml sccd resampling explicit_vectorization | ok | 77/77 |
| 0006 | lapack ryaml sccd ssdf resampling explicit_vectorization | ok | 82/82 |
| 0007 | all | ok | 89/89 |

## Alps (CPU)

| # | Options ON | Build | ctest |
|---|---|---|---|
| 0000 | none | ok | 59/59 |
| 0001 | amg openmp mpi lapack ryaml ssdf | ok | 80/80 |
| 0002 | amg openmp blas lapack sccd resampling | ok | 68/68 |
| 0003 | amg mpi blas ryaml sccd explicit_vectorization | ok | 77/77 |
| 0004 | openmp mpi blas ssdf resampling explicit_vectorization | ok | 73/73 |
| 0005 | lapack ryaml sccd ssdf resampling explicit_vectorization | ok | 82/82 |
| 0006 | all | ok | 89/89 |

## Alps (CUDA)

| # | Options ON (with cuda) | Build | ctest |
|---|---|---|---|
| 0000 | none | ok | 65/65 |
| 0001 | amg mpi nccl blas ryaml sccd ssdf | ok | 100/100 |
| 0002 | amg openmp mpi blas lapack ryaml ssdf resampling | ok | 88/88 |
| 0003 | amg openmp nccl blas lapack sccd ssdf explicit_vectorization | ok | 89/89 |
| 0004 | openmp mpi nccl blas resampling explicit_vectorization | ok | 70/70 |
| 0005 | amg openmp mpi nccl lapack ryaml sccd resampling explicit_vectorization | ok | 88/88 |
| 0006 | lapack ryaml sccd ssdf resampling explicit_vectorization | ok | 92/92 |
| 0007 | all | ok | 100/100 |

The managed and unified memory models are not in this sweep (`CUDA_MEMORY_MODELS=host`).

## Options

- `SFEM_ENABLE_DEV_MODE` applies per language through `add_compile_options` after the
  dependencies are added: SFEM's targets only. SCCD's target is `SYSTEM`, matrix.io's sources are
  compiled with `-w`. Unknown pragmas warn without failing; designated initializers and
  `-Wmissing-field-initializers` are allowed (used throughout by design). Under DEV_MODE ptxas
  reports spills with `-v` but not as `-warn-spills` warnings.
- Removed, as they had no effect: `SFEM_ENABLE_FP16_JACOBIANS`, `SFEM_ENABLE_CODEGEN`,
  `SFEM_ENABLE_SRESAMPLE`, the forwarding of the never-defined `SFEM_ENABLE_CUBLAS` and
  `SFEM_ENABLE_CUSPARSE`, and the second (dependent) `SFEM_ENABLE_LAPACK`.
- `SFEM_CPU_ARCH` and `SFEM_CUDA_MEMORY` are string cache variables; an unknown memory model
  is an error (a cache holding the old boolean `OFF` maps to `host`).
- Without RYAML the playground, and the tests whose cases are YAML, are not built. Without MPI the
  resampling drivers written against communicators are not built. ssdf tells cuBQL not to enable
  CUDA in a build without CUDA.

## Defects found by the sweep

- MPI-off builds link the real MPI library but never call `MPI_Init`; Cray MPICH aborts on any MPI
  call then, Open MPI does not. Every `MPI_Wtime` call is now `smesh::time_seconds()`.
- `tet10_apply_inv_lumped_mass` (and the TRI6 one) scattered through uninitialized indices.
- `hex8_mooney_rivlin_active_strain_hessian_partial_assembly` overwrote the tangent at each
  quadrature point instead of summing it (`_add`); the partial-assembly test now checks the apply
  against a central difference of the gradient.
- `sfem_MPRGPSolveTest` (GCC, no OpenMP): the pass-by-pass and one-region routes executed different
  inlined copies of the same passes, which GCC fused differently (`-ffp-contract=fast`); they
  differed by one ulp at iteration 267. The shared passes are `SFEM_NOINLINE`. The OpenMP GCC
  checksum is now `08a21c14dd38fead` (was `8986959ae54ca2f7`); Grace, 72 threads, whole test,
  5 runs: 2.17-2.66 s before, 2.19-2.41 s after.
- CUDA resampling, unified memory: the result was never copied back to the caller. Fixed, not
  covered by this sweep (host memory model only).
- Generated ops without RYAML: the material parameter names are needed by `derefine` either way
  (generator fix, regenerated).
- SCCD (branch tight-api): uninitialized midpoint lanes in `sccd_narrowphase_tight.hpp`, and its
  ctypes binding test skipped for a static library.
- `sfem_CudaGeneratedMooneyRivlinPronyTest`, TET10: the test's nodal noise inverted an element at
  step 6 (host value NaN), failing since the case was added; TET10 takes half the TET4 amplitude.
