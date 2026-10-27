# Building SFEM

## Requirements

- A C and C++17 compiler, CMake 3.20 or newer, and MPI (on by default; `SFEM_ENABLE_MPI=OFF`
  builds without it).
- Git submodules: `external/smesh` (mesh library, required), `external/sresample` (resampling
  and SDF obstacles, on by default), `external/sccd` and `external/ssdf` (collisions and
  distances, optional).
- Fetched by CMake at configure time: RapidYAML (with `SFEM_ENABLE_RYAML`, the default) and, in
  MPI builds, matrix.io. Without MPI, matrix.io is looked for in `../matrix.io` or `MATRIXIO_DIR`.
- Optional: OpenMP, CUDA, BLAS/LAPACK, METIS, NCCL, Python 3 with nanobind.

## Configure, build, test

```sh
git clone --recursive https://github.com/zulianp/sfem.git
cd sfem
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSFEM_ENABLE_OPENMP=ON
cmake --build build -j 8
cd build && ctest -j 8
```

For an existing clone, `git submodule update --init --recursive` fetches the submodules.

**macOS with Apple Clang**: OpenMP comes from Homebrew's `libomp`; point CMake at it with
`-DOPENMP_DIR=$(brew --prefix libomp)`. Homebrew's LLVM (`-DCMAKE_C_COMPILER=$(brew --prefix
llvm)/bin/clang -DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++`) works without it.

## Options

| Option | Default | Meaning |
|---|---|---|
| `SFEM_ENABLE_OPENMP` | OFF | Threaded host kernels. |
| `SFEM_ENABLE_MPI` | ON | Distributed meshes and solvers. |
| `SFEM_ENABLE_CUDA` | OFF | GPU kernels and solvers. |
| `SFEM_ENABLE_NCCL` | ON | NCCL for device communication within a node, in builds with MPI and CUDA; off when not found. |
| `SFEM_ENABLE_RYAML` | ON | YAML input (the Playground, operator and condition files). |
| `SFEM_ENABLE_SRESAMPLE` | ON | The sresample library: SDF obstacles and grid/mesh transfer. |
| `SFEM_ENABLE_AMG` | ON | Algebraic multigrid. |
| `SFEM_ENABLE_PYTHON` | OFF | The `pysfem` Python bindings (needs nanobind). |
| `SFEM_ENABLE_SCCD`, `SFEM_ENABLE_SSDF` | OFF | Continuous collision detection and signed distances (self-contact). |
| `SFEM_ENABLE_METIS` | OFF | METIS partitioning. |
| `SFEM_ENABLE_BLAS`, `SFEM_ENABLE_LAPACK` | OFF | BLAS/LAPACK kernels. |
| `SFEM_ENABLE_RESAMPLING` | OFF | The grid-to-mesh field transfer drivers (being moved to sresample). |
| `SFEM_ENABLE_TESTING` | ON | The unit tests (off when SFEM is a subproject). |
| `SFEM_ENABLE_MINIMAL` | OFF | The library only, no drivers or tests. |
| `SFEM_ENABLE_DEV_MODE` | OFF | Warnings are errors in SFEM's own C, C++ and CUDA sources. |
| `SFEM_ENABLE_TRACE` | ON | Timings of the main phases, written to `smesh.trace.csv`. |
| `SFEM_ENABLE_EXPLICIT_VECTORIZATION` | ON | Hand-vectorised kernels. |
| `SFEM_ENABLE_AVX2`, `SFEM_ENABLE_AVX512` | OFF | x86 intrinsics. |
| `BUILD_SHARED_LIBS` | OFF | |

Number types: `SFEM_REAL_TYPE` (float64), `SFEM_GEOM_TYPE` (float32), `SFEM_IDX_TYPE` (int32),
and `SFEM_SCALAR_TYPE`, `SFEM_COUNT_TYPE`, `SFEM_ELEMENT_IDX_TYPE`, `SFEM_LOCAL_IDX_TYPE`
(`cmake/SFEMNumbers.cmake`).

CUDA: `CMAKE_CUDA_ARCHITECTURES` (default 90), `SFEM_CUDA_MEMORY` (`host`, `managed` or
`unified`).

[Build matrix](build-matrix.md) lists the option combinations tested on macOS and Alps.

## Python

```sh
python3 -m venv venv && source venv/bin/activate
pip install -r python/requirements.txt          # plus requirements_cu12.txt for CuPy on CUDA 12
cmake -S . -B build -DSFEM_ENABLE_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$PWD/install
cmake --build build -j 8 --target install
export PYTHONPATH=$PWD/install/lib:$PWD/install/scripts:$PYTHONPATH
python3 -c "import pysfem"
```

## Alps (CSCS)

Build on a login node inside the `prgenv-gnu/24.11:v2` user environment, and run on the compute
nodes (see [Parallel and GPU runs](playground/parallel-and-gpu.md)):

```sh
uenv image pull prgenv-gnu/24.11:v2
uenv run --view=default prgenv-gnu/24.11:v2 -- bash -c '
  cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
    -DSFEM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DSFEM_ENABLE_OPENMP=ON &&
  cmake --build build-cuda -j 64'
```

`scripts/testing/test_CPU_configs.sh` and `test_CUDA_configs.sh` build (and with `RUN_TESTS=1`
test) many option combinations; on Alps run them with `SFEM_USE_UENV=1`.
