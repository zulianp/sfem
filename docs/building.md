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
| `BUILD_SHARED_LIBS` | ON | libsfem and the libraries built with it (smesh, sresample, SCCD, SSDF, RapidYAML) are shared; executables load them at run time. OFF links everything statically into each executable. |

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

## Using SFEM from another project

Install SFEM first (`cmake --install build --prefix <prefix>`). The installed executables,
libraries and the `pysfem` module find the shared libraries in `<prefix>/lib` on their own;
`LD_LIBRARY_PATH` is not needed.

**CMake**: point `SFEM_DIR` at `<prefix>/lib/cmake` and link `SFEM::sfem`, which brings its
include directories, flags and dependencies:

```cmake
find_package(SFEM REQUIRED)            # cmake -DSFEM_DIR=<prefix>/lib/cmake ...
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE SFEM::sfem)
```

**Makefiles**: `cmake --build build --target install_all` also writes
`<prefix>/config/sfem-config.makefile`, with SFEM's compilers (`SFEM_C_COMPILER`,
`SFEM_CXX_COMPILER`), `SFEM_INCLUDES`, `SFEM_CXXFLAGS` and `SFEM_LIBRARIES` (the libraries with
an rpath to their directory):

```make
include <prefix>/config/sfem-config.makefile

my_app: main.cpp
	$(SFEM_CXX_COMPILER) -std=c++17 $(SFEM_CXXFLAGS) $(SFEM_INCLUDES) $< -o $@ $(SFEM_LIBRARIES)
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

## Aurora (ALCF): Intel GPUs with chipStar

The device path (`SFEM_ENABLE_CUDA=ON`) is compiled as HIP (`SFEM_ENABLE_HIP=ON`) by chipStar, which
targets SPIR-V and runs on Level Zero; `CMAKE_HIP_PLATFORM=spirv` selects it, for SFEM and the
libraries built with it. It needs CMake 4.3 or later (Aurora's module is 3.31: use a Kitware
release). Build on a login node; the host code is for the compute nodes' Sapphire Rapids:

```sh
module use /soft/modulefiles
module load chipStar/llvm22/20260926-22/release gcc/14.3.0
cmake -S . -B build-pvc -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_HIP_PLATFORM=spirv -DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang++ \
  -DSFEM_ENABLE_CUDA=ON -DSFEM_ENABLE_HIP=ON -DSFEM_ENABLE_OPENMP=ON \
  -DSCCD_ENABLE_NATIVE_ARCH=OFF -DCMAKE_CXX_FLAGS=-march=sapphirerapids -DCMAKE_C_FLAGS=-march=sapphirerapids
cmake --build build-pvc -j 24
```

Run with `CHIP_BE=level0`. One MPI rank per GPU tile (12 a node) is
`mpiexec -n 12 --ppn 12 gpu_tile_compact.sh <app>`, in the default `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`;
with `FLAT` instead, one process sees every tile as a device. MPI takes device buffers directly. There is no NCCL. Graphs are not captured on the per-thread default stream
(chipStar replays a captured graph and gains nothing from it): the solvers run their iterations from
the host there.

The first launch of a large module compiles it for the GPU, which takes minutes; a module cache,
`export CHIP_MODULE_CACHE_DIR=<a directory on /lus/flare>`, makes later runs start at once.

Intel's driver on Aurora (compute-runtime 25.18) has a bug in kernels that spill registers to
private memory: after an eviction under memory pressure it leaves their private memory unmapped, and
the next launch dies with `Segmentation fault from GPU ... (NotPresent) ... (Write)`. Several
processes on one tile (ctest, for instance) bring it out in the high-order (hex64) operators; one
process per tile does not. The fix is in compute-runtime 26.35.39758.10
([intel/compute-runtime#973](https://github.com/intel/compute-runtime/issues/973)). The driver's own
workaround, `NEOReadDebugKeys=1 MakeEachAllocationResident=2`, removes the faults but makes most of
SFEM's device tests run past their 15-minute limit, so it is no remedy.
