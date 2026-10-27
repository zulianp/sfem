# SFEM

SFEM is a C++/CUDA finite element library for large solid mechanics simulations on CPUs and
GPUs. Its operators are matrix-free or partially assembled, on unstructured and semi-structured
(refined) meshes, and most are generated from symbolic descriptions of the material.

- **Materials**: linear elasticity, NeoHookean-Ogden, Mooney-Rivlin with Prony-series
  viscoelasticity, linear and Mooney-Rivlin Kelvin-Voigt, Laplacian and flow operators.
- **Elements**: TET4, TET10, HEX8, semi-structured HEX8 and TET4, shells and beams.
- **Solvers**: Newton with line search, CG and BiCGStab, Chebyshev, geometric and algebraic
  multigrid, MPRGP for bound constraints.
- **Contact**: rigid obstacles given as signed distance fields, solved per Newton step or by a
  shifted-penalty multigrid.
- **Time integration**: quasi-static stepping, Newmark, BDF2, backward Euler.
- **Execution**: OpenMP, CUDA (with streams and graphs), MPI, NCCL within a node.
- **Code generation**: `python/codegen/framework` writes the operators' CPU and GPU kernels.

## Quick start

Requirements: a C++17 compiler, CMake 3.20+, MPI; optionally OpenMP, CUDA and Python 3.

```sh
git clone --recursive https://github.com/zulianp/sfem.git
cd sfem
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSFEM_ENABLE_OPENMP=ON
cmake --build build -j 8
(cd build && ctest -j 8)
```

On macOS with Apple Clang add `-DOPENMP_DIR=$(brew --prefix libomp)`; for a GPU build add
`-DSFEM_ENABLE_CUDA=ON`. [Building](docs/building.md) lists the options, the dependencies CMake
fetches, the Python bindings and the Alps (CSCS) setup.

## Run a simulation

The Playground runs a simulation described by one YAML file:

```sh
build/playground workflows/kelvin_voigt/case.yaml --set output.path=$PWD/kv
```

[The Playground documentation](docs/playground/overview.md) explains cases, every key, outputs,
and parallel and GPU runs; `workflows/` holds example cases.

## Documentation

- [Building](docs/building.md) and [testing](docs/testing.md)
- [The Playground](docs/playground/overview.md): [case reference](docs/playground/case-reference.md),
  [examples](docs/playground/examples.md), [outputs](docs/playground/outputs.md),
  [parallel and GPU runs](docs/playground/parallel-and-gpu.md)
- [All documents](docs/README.md)
- [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zulianp/sfem) (generated)

## Repository layout

| Folder | Content |
|---|---|
| `src/` | The library: `base`, `algebra`, `mesh`, `operators`, `frontend` (functions, operators, nonlinear solvers, contact, the Playground), `ssmg` and `amg` (multigrid), `resampling`, `solver`, `matrix`, `pizzastack`. Unit tests are in each module's `tests/`. |
| `drivers/` | Executables, including `drivers/mech/playground.exe.cpp`. |
| `python/` | The `pysfem` bindings, the `sfem` Python tools, and the code generator. |
| `workflows/` | Example cases and pipelines. |
| `verification_and_validation/` | Cases checked against analytic solutions. |
| `docs/` | Documentation. |
| `data/` | Small meshes used by examples. |
| `scripts/` | Build-matrix and maintenance scripts. |
| `spikes/` | Separate research projects built on SFEM. |
| `external/` | Submodules: smesh, sresample, sccd, ssdf, x86-simd-sort. |
| `dev/` | Development notes, plans, papers and lab experiments. |

## Citing SFEM

```bibtex
@misc{sfemgit,
	author = {Zulian, Patrick and Riva, Simone and Yang, Haoyu},
	title = {{SFEM}: Simple {FEM}},
	url = {https://github.com/zulianp/sfem},
	howpublished = {https://github.com/zulianp/sfem},
	year = {2024}
}
```

## License

BSD 3-Clause; see [LICENSE](LICENSE).
