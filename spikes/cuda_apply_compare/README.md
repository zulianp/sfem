# The generated kernels, host against device

The framework names CUDA as a target and has named it for a long time. Nothing had ever
compiled its output with `nvcc`, so what "CUDA works" meant was unmeasured. This spike measures
it: generated kernels, generated for both targets, compiled for both, run on one Grace-Hopper
node, and compared.

## What it is

`apply_driver.cpp` is **one source compiled twice** — by `g++` against an OpenMP generation and
by `nvcc` against a CUDA generation. Both link the same `extern "C"` entry points under the same
names and run the same mesh. That is deliberate: there is no second implementation of any
operator in this directory, so what the two runs differ by is the generated kernel and nothing
else. The device arm is selected by `__CUDACC__` and is only memory plumbing — allocate, copy,
launch, copy back. `build.sh` builds both; `gh200.sbatch` runs them.

Two things are measured, and they want different meshes.

**Agreement** runs every kernel whose device lowering is worth doubting, on a small mesh whose
connectivity is *shared* — several elements landing on the same node — so the scatter is
contended and `atomicAdd` is exercised rather than bypassed. Each output vector is written out
by name and `compare.py` differences the two files entry by entry, relative to the largest entry
of each vector.

The kernels are the neo-Hookean Ogden matrix-free apply on a tensor-product hexahedron, which is
the shape this framework cares most about; the two P1-simplex metric bodies (laplace TRI3 and
TET4, gradient and apply); the constant-P1 AoS unit body (linear elasticity TET4 gradient); and
the two mesh-order tensor-product bodies whose stream arrays were miscounted (laplace QUAD4 and
HEX8 gradient). The last five are there because they were shipping a serial mesh loop inside
`__global__`, or not compiling at all, until that was measured.

**Throughput** uses a structured lattice, because a timing on scrambled connectivity is not a
result: hexahedra for the tensor-product apply, and the Kuhn subdivision of the same lattice —
six tetrahedra per cell, so the node numbering keeps its locality — for the simplex gradient.

## Reproducing it

Generate both trees where sympy lives, ship them, build and run where the GPU is — the same
split `spikes/inexact_apply_compare` uses, because no interpreter on Alps can import the plan
layer:

```python
from sfem import gen
from codegen.framework.materials import laplace, linear_elasticity, neohookean_ogden
for name, module in (("laplace", laplace), ("linear_elasticity", linear_elasticity),
                     ("neohookean_ogden", neohookean_ogden)):
    for target, sub in (("openmp", "omp"), ("cuda", "cuda")):
        gen.generate(module.material, f"{sub}/{name}",
                     elements=("TRI3", "QUAD4", "TET4", "HEX8"), target=target)
```

```bash
uenv run --view=default prgenv-gnu/24.11:v2 -- ./build.sh
sbatch gh200.sbatch
```

A standalone generation emits no `op/` wrapper, which is what makes this build self-contained —
and is the same `supports_op_wrapper=False` that means nothing in the SFEM frontend can reach
these kernels yet. `shim/` supplies the scalar aliases and the two `smesh` enums that
`op/*_c_abi.hpp` names in declarations and that a standalone generation does not carry.

## Measured

One Alps GH200 node (`nid006546`): NVIDIA GH200 120GB, compute capability 9.0, Neoverse-V2 host,
72 OpenMP threads with `OMP_PROC_BIND=true` and `OMP_PLACES=cores`, nvcc 12.6.20 at `-arch=sm_90`,
best of 3, `real_t` = double. `gh200.sbatch` writes its log to `$SCRATCH` and `*.log` is
ignored, so the tables below are the record — the sibling spike keeps its results the same way.

Neo-Hookean Ogden apply, PROTEUS_HEX8:

| ndof | elements | host MDOF/s | GH200 MDOF/s | device / host |
|---|---|---|---|---|
| 107,811 | 32,768 | 220.8 | 1899.2 | 8.6x |
| 823,875 | 262,144 | 216.9 | 2258.5 | 10.4x |
| 2,738,019 | 884,736 | 216.5 | 2305.6 | 10.6x |
| 6,440,067 | 2,097,152 | 214.5 | 2337.5 | 10.9x |
| 12,519,843 | 4,096,000 | 213.7 | 2334.3 | **10.9x** |

Laplace gradient, TET4, constant-metric closed form:

| ndof | elements | host MDOF/s | GH200 MDOF/s | device / host |
|---|---|---|---|---|
| 35,937 | 196,608 | 442.3 | 2344.5 | 5.3x |
| 274,625 | 1,572,864 | 516.4 | 4460.7 | 8.6x |
| 912,673 | 5,308,416 | 524.8 | 5218.1 | 9.9x |
| 2,146,689 | 12,582,912 | 520.4 | 5413.7 | 10.4x |
| 4,173,281 | 24,576,000 | 522.1 | 5481.2 | **10.5x** |

Both sides are saturated at the top of each sweep. The host is flat across the whole range in
both cases (214-221 and 516-525), and the device gains 0.1% on the hexahedral apply and 1.2% on
the tetrahedral gradient over the last doubling, so each ratio is between two saturated numbers
rather than an artefact of an under-filled problem.

**Every output vector agrees to at most 6e-16** relative to its largest entry, across all twelve.
The comparison is not vacuous in either of the two ways it could be: every vector is fully
non-zero with every entry distinct, so nothing agrees by being empty, and moving a single entry
by 1e-9 relative makes `compare.py` report a failure at 1.2e-10.

This replaced a comparison that printed each run's own L2 and L-infinity norms. Two answers can
share both norms and still differ entrywise, so that version could only see gross disagreement —
and it was the only check the PROTEUS_HEX8 apply had.

## What this establishes, and what it does not

**Establishes:** the framework generates CUDA that compiles with nvcc 12.6, runs on an H100, and
computes the same answer as its own OpenMP kernel — entry by entry — for both the tensor-product
and the constant-P1 simplex lowerings, at throughputs measured on saturated problems.

**Does not establish** anything about the other two kernel families. `residual_codegen` and
`inexact_apply_codegen` emit `_operator.cpp` with host `extern "C"` functions around an OpenMP
mesh loop, and `_validate_cuda_source_contract` wants a file ending `_operator.cu` containing
`__global__ void`. Nor does it establish that anything in the SFEM library can *call* these
kernels: `CMakeLists.txt` globs only `*.cpp`/`*.hpp`, so no `.cu` reaches the build, and the CUDA
backend publishes no `sfem::Op`.

**A gap this measurement walked past:** the generated launcher returns `SFEM_SUCCESS` immediately
after an asynchronous `<<<>>>` launch, so it cannot report a launch failure. This driver calls
`cudaDeviceSynchronize` itself and checks it. Correct for a stream-based API, but it is currently
inherited rather than decided.
