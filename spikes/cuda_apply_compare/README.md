# The generated CUDA apply, host against device

The framework names CUDA as a target and has named it for a long time. Nothing had ever
compiled its output with `nvcc`, so what "CUDA works" meant was unmeasured. This spike measures
it: one generated kernel, generated for both targets, compiled for both, run on one
Grace-Hopper node, and compared.

## What it is

`apply_driver.cpp` is **one source compiled twice** — by `g++` against an OpenMP generation and
by `nvcc` against a CUDA generation. Both link the same `extern "C"` entry point under the same
name and run the same mesh. That is deliberate: there is no second implementation of the
operator in this directory, so what the two runs differ by is the generated kernel and nothing
else. The device arm is selected by `__CUDACC__` and is only memory plumbing — allocate, copy,
launch, copy back.

The kernel is `neohookean_ogden_proteus_hex8_apply_i_msoa`: the matrix-free Hessian action of a
neo-Hookean Ogden material on a tensor-product hexahedron, which is the shape this framework
cares most about.

Answers are compared by L2 and L-infinity norms rather than byte-for-byte. The device sums its
contributions through atomics in a different order, so the two agree to round-off and not to the
bit, and printing both norms is what lets the comparison state its tolerance instead of assuming
one.

## Reproducing it

Generate both trees where sympy lives, ship them, build and run where the GPU is — the same
split `spikes/inexact_apply_compare` uses, because no interpreter on Alps can import the plan
layer:

```python
from sfem import gen
from codegen.framework.materials.neohookean_ogden import material
gen.generate(material, "omp",  elements=("HEX8",), target="openmp")
gen.generate(material, "cuda", elements=("HEX8",), target="cuda")
```

```bash
uenv run --view=default prgenv-gnu/24.11:v2 -- g++  -std=c++17 -O3 -march=native -fopenmp \
    -DNDEBUG -I omp  -o apply_cpu apply_driver.cpp omp/d3/proteus_hex8/*_operator.cpp
uenv run --view=default prgenv-gnu/24.11:v2 -- nvcc -std=c++17 -O3 -arch=sm_90 \
    -DNDEBUG -diag-suppress 177 -I cuda -o apply_gpu -x cu apply_driver.cpp cuda/d3/proteus_hex8/*_operator.cu
sbatch gh200.sbatch
```

A standalone generation emits no `op/` wrapper, which is what makes this build self-contained —
and is the same `supports_op_wrapper=False` that means nothing in the SFEM frontend can reach
these kernels yet.

## Measured

One Alps GH200 node (`nid006544`): NVIDIA GH200 120GB, Neoverse-V2 host, 72 OpenMP threads with
`OMP_PROC_BIND=true` and `OMP_PLACES=cores`, best of 3, `real_t` = double. Full log in
`gh200.log`.

| ndof | elements | host MDOF/s | GH200 MDOF/s | device / host |
|---|---|---|---|---|
| 107,811 | 32,768 | 208.3 | 1705.9 | 8.2x |
| 823,875 | 262,144 | 203.5 | 2224.9 | 10.9x |
| 2,738,019 | 884,736 | 209.1 | 2332.9 | 11.2x |
| 6,440,067 | 2,097,152 | 210.0 | 2347.9 | 11.2x |

Both sides are saturated at the top of the sweep: the host is flat across the whole range
(203-210) and the device gains 0.6% from 2.7M to 6.4M dof, so the 11.2x is a ratio between two
saturated numbers rather than an artefact of an under-filled problem.

Answers agree to round-off at every size. Three of the four sizes match to all 17 printed digits
in both norms; at 2,738,019 dof the L2 norms differ in the last digit — 1.0738402690258097e-2
against ...95, about 2e-16 relative — which is what reassociating a sum through atomics does.

## What this establishes, and what it does not

**Establishes:** the framework generates CUDA that compiles with nvcc 12.6, runs on an H100, and
computes the same answer as its own OpenMP kernel for the same material and element, at a
throughput measured on a saturated problem.

**Does not establish** anything about the other two kernel families. The energy family reaches
CUDA because it already lowers its mesh loop to a grid-stride `__global__`; `residual_codegen`
and `inexact_apply_codegen` do not, and they additionally spell a CPU vector lane that
`backends/cuda.py` rejects. Nor does it establish that anything in the SFEM library can *call*
these kernels: `CMakeLists.txt` globs only `*.cpp`/`*.hpp`, so no `.cu` reaches the build, and
the CUDA backend publishes no `sfem::Op`.

**A gap this measurement walked past:** the generated launcher returns `SFEM_SUCCESS` immediately
after an asynchronous `<<<>>>` launch, so it cannot report a launch failure. This driver calls
`cudaDeviceSynchronize` itself and checks it. Correct for a stream-based API, but it is currently
inherited rather than decided.
