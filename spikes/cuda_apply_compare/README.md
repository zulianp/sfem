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

The kernels cover three of the framework's four families. From the **energy** family: the
neo-Hookean Ogden matrix-free apply on a tensor-product hexahedron, which is the shape this
framework cares most about; the two P1-simplex metric bodies (laplace TRI3 and TET4, gradient and
apply); the constant-P1 AoS unit body (linear elasticity TET4 gradient); and the two mesh-order
tensor-product bodies whose stream arrays were miscounted (laplace QUAD4 and HEX8 gradient).
Those five are there because they were shipping a serial mesh loop inside `__global__`, or not
compiling at all, until that was measured.

From the **residual** family: the viscous Mooney-Rivlin Kelvin-Voigt residual on a TET4, reading
a current and a previous state. From the **boundary-residual** family: the Neumann traction on a
triangle shell. Neither could be generated for CUDA at all until the mesh-kernel lowering moved
onto the target — the backend accepted `energy_soa` units and raised on everything else.

**Throughput** uses a structured lattice, because a timing on scrambled connectivity is not a
result: hexahedra for the tensor-product apply, and the Kuhn subdivision of the same lattice —
six tetrahedra per cell, so the node numbering keeps its locality — for the simplex kernels.

`handwritten_compare.cu` is the third question and the one the other two cannot answer: **is the
generated kernel as fast as the one SFEM wrote by hand?** It links `cu_tet4_laplacian_apply` and
`cu_tet4_linear_elasticity_apply` beside their generated counterparts and runs both on the same
mesh. One flat geometry buffer serves both — the hand-written kernel reads it with an
`nelements` stride, the generated one as component pointers into the same memory — so neither
gets a layout the other does not. With `kappa = 1`, SFEM's Laplacian *apply* and the generated
Laplacian energy's *gradient* are the same matrix-vector product, so each row is a correctness
check against the reference implementation as well as a rate.

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

`build_handwritten.sh` builds the comparison against SFEM's own kernels. It needs the library's
include flags rather than the shim, because the hand-written kernel is part of the library:
it reads `CXX_INCLUDES` and `CXX_DEFINES` out of an existing `build/CMakeFiles/sfem.dir/flags.make`,
adds the uenv's `mpi.h` (nvcc is not the MPI wrapper), and links `base/sfem_base.cpp` and `-lmpi`
for `sfem_abort`'s error path.

A standalone generation emits no `op/` wrapper, which is what makes this build self-contained —
and is the same `supports_op_wrapper=False` that means nothing in the SFEM frontend can reach
these kernels yet. `shim/` supplies the scalar aliases and the two `smesh` enums that
`op/*_c_abi.hpp` names in declarations and that a standalone generation does not carry.

## Measured

One Alps GH200 node (`nid006546`): NVIDIA GH200 120GB, compute capability 9.0, Neoverse-V2 host,
72 OpenMP threads with `OMP_PROC_BIND=true` and `OMP_PLACES=cores`, nvcc 12.6.20 at `-arch=sm_90`,
best of 3, `real_t` = double. `*.log` is gitignored, so the tables below are the record — the sibling spike keeps its results the same way.

Neo-Hookean Ogden apply, PROTEUS_HEX8:

| ndof | elements | host MDOF/s | GH200 MDOF/s | device / host |
|---|---|---|---|---|
| 107,811 | 32,768 | 220.8 | 1899.2 | 8.6x |
| 823,875 | 262,144 | 216.9 | 2258.5 | 10.4x |
| 2,738,019 | 884,736 | 216.5 | 2305.6 | 10.6x |
| 6,440,067 | 2,097,152 | 214.5 | 2337.5 | 10.9x |
| 12,519,843 | 4,096,000 | 211.5 | 2352.6 | **11.1x** |

Laplace gradient, TET4, constant-metric closed form:

| ndof | elements | host MDOF/s | GH200 MDOF/s | device / host |
|---|---|---|---|---|
| 35,937 | 196,608 | 442.3 | 2344.5 | 5.3x |
| 274,625 | 1,572,864 | 516.4 | 4460.7 | 8.6x |
| 912,673 | 5,308,416 | 524.8 | 5218.1 | 9.9x |
| 2,146,689 | 12,582,912 | 520.4 | 5413.7 | 10.4x |
| 4,173,281 | 24,576,000 | 523.1 | 5491.3 | **10.5x** |

Both sides are saturated at the top of each sweep. The host is flat across the whole range in
both cases (214-221 and 516-525), and the device gains 0.1% on the hexahedral apply and 1.2% on
the tetrahedral gradient over the last doubling, so each ratio is between two saturated numbers
rather than an artefact of an under-filled problem.

**Every output vector agrees to at most 5.7e-16** relative to its largest entry, across all
eighteen — including the residual family's three and the boundary family's three.

**The two arms call two different names with two different arities**, and
`SFEM_KERNEL` / `SFEM_STREAM_PARAM` / `SFEM_STREAM_ARG` at the top of the driver are the whole
of that difference. SFEM names every device entry point `cu_` and ends it with a stream —
`cu_laplacian_apply`, `cu_linear_elasticity_apply` — because a host and a device implementation
of one operator are two symbols in one library, and the generated tree follows that now. The
numbers above are measured across that change and are within 1% of what the same sweep recorded
before it, so neither the rename nor the stream argument costs anything.
The comparison is not vacuous in either of the two ways it could be: every vector is fully
non-zero with every entry distinct, so nothing agrees by being empty, and moving a single entry
by 1e-9 relative makes `compare.py` report a failure at 1.2e-10.

This replaced a comparison that printed each run's own L2 and L-infinity norms. Two answers can
share both norms and still differ entrywise, so that version could only see gross disagreement —
and it was the only check the PROTEUS_HEX8 apply had.

## Against SFEM's own kernels

Same node, same sweep. Ratios at the top, where both sides are saturated.

| TET4 apply | ndof | hand-written | generated | generated / hand-written | worst rel diff |
|---|---|---|---|---|---|
| Laplacian | 4,173,281 | 7245.5 | 7149.4 MDOF/s | **0.987x** | 4.71e-16 |
| linear elasticity | 12,519,843 | 7904.0 | 7921.8 MDOF/s | **1.002x** | 6.47e-16 |

Within 1.3% either way, and a dead heat on the heavier of the two. Both sides saturate over the
last doubling (Laplacian 7118.6 → 7245.5 hand-written and 7021.0 → 7149.4 generated; elasticity
7500.4 → 7904.0 and 7572.4 → 7921.8), so these are ratios between saturated numbers. Every
output entry is non-zero and agrees to at most 6.5e-16 relative.

**A hazard this found, which compiling did not.** The first run faulted with an illegal memory
access. The generated tree guards its scalar types with `#if __has_include("sfem_base.hpp")` and
falls back to `idx_t = ptrdiff_t`, but **SFEM's `idx_t` is `int`** — compiled standalone, the
generated half read the driver's four-byte indices as eight-byte ones. Both halves compiled
cleanly; only running it showed anything. The generated half takes SFEM's include path now, so
both sides take their types from the same header.

## Mooney-Rivlin Kelvin-Voigt Newmark

The heaviest thing the framework generates, and the one with no hand-written counterpart in
SFEM, so the number stands on its own rather than as a ratio. Two kernels, because the material
is two units: the *elastic* apply is the energy family's matrix-free Hessian action on the
hyperelastic solid, and the *viscous* Jacobian action is the residual family's, reading a
current state, a previous state and a direction.

| TET4, 12,519,843 dof, 24,576,000 elements | host | GH200 | device / host |
|---|---|---|---|
| elastic apply | 245.3 | **4252.0** MDOF/s | **17.3x** |
| viscous Jacobian action | 136.9 | **1938.7** MDOF/s | **14.2x** |

Both saturate: the host is flat across the whole range (245-256 and 135-147) and the device
gains 0.1% and loses 0.2% over the last doubling.

Two things worth reading off this. The speedups are much larger than the 11.1x and 10.5x the
simpler kernels give, which is the expected shape — this material is far enough from memory
bound that the GH200's arithmetic advantage over 72 Grace cores matters more than its bandwidth
advantage. And the viscous unit costs about 2.2x the elastic one on *both* sides (245 against
137 on the host, 4252 against 1939 on the device); it reads three vector fields where the
elastic reads two, and a ratio that holds across hardware says the cost is in the arithmetic and
the extra field rather than in anything target-specific.

These are the standard mesh layout. The packed layout exists on the host only: the OpenMP
thread-local scratch it needs has no device meaning, which is a design question rather than a
spelling one.

## What this establishes, and what it does not

**Establishes:** the framework generates CUDA that compiles with nvcc 12.6, runs on an H100,
computes the same answer as its own OpenMP kernel — entry by entry — for the energy, residual and
boundary-residual families, across the tensor-product, constant-P1 simplex and mesh-order
lowerings, at throughputs measured on saturated problems, and **matches SFEM's hand-written
device kernels to within 1.3% on the two elements where both exist**.

**Does not establish** anything about the **mixed-order** residual family, which generates and
compiles for CUDA but is not compared here: its driver case needs two element types (TET10
velocity over TET4 pressure) and that is its own piece of plumbing. Nor anything about
`inexact_apply_codegen`, which still emits `_operator.cpp` with host `extern "C"` functions around
an OpenMP mesh loop. Nor that anything in the SFEM library can *call* any of these kernels:
`CMakeLists.txt` globs only `*.cpp`/`*.hpp`, so no `.cu` reaches the build, and the CUDA backend
publishes no `sfem::Op`.

**Two defects only the link stage found.** Both binaries compiled cleanly with a host function
containing `blockIdx` in it — `mooney_rivlin..._hessian_bsr_i_msoa` got a grid-stride loop while
still declared `__host__ __device__ int`, and the failure was
`undefined reference to __device_builtin_variable_blockIdx`. And the boundary family's reference
tables were `__host__`-only while the `__host__ __device__` element function calls them, which is
`warning #20011-D` and not an error. Compiling every translation unit is not the same gate as
linking one binary out of them.

**A gap this measurement walked past:** the generated launcher returns `SFEM_SUCCESS` immediately
after an asynchronous `<<<>>>` launch, so it cannot report a launch failure. This driver calls
`cudaDeviceSynchronize` itself and checks it. Correct for a stream-based API, but it is currently
inherited rather than decided.
