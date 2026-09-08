# python/

Kernel synthesis, and the analysis scripts around the benchmark.

Everything here runs against the repository venv (`../../../venv/bin/python`), which is
where SymPy, NumPy and matplotlib live.

## Code generators

| script | writes | consumed by |
|---|---|---|
| `synthesize_cvfem_hex8_ns_upwind_sympy.py` | `src/generated/cvfem_hex8_ns_upwind_sympy_kernels.hpp`, `subpar/cvfem_hex8_ns_upwind_sympy_subpar.hpp` | `src/hex8/cvfem_hex8_layout_common.hpp`, `src/hex8/cvfem_hex8_ns_core.hpp`, `cuda/cvfem_hex8_ns_cuda.cu` |
| `synthesize_cvfem_tet4_ns_upwind_sympy.py` | `src/generated/cvfem_tet4_ns_upwind_sympy_kernels.hpp` | `src/tet4/cvfem_tet4_ns_upwind_kernels.hpp` |
| `gen_mms_case.py` | `src/cases/cvfem_ns_mms_case.hpp` (stdout by default) | `drivers/cvfem_hex8_ns_ssgmg.cpp` |

The generated headers are **committed artifacts**, compiled as ordinary source. Nothing in
the build invokes SymPy; regeneration is a deliberate manual step.

```sh
../../../venv/bin/python synthesize_cvfem_hex8_ns_upwind_sympy.py           # write
../../../venv/bin/python synthesize_cvfem_hex8_ns_upwind_sympy.py --check   # verify only
../../../venv/bin/python gen_mms_case.py > ../src/cases/cvfem_ns_mms_case.hpp
```

`--check` regenerates in memory and diffs against what is on disk, writing nothing and
exiting non-zero on a difference. It ignores the recorded SymPy version line, which is the
one thing that is *supposed* to move when the generator is run under a different SymPy.
Run it after any change to `cvfem_codegen.py`: the standing contract for that file is that
the emitted headers come out byte for byte as they were.

The committed headers were emitted with SymPy 1.14.0. They regenerate byte-identically
under 1.12.1, so the CSE output is stable across at least that range — but the version is
recorded in each header precisely because that is not guaranteed in general.

## `cvfem_codegen.py`

The shared layer, extracted from the two synthesizers, which were a fork pair. It holds
the `scalar_t` C99 printer, one parameterized `cse_emit` in place of the seven CSE
emitters that existed, the DOF numbering, the sign locals, the BSR block selection, and
`face_flux_residual` — the sub-control-surface flux, which was thirty-five lines duplicated
character for character between HEX8 and TET4.

Element-specific machinery deliberately stays with its element: the `SCS` and shape
function tables, HEX8's isoparametric per-face geometry and its `split_generated`
quarantine partitioner, and TET4's SIMD lane machinery.

### Two known, deliberate divergences

**TET4 does not pin the reciprocal spelling.** HEX8 prints `1/x` as `scalar_t(1)/x`;
TET4 uses C99CodePrinter's own rendering, which is `1.0/x` on SymPy 1.12 and
`scalar_t(1)/x` on newer releases — so the TET4 header's bytes depend on the SymPy
version, and HEX8's do not. `set_stable_pow(True)` in
`synthesize_cvfem_tet4_ns_upwind_sympy.py` fixes that and changes 35 lines of the
generated header. Several of those lines are inside the SIMD paths, where the reciprocal
is taken against a vector type and the literal's type selects an overload. The flipped
header was generated and compiled here and builds clean, SIMD paths included; what has not
been checked is whether it moves the object code or the benchmark. So it is a change to a
measured kernel, and wants a measurement behind it rather than a refactor.

**These generators do not use SFEM's own codegen framework.** `python/codegen/sfem_codegen.py`
in the parent repository provides direct analogues of most of `cvfem_codegen.py` —
`SFEMCodePrinter`, `c_gen`, `adjugate3`, `assign_nnz_matrix` — and there is prior CVFEM art
beside it (`cvfem_tet4_convection.py` and friends). The spike does not use them because it
needs `scalar_t(...)` spellings rather than `real_t`, `SFEM_INLINE SFEM_HOST_DEVICE`
qualifiers, `CVFEM_ATOMIC_ADD`, and `slots[]*16` BSR addressing, none of which the
framework's emitters produce. That is a real reason, not an oversight, but it is a
divergence worth revisiting whenever the framework grows those.

## Reference implementation

`tet4_cvfem_ns_upwind_kernel.py` is a pure-Python float implementation of the same TET4
kernel — no SymPy, no NumPy — used as an executable specification. It writes nothing;
running it executes its self-tests, including a finite-difference check of the Jacobian.

## Analysis

| script | |
|---|---|
| `benchmark_jacobian_variants.py` | drives `build/cvfem_tet4_ns_upwind_bench` and tabulates the layouts |
| `report_cvfem_bench.py` | the HTML + inline-SVG benchmark report |
| `plot_cvfem_bench.py` | matplotlib figures from the same CSV; imports `report_cvfem_bench` |
| `analyze_xctrace_export.py` | parses Instruments trace exports (macOS) |
| `nullspace_eval.py` | reproduces the multigrid null-space behaviour on a model Stokes system |
| `create_xdmf.py` | SFEM raw output to ParaView XDMF; driven by `scripts/create_xdmf.sh` |
