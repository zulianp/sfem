# Shared CPU/GPU micro-kernels: feasibility and plan

Item from TIGHT_API.md: "Same/Common CPU and GPU micro kernels should not be duplicated, but
collected into headers with template functions and proper macros, that can be included directly
in the kernel files (careful handling of CUDA-related macros)."

Surveyed at `717c5cfbf` (branch `tight-api`). Read-only survey: nothing was built or timed.

## Verdict

| Area | Verdict | Main obstacle |
| --- | --- | --- |
| Handwritten: index helpers (`sshex8_{nxe,txe,lidx}`, `sstet4_{nxe,txe}`) | Feasible now | 4 copies, one of them in smesh |
| Handwritten: CRS lookup (`*_linear_search`, `*_find_col(s)`, `*_local_to_global`) | Feasible | Host binary search and `omp atomic` need host/device forms |
| Handwritten: geometry (`hex8/tet4_fff`, `*_adjugate_and_det`, `sub_fff`, `sub_adj`) | Feasible | Device reads SoA with `stride`, host reads AoS. smesh already has templated host-device versions |
| Handwritten: Laplacian element kernels (hex8, tet4, tet10, spectral, lagrange) | Feasible | About half are identical apart from qualifiers. Taylor variants use different table layouts |
| Handwritten: HEX8 linear elasticity | Partly feasible | Same math in different shapes (whole matrix vs 9 blocks). Register pressure on device |
| Generated: support headers (`kernel_math`, `geometry_kernels`, `tensor_product_kernels`) | Feasible now | The `.cuh` is a strict subset of the `.hpp` once qualifiers are substituted |
| Generated: per-qp math in `*_local` / `*_hessian` | Text is already identical. Emitting it once is feasible; that it performs is not proven | CPU vectorisation depends on the lane-blocked layout |

## 1. Macros and language constraints

### What exists (`base/sfem_base.hpp`)

| Macro | Line | Host compiler | nvcc / hipcc |
| --- | --- | --- | --- |
| `SFEM_INLINE` | :70, :73 | `inline` in both branches | `inline` |
| `SFEM_FORCE_INLINE` | :71, :74 | `inline __attribute__((always_inline))` with NDEBUG, plain `inline` otherwise | same. nvcc's own `__forceinline__` is `__inline__ __attribute__((always_inline))` (crt/host_defines.h), so the NDEBUG spelling is equivalent on device |
| `SFEM_HOST_DEVICE` | :79-83 | empty | `__host__ __device__` (`__CUDACC__` or `__HIPCC__`) |
| `SFEM_DEVICE_INLINE` | :92 | `inline` | `__host__ __device__ inline` |
| `SFEM_RESTRICT` | :95-99 | `__restrict__` (`__restrict` on _WIN32) | `__restrict__` |

- **The comment at :85-91 is stale.** It says `SFEM_INLINE` "expands to *nothing* unless NDEBUG is set", but :69-75 define it as `inline` in both branches. The guidance it gives still works (`static SFEM_INLINE SFEM_HOST_DEVICE`), but the reason it states is false. Correcting it needs Patrick's approval because of the standing rule against rewording existing comments. The alternative is a new comment next to it.
- `SFEM_INLINE`, `SFEM_FORCE_INLINE` and `SFEM_RESTRICT` are **unguarded** in sfem_base.hpp. Generated headers define all three under `#ifndef` (see section 3.4), so the value a translation unit (TU) sees depends on include order. A generated header that sets `SFEM_RESTRICT` empty and is included before sfem_base.hpp leads to a macro-redefinition warning, and the TU then mixes two spellings.
- smesh (`external/smesh/src/base/smesh_base.hpp`) mirrors these:
  - `SMESH_INLINE` is `inline` (:83, and its debug branch is commented out)
  - `SMESH_FORCE_INLINE` uses `__attribute__((always_inline))` unconditionally (:84)
  - `SMESH_RESTRICT` (:91)
  - `SMESH_HOST`/`SMESH_DEVICE` (:97-106) and `SMESH_HOST_DEVICE` (:114).
- Precedent that already works: `frontend/sfem_node_obstacle_kernels.hpp` is written as `SFEM_HOST_DEVICE SFEM_INLINE` and included by both `frontend/sfem_NodeObstacleView.cpp` and `frontend/cuda/sfem_NodeObstacleView_cuda.cu`. That is the pattern this plan generalises.

### C translation units

- No `.c` file under `operators/`, `frontend/` or `base/`. operators/ holds 106 `.cpp` and 24 `.cu` files.
- The only `.c` files in the tree are:
  - `hardware/opencl_devices.c`
  - `workflows/ldp_containment/omp/{region_cost,region_count,team_cost}.c` (standalone microbenchmarks)
  - smesh's vendored `external/mpi-sort`.
  The CMake glob `drivers/cuda/*.c` (CMakeLists.txt:380) matches nothing.
- `project(... LANGUAGES "C" "CXX")` (CMakeLists.txt:7) still enables C, and 102 operator headers keep `#ifdef __cplusplus extern "C"` guards from the C era.
- **Constraint:** the shared kernel headers are C++17-only. They use templates (`CMAKE_CUDA_STANDARD 17`, cmake/SFEMDependencies.cmake:18-20).
  - Each one starts with `#ifndef __cplusplus #error ... #endif`.
  - A shared header must never be included *inside* an `extern "C" {` block: a template with C linkage is an error. Today the legacy headers put their includes before the `extern "C"` block (e.g. operators/neohookean_ogden.hpp:4-11). Keep it that way.

### What the shared headers need

These go in `base/sfem_base.hpp` (or a new `base/sfem_kernel_macros.hpp` that sfem_base includes). Each is small:

1. `SFEM_DEVICE_CODE`: `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`. It is the only sanctioned way to branch inside a host-device body; `__CUDACC__` is true in nvcc's host pass too.
2. `sfem_atomic_add(T *p, T v)`, a `SFEM_HOST_DEVICE` template:
   - device: `atomicAdd`
   - host: `#pragma omp atomic update`
   It replaces the split between `tet4_local_to_global` (tet4_inline_cpu.hpp:67, `#pragma omp atomic update`) and `cu_tet4_local_to_global` (cu_tet4_inline.hpp:170, `atomicAdd`).
3. Reorder `base/sfem_unroll.hpp`: test `__CUDACC__`/`__HIPCC__` **before** `_OPENMP >= 202011`. Today the OpenMP-5.1 branch wins whenever `_OPENMP` is set, so a host-device body would get `omp unroll` in nvcc's host pass. Replace the clang-only `#pragma unroll(8)` (hex8_inline_cpu.hpp, `hex8_find_cols`) with `SFEM_UNROLL_N(8)`: gcc ignores `#pragma unroll(8)`.
4. Guard `SFEM_RESTRICT`/`SFEM_INLINE` in sfem_base.hpp with `#ifndef`, or (preferred) make the generator stop defining them whenever sfem_base.hpp is present (section 3.4).
5. Remove private `MIN`/`MAX`/`POW2` fallbacks from shared code. `cu_tet4_inline.hpp:4-14` defines `MIN`/`MAX` with `<`/`>` under `#ifndef`, while `base/sfem_macros.hpp:7,11` uses `<=`/`>=`. Whichever header comes first wins. Shared headers include `sfem_macros.hpp` instead.

Function form used throughout:

```cpp
template <typename geom_t, typename fff_t>
static SFEM_INLINE SFEM_HOST_DEVICE void hex8_fff(/* ... */);
```

- `static` gives internal linkage, as the stale comment correctly prescribes.
- `SFEM_FORCE_INLINE` is used only where the current device copy is `__forceinline__`.
- No body may call a host-only function. That rules out `fflush`, `SFEM_ERROR`, and `find_idx_binary_search` (`extern "C"`, algebra/sfem_sortreduce.cpp:38).
  - Example: `hex8_find_cols` has an NDEBUG block with `printf` + `fflush` (hex8_inline_cpu.hpp:251-262). It becomes `assert` only.
  - nvcc only warns about such calls in a template until the template is instantiated in device code, so the device agreement test (section 4) must instantiate every function.

## 2. Handwritten code

### 2.1 Inventory

- **Name pairs:** 257 distinct `cu_*` identifiers appear under `operators/**/cuda`. 89 of them have a same-named host definition.
  - About 40 of the 89 are launcher or dispatch pairs (`cu_laplacian_apply` vs `laplacian_apply`, `cu_tet4_linear_elasticity_apply`, ...). They are not math duplicates; they stay target-specific.
- **Element-level pairs:** 116 device-callable (`__device__`) `cu_*` functions; 48 have a host twin of the same name.
  - Those 48 total 1,997 device-side and 1,782 host-side lines.
  - A few more are twins under a different name: `cu_macro_tet4_sub_fff_0`, and the HEX8 elasticity matrix blocks.
- Comparison method: token-level diff after stripping comments.
  - **A** = identical apart from qualifiers (`static`, `inline`, `__host__ __device__`, `SFEM_INLINE`), the `template<...>` head, the `cu_` prefix and `const`.
  - **B** = A plus casts or parameter types.
  - **C** = the device reads per-element data in SoA with a `stride` argument while the host reads AoS (otherwise A/B).
  - **D** = different algorithm or shape.

| # | Host | Device (and smesh) | Lines h/d | Class |
| --- | --- | --- | --- | --- |
| 1 | `hex8_laplacian_diag_fff_integral` hex8/hex8_laplacian_inline_cpu.hpp:445 | `cu_…` hex8/cuda/cu_hex8_laplacian_inline.hpp:443 | 25/26 | A |
| 2 | `hex8_laplacian_apply_fff_integral` :332 | :130 | 112/113 | A |
| 3 | `hex8_laplacian_matrix_fff_integral` :213 | :10 | 118/119 | B: host has 15 extra `(scalar_t)` casts; one device temporary is `accumulator_t` |
| 4 | `hex8_laplacian_matrix_ij_taylor` :471 | :288 | 35/43 | D: host takes `H[3]` arrays, device takes `Hx,Hy,Hz` scalars |
| 5 | `hex8_laplacian_matrix_fff_taylor` :507 | :332 | 18/52 | D: host table `hex8_g_0[i][3]`, device split tables `hex8_g_0_x/_y/_z` |
| 6 | `hex8_laplacian_apply_fff_taylor` :526 | :385 | 41/57 | B (casts) + calls #4 |
| 7 | `hex8_adjugate_and_det` hex8/hex8_inline_cpu.hpp:90 | cu_hex8_inline.hpp:52; smesh_hex8_inline.hpp:106 | 61/63 | C. The smesh copy is already `template` + `SMESH_HOST_DEVICE` and takes scalar corner arguments |
| 8 | `hex8_fff` hex8_inline_cpu.hpp:13 | hex8/cuda/cu_hex8_fff.cu:41; smesh_hex8_inline.hpp:13 | | C. The smesh copy writes `fff0..fff5` through separate pointers, which serves any stride |
| 9 | `hex8_linear_search` hex8_inline_cpu.hpp:211 | cu_hex8_inline.hpp:8 | 13/14 | A. Also identical to the tet4, tri3, tri6 and tet10 copies |
| 10 | `hex8_find_col` :225 | :23 | 3/4 | A |
| 11 | `hex8_find_cols` :229 | :28 | 34/23 | A apart from the host NDEBUG printout |
| 12 | `hex8_ref_shape_grad` hex8_inline_cpu.hpp:383 | cu_hex8_linear_elasticity_matrix_inline.hpp:54 | | A (`0` vs `false`) |
| 13 | `hex8_ref_shape_grad_{x,y,z}` hex8_inline_cpu.hpp:459/480/501 | cu_hex8_linear_elasticity_inline.hpp:7/28/49 | 20 each | A (host has `SFEM_UNUSED`) |
| 14 | `hex8_linear_elasticity_diag` hex8/hex8_linear_elasticity_inline_cpu.hpp:3236 | cu_hex8_linear_elasticity_inline.hpp:2673 | 274/280 | A |
| 15 | `linear_elasticity_matrix_sym` …_inline_cpu.hpp:3180 | cu_hex8_linear_elasticity_matrix_inline.hpp:306 | 55/56 | A |
| 16 | `hex8_linear_elasticity_apply_adj` :152 | cu_hex8_linear_elasticity_inline.hpp:70 | 21(+94 helper)/130 | D: different formulation |
| 17 | `hex8_linear_elasticity_matrix` :207 (affine, integrated) | `cu_hex8_linear_elasticity_integral_matrix_block_{i}_{j}` cu_hex8_linear_elasticity_integral_inline.hpp:10… (9 functions) | 2249/3166 | D: same closed form, split per 3×3 block, separate CSE |
| 18 | none | `cu_hex8_linear_elasticity_matrix_block_{i}_{j}` (per qp) cu_hex8_linear_elasticity_inline.hpp:205… | –/2467 | Device only (cu_sshex8_linear_elasticity.cu:315,507) |
| 19 | `tet4_fff` tet4/tet4_inline_cpu.hpp:224 | cu_tet4_inline.hpp:68; smesh_tet4_inline.hpp:18 | 55/57 | C |
| 20 | `tet4_adjugate_and_det` :285 | :17; smesh_tet4_inline.hpp:82 | 49/50 | C |
| 21 | `tet4_linear_search` :17 | :126 | 15/14 | A |
| 22 | `tet4_find_col` :33 | :141 | 10/4 | D: the host switches to `find_idx_binary_search` above 32 entries (host-only `extern "C"`) |
| 23 | `tet4_find_cols` :44 | :146 | 22/23 | A |
| 24 | `tet4_local_to_global` :67 | :170 | 25/24 | D: `omp atomic` vs `atomicAdd`, `#pragma unroll(4)` |
| 25 | `tet4_laplacian_apply_fff` tet4/tet4_laplacian_inline_cpu.hpp:54 | cu_tet4_laplacian_inline.hpp:7 | 20/20 | C: host also takes `u0..u3` scalars |
| 26 | `tet4_laplacian_diag_fff` :32 | :28 | 10/11 | C: host also writes `e0..e3` |
| 27 | `tet4_linear_elasticity_apply_adj` tet4_linear_elasticity_inline_cpu.hpp:63 | tet4/cuda/cu_tet4_linear_elasticity.cu:16 | | D |
| 28 | `tet10_laplacian_trial_operand` tet10/tet10_laplacian_inline_cpu.hpp:6 | cu_tet10_laplacian_inline.hpp:7 | 23/25 | C |
| 29 | `lagrange_hex_triad_interpolate` spectral_hex/lagrange_hex_laplacian_inline.hpp:14 | cu_lagrange_hex_laplacian_inline.hpp:7 | 64/69 | B: device hoists row pointers. Summation order to be checked |
| 30 | `lagrange_hex_triad_integrate_add` :79 | :77 | 66/66 | B (as #29) |
| 31 | `lagrange_hex_laplacian_apply` :146 | :144 | 109/46 | D |
| 32 | `spectral_hex_laplacian_apply` spectral_hex_laplacian_inline.hpp:70 | cu_spectral_hex_laplacian_inline.hpp:7 | 77/77 | A |
| 33 | `sshex8_{nxe,txe,lidx}` sshex8/sshex8.hpp:8/16/18 | cu_sshex8_inline.hpp:21/29/31; smesh_sshex8.hpp:8/16/21; smesh/…/cuda/smesh_sshex.cuh:21/29/31 | 7/1/7 | A, **4 copies** |
| 34 | `hex8_sub_adj_0` sshex8.hpp:26 | cu_hex8_inline.hpp:116; smesh_sshex8.hpp:50 | 17/19 | C, 3 copies (smesh: `template` + `SMESH_HOST_DEVICE`) |
| 35 | `hex8_sub_fff_0` sshex8/sshex8_laplacian.cpp:19 | cu_sshex8_inline.hpp:9; smesh_sshex.cuh:9 | | C, 3 copies |
| 36 | `sstet4_sub_fff_{0..5}` sstet4/sstet4_laplacian.cpp:33,45,64,82,106,124 | sstet4/cuda/cu_sstet4_laplacian.cu:51,64,84,103,128,147 | | A, except that `L` is `scalar_t` on the host and `int` on the device |
| 37 | `sstet4_{nxe,txe}` sstet4_laplacian.cpp:15/31 | cu_sstet4_laplacian.cu:32/48; smesh_sstet4.hpp:24/27 | | B: `floor`, `pow` vs `POW3` |
| 38 | `tet4_sub_fff_0` macro_tet4/macro_tet4_inline_cpu.hpp:8 | `cu_macro_tet4_sub_fff_0` macro_tet4/cuda/cu_macro_tet4_inline.hpp:7 | | C |
| 39 | `subtet_gather`/`subtet_scatter_add` macro_tet4_linear_elasticity.cpp:26/35 | cu_macro_tet4_linear_elasticity.cu:196/257 | | D: table loop vs 4 explicit indices |
| 40 | `quadshell_4_integrate` quadshell4/quadshell4_integrate_values.cpp:8 | cuda/cu_quadshell4_integrate_values.cu:9 | | A (`scalar_t`→`T`) |

Notes on the inventory:

- **`smesh_sshex.cuh` is an orphan copy.** `external/smesh/src/mesh/semistructured/cuda/smesh_sshex.cuh` (283 lines) is byte-identical to `operators/sshex8/cuda/cu_sshex8_inline.hpp` apart from includes and guard. Nothing in smesh or SFEM includes it. It has no namespace, so a TU that included both would get redefinition errors.
- **The same lookup in 7 headers.** `*_linear_search`, `*_find_col` and `*_find_cols` are copied in `hex8`, `tet4`, `tri3`, `tri6` and `tet10` `*_inline_cpu.hpp` plus `cu_hex8_inline.hpp` and `cu_tet4_inline.hpp`: 21 definitions. They differ only in the node count (`8`, `4`, `10`, ...). That is three templates on `<int N, typename idx_t>`.
- **The HEX8 elasticity file mostly is not duplicated in the device tree.** `hex8_linear_elasticity_inline_cpu.hpp` (3,723 lines) is dominated by two functions:
  - `hex8_linear_elasticity_matrix` (2,249)
  - `hex8_linear_elasticity_matrix_coord_taylor_sym` (722). It is referenced only from commented-out calls (hex8_linear_elasticity.cpp:908,980), so it belongs in DEAD.md, not here.
  The device side is cu_hex8_linear_elasticity_inline.hpp (2,953 lines: per-qp blocks + diag) and cu_hex8_linear_elasticity_integral_inline.hpp (3,166 lines: the integrated blocks, used only at cu_sshex8_linear_elasticity.cu:309).
- **The real layout split.** Rows 7, 8, 19, 20, 25, 26, 28, 34, 35 and 38 differ only because device code reads the per-element geometry in SoA (`fff[k * stride]`), while the host first copies it into a local `fff[6]`. That is not a numerical difference, and it is the one design decision of the plan (section 2.2).

### 2.2 Layout of the shared headers

- **File placement.**
  - `operators/<elem>/<elem>_kernels.hpp`: geometry, reference shape functions, sub-element maps.
  - `operators/<elem>/<elem>_<physics>_kernels.hpp`: `laplacian`, `linear_elasticity`, ...
  - `operators/sfem_crs_kernels.hpp`: the element-agnostic CRS lookup and `local_to_global`.
- **Why outside `cuda/`.** The host-only build never puts `operators/*/cuda` on the include path (the module list at CMakeLists.txt:162-183 is CUDA-only). So the shared header must sit next to the host sources. The `cuda/` folder keeps the `.cu` launchers, in line with the rule that CUDA sources live in local `cuda/` folders.
- **Unique names.** File names must be unique across the tree, because install flattens headers into one `include/` (the CMakeLists.txt comment at the generated-include block says the same for generated files).
- **Includes.** A shared header includes only `sfem_base.hpp` and `sfem_macros.hpp`. Never `sfem_cuda_base.hpp`, `<cuda_runtime.h>` or `<omp.h>`: today `cu_hex8_laplacian_inline.hpp:4` pulls in `sfem_cuda_base.hpp`, which a `.cpp` cannot include in a host-only build.
- **Templates.** Every function is a template on every type it computes with (`geom_t`, `scalar_t`, `accumulator_t`, `idx_t`), per the standing rule. Host call sites today rely on the global typedefs from `configuration/sfem_config.h.in:46-77`. A template parameter of the same name shadows them, and deduction from the argument pointers keeps call sites unchanged in most cases.
- **SoA vs AoS.** Each geometry or per-element function takes a `const ptrdiff_t stride` (the device's current signature). Host callers pass the literal `1`, which constant-folds after inlining. This choice must be confirmed by the timing gate, not assumed. The alternative is smesh's per-component output pointers (`fff0..fff5`); that is preferred for the geometry functions, because smesh already has them (next bullet).
- **smesh copies: single source in smesh.**
  - `smesh_hex8_inline.hpp` and `smesh_tet4_inline.hpp` already provide `template` + `SMESH_HOST_DEVICE` `hex8_fff`, `hex8_adjugate_and_det`, `tet4_fff` and `tet4_adjugate_and_det` in `namespace smesh`. They are used by `smesh_jacobians.impl.hpp` / `smesh_fff.impl.hpp`.
  - SFEM's rows 7, 8, 19 and 20 become calls to `smesh::…`. This is also TODO.md's P0 "FFF/Jacobian single source of truth".
  - `sshex8_{nxe,txe,lidx}`, `hex8_sub_adj_0` and `sstet4_{nxe,txe,lidx}` already exist in smesh with `SMESH_HOST_DEVICE`, so SFEM deletes its copies. `smesh_sshex.cuh` is deleted. `hex8_sub_fff_0` moves into `smesh_sshex8.hpp`.
  - smesh changes land on smesh `main` (standing rule).
  - Note: `sshex8_txe` in smesh_sshex8.hpp:16 lacks `static`. Harmless, but make it consistent.
- **The `cu_` prefix.** Micro-kernels lose it: one name, host and device. `cu_` stays for launchers and entry points, which is also the generator's convention (`CUDATarget.entry_point_name`, targets.py:1058). No aliases are kept. Every call site is renamed in the same commit (no-redundant-paths).
- **HIP.** `SFEM_HOST_DEVICE` already covers `__HIPCC__`. The device branch macro must test `__HIP_DEVICE_COMPILE__` as well as `__CUDA_ARCH__`. The hipify shim `base/hip/{cuda_runtime,cuda,cublas_v2}.hpp` exists only untracked in the main worktree, not in `tight-api`, and `SFEM_ENABLE_HIP` is commented out (cmake/SFEMDependencies.cmake:283). Since the shared headers never include CUDA headers, the shim never matters to them.
- **Where the device really wants another shape.** Rows 4, 5 and 17 (Taylor tables split per axis, elasticity matrix per 3×3 block) are cases where the device copy was reshaped, presumably for registers. Under the device-kernels-get-their-own-variant rule, the two shapes become **two functions in the same shared header**, both `SFEM_HOST_DEVICE`, generated from one symbolic source where applicable. They are not two files with diverging copies. The rule leaves the choice of one shape to measurement.

### 2.3 Risks

- **NVCC inlining and registers.** The host formulation of a kernel can raise register count or spill once it is compiled for the device. This applies to rows 16 and 27 (apply_adj) and row 17, if the whole-matrix form were used on device. The CUDA flags already include `-Xptxas=-O3,-v,-warn-spills` (cmake/SFEMBuildTypes.cmake:114). Every step diffs the ptxas register/spill lines of the affected kernels before and after. Any increase blocks the step until the GH200 timing shows no regression.
- **Host vectorisation.**
  - The handwritten host micro-kernels use no intrinsics, no `alignas` and no `__builtin_assume_aligned`.
  - `#pragma omp simd` appears in only 6 `.cpp` sweeps: tet4_patch_gradient, hex8_linear_elasticity, sstet4_{laplacian,linear_elasticity}, sshex8_neohookean_ogden{,_active_strain}.
  - GNU vector types (`vector_size`, base/sfem_vec.hpp) are used only in `vector_laplacian` and `{hex8,sshex8}_vector_laplacian`. Those have no device twin and stay out of scope.
  - The host lane template `hex8_laplacian_apply_fff_integral_soa_tpl` (hex8_laplacian_inline_cpu.hpp:569, used by frontend/ops/sfem_PackedLaplacian.cpp:396) must keep its by-value scalar interface. Templating on the type is compatible with it.
  - What vectorisation does depend on is `SFEM_RESTRICT` on the host signatures. Some device copies drop it (rows 13 and 22). The shared versions keep it: `__restrict__` is valid for nvcc too.
- **Mixed precision.** Unifying B-class rows changes where casts happen. For `scalar_t = jacobian_t = double` the result is unchanged; with the float32 tangent (`storage_bytes 4`) or float geometry it may not be. Each B row needs an explicit choice of which cast placement is kept.
- **Bitwise expectations.**
  - Host before vs after: bitwise identical for A- and C-class rows. The expression order and types are unchanged, and C only changes addressing.
  - B and D rows: tolerance-based. State the tolerance in the test.
  - Host vs device: never bitwise. CUDA builds with `-use_fast_math` (cmake/SFEMBuildTypes.cmake:114) and FMA contraction. The host may add `-ffast-math` under `SFEM_ENABLE_AGGRESSIVE_OPT` (:124).
- **The HEX8→PROTEUS_HEX8 delegation item** (TIGHT_API.md) removes host users of some `hex8_*` kernels. Rows 1-6 and 14-17 are also called by the sshex8 kernels (e.g. cu_sshex8_laplacian.cu:29 calls `cu_hex8_laplacian_matrix_fff_integral`), so they stay needed either way. Do the micro-kernel merge first; it makes the delegation smaller.

## 3. Generated code

### 3.1 What the two targets emit (python/codegen/framework/targets/targets.py)

- **`OpenMPTarget` (:705-773):**
  - qualifiers: `static SFEM_INLINE` (:718), `inline_qualifier` `SFEM_INLINE` (:721)
  - its own `#ifndef SFEM_INLINE` define (:724-729), restrict spelled `RSTR` (:731) with an *empty* inherited `restrict_definition`
  - `#pragma omp parallel for`, `omp simd` and `omp atomic update` (:734-754)
  - policy `VECTOR_LANES` with an `int VS` template parameter and a `for (int lane = 0; lane < ne; ++lane)` loop under `#pragma omp simd` (:761-773, base helpers :253-287, :350-400).
  - The AVX512, ARM SVE and ARM SME subclasses (:776-850) change only metadata and flags; none emits intrinsics.
- **`CUDATarget` (:853-1113):**
  - qualifier `__host__ __device__ __forceinline__` with no `static` (:862-866), and `__restrict__` (:868-872)
  - no `VS` (`effective_vector_size` 1, :964); work items named `*_valueN` instead of `*_laneN` (:905)
  - `atomicAdd` scatter (:911-913), grid-stride loop (:915-950), `block_size = 256` launches (:1077-1098), `cu_` entry points (:1058)
  - `HIPTarget` (:1116) subclasses it.
- **One printer.** The per-qp arithmetic comes from `sp.cse` in the emitters (energy_codegen.py:3486-3502, residual_codegen.py:3805/4515/6361). It is printed by `_SfemCCodePrinter` (emitters/cprinter.py:23-62), which **takes no target argument**: both targets print identical expression text.
- **One mismatch in builder structure.** The CUDA text is emitted target-driven, not rewritten from the CPU text, but by parallel builder classes: `OpenMPEnergySoASourceBuilder` vs `CUDAEnergySoASourceBuilder` (emitters/energy.py:17-129 vs :132-247). backends/cuda.py:142-176 rejects `#pragma omp`, `lane` and `SFEM_INLINE` in device output.

### 3.2 A real pair

`frontend/ops/generated/neohookean_ogden/d3/neohookean_ogden_d3_tensor_product_local.hpp` (557 lines) vs `cuda/…_local.cuh` (544 lines): the raw diff is 443 lines.

Differences:
- `template <s_t, NQ, NS, VS>` + `static SFEM_INLINE` vs `template <s_t, NQ, NS>` + `static __host__ __device__ __forceinline__` (hpp:33-34 / cuh:30-31)
- a leading `ne` argument
- `[9*NQ*VS]` vs `[9*NQ]` buffers
- `tensor_gradient<…,VS,…>(ne, …)` vs `tensor_gradient_scalar<…>(…)`
- `#pragma omp simd for (lane…)` vs `{`
- `x[k*VS+lane]` vs `x[k]`.

After about 20 mechanical substitutions:
- the diff is **0 lines** for this file and for every one of the 48 energy-family `*_local`/`*_hessian` files (laplace, linear_elasticity, modified_mooney_rivlin, mooney_rivlin_prony, neohookean_ogden, saint_venant_kirchhoff; about 27k CPU lines)
- the 62 residual-family files keep only lane-structure differences
- all 17,164 CSE-temporary lines across the 110 pairs are identical with `lane`→`0`.

### 3.3 Support headers

| Pair | hpp / cuh lines | Generator | Shared after qualifier substitution |
| --- | --- | --- | --- |
| `kernel_math` | 242 / 238 | cprinter.py `_sfem_math_header_source` :193-228 | All but 4 lines. `sfem_log1p` already branches on `__CUDACC__` inside |
| `geometry_kernels` | 227 / 147 | tensor_product_geometry.py :136-219 | The scalar section (hpp:152-227 = cuh:72-147) diffs to 0. hpp:76-151 is the VS-blocked part, host only |
| `tensor_product_kernels` | 2860 / 1485 | tensor_product_kernels.py :289-306, templates :402/:731/:1201/:1376 | The cuh is a strict subset. The CPU renders both blocked and `_scalar` spellings (`_RENDERED_SPELLINGS`, :234), CUDA only `_scalar` |
| `kernel_diagnostics` | 255 / 281 | energy_codegen.py :10492, residual_codegen.py :412 | Struct fields are named differently (`*_per_qp_lane` vs `*_per_qp_scalar`). The cuh adds `SFEM_CODEGEN_DEVICE_INLINE` and `launch_status()` |

### 3.4 Macro inconsistencies in generated output

Counted over all generated `.hpp/.cuh/.cpp/.cu` files:

| Definition | Count |
| --- | --- |
| `#define RSTR SFEM_RESTRICT` | 315 |
| `#define SFEM_RESTRICT` (empty) | 210 (112 hpp, **62 cuh**) |
| `#define SFEM_RESTRICT __restrict__` | 105 |
| `#define SFEM_INLINE inline` | 133 |

- All of these are `#ifndef`-guarded, so the value is set by include order.
- Empty restrict comes from `geometry_kernels.hpp:11`, `tensor_product_kernels.hpp:12` and every CPU local header. `__restrict__` comes from `kernel_math.hpp:11`, `packed_thread_scratch.hpp:7` and the `*_c_abi.hpp` files.
- In tree, the local headers include `sfem_base.hpp` first via `__has_include` (e.g. neohookean_ogden_d3_tensor_product_local.hpp:5-8), so they get `__restrict__`. But `geometry_kernels.hpp` and `tensor_product_kernels.hpp` do not include sfem_base. A TU that includes one of them first gets an empty `SFEM_RESTRICT`, then a redefinition when sfem_base.hpp arrives.

Sources in the generator:
- `plans/conventions.py` `restrict_prelude(definition="__restrict__")` :218-235.
- Callers that pass `""`: energy.py:39, and residual_codegen.py:2577 and :6880, the latter two **unconditionally, also for CUDA** (the 62 `.cuh` files).
- `targets.py` OpenMP `restrict_definition` (inherited empty) feeds tensor_product_geometry.py:154 and tensor_product_kernels.py:325.

Other defects:
- 89 `.cuh` files carry `_HPP` include guards: quadrature_codegen.py:59 (reference/cuda shares the guard with the `.hpp`) and residual_codegen.py:2566.
- Residual-family device functions are non-static (`_function_qualifier`, residual_codegen.py:361, :2842): 238 non-static vs 800 static occurrences in `.cuh`.
- `SFEM_HOST_DEVICE` has 0 uses in the generated tree and in `python/codegen`. The only conditional qualifier machinery is `SFEM_CODEGEN_DEVICE_INLINE`/`HOST_INLINE`, emitted by `conventions.inline_qualifier_prelude` (:191-213) into one file.

### 3.5 Feasibility

**Route 1: one qualifier and one header for the scalar spellings. Feasible now, low risk.**

1. Both targets emit `static SFEM_INLINE SFEM_HOST_DEVICE` for kernel-callable functions. `__forceinline__` is kept where the device needs it, via a `SFEM_FORCE_INLINE` that is also `always_inline` on device. The change is in `TargetPlatform.function_qualifier`/`inline_qualifier` and `conventions.QUALIFIERS`. Generated headers stop defining `SFEM_INLINE`/`SFEM_RESTRICT` themselves and include `sfem_base.hpp` unconditionally. They are never compiled standalone in tree; if standalone compilation must stay, keep a single fallback block that defers to `__restrict__`, never empty.
2. Fix the residual emitter's hard-coded `restrict_prelude("")`, the `_HPP` guards on `.cuh`, and the non-static residual qualifier.
3. Support headers become one file each:
   - `kernel_math.hpp`
   - `geometry_kernels.hpp`, the scalar part, `SFEM_HOST_DEVICE`
   - `tensor_product_kernels.hpp`, the scalar part, `SFEM_HOST_DEVICE`
   - plus host-only `geometry_kernels_lanes.hpp` and `tensor_product_kernels_lanes.hpp` holding the VS-blocked templates and the `#pragma omp simd`, so that nvcc never sees OpenMP pragmas in a header it compiles.
   The `.cuh` support headers are deleted. The `kernel_diagnostics` field names are unified (one struct, `*_per_qp`).
4. `*_local`/`*_hessian`/`*_element`: one header per kernel holding the host lane spelling `name<…,VS>` and the host-device scalar spelling `name_scalar<…>`. This is the same `_RENDERED_SPELLINGS` mechanism that `tensor_product_kernels` already uses. The 166 paired `.cuh` files (about 98k lines) disappear. The total generated text shrinks less, because the scalar half moves into the `.hpp`. What is removed is the file duplication, the macro drift, and the parallel CUDA builder classes in `emitters/energy.py`.

**Route 2: per-qp math emitted once, only the lane or thread loop target-specific. Feasible to emit; performance not proven.**

- The scalar per-qp body becomes a `SFEM_HOST_DEVICE` function. The CPU calls it once per lane inside the `omp simd` lane loop; CUDA calls it once per thread.
- The risk is CPU vectorisation:
  - The current CPU code works on lane-innermost SoA buffers (`x[k*VS+lane]`), which gcc on Grace vectorises only with the lane loop innermost (standing rule).
  - A per-lane call on gathered scalars changes the access pattern. It may need `#pragma omp declare simd` on the body, and may leave gcc scalar.
- It is a measurement, not a design decision: implement it for one material/element (neohookean_ogden HEX8 tensor-product, CPU packed and standard). Measure on Grace with gcc and clang at 72 threads. Adopt only with no regression (no-perf-regressions-for-clean-code).
- If adopted, the qp arithmetic appears once in the generated tree. The upper bound on removed lines is about the size of today's `.cuh` local/hessian/element trees, 98k lines.

- **Out of scope for both routes:** the mesh-level `*_operator.cpp` vs `*_operator.cu` (286 files / 199k lines vs 175 files / 124k lines). They are launchers: OpenMP loops, packed and two-pass variants vs grid-stride kernels and launches. The device keeps its own variant.
- **Rule for every step:** every change is made in `python/codegen/framework` and the tree is regenerated. The generated text is never edited.

## 4. Step order and verification

Order: smallest and most shared first. Each step is one commit and passes its gate before the next starts.

| Step | Content | Lines removed (estimate) |
| --- | --- | --- |
| 0 | Macros: `SFEM_DEVICE_CODE`, `sfem_atomic_add`, the sfem_unroll.hpp branch order, `#ifndef` guards (or the generator change of step G1), the stale-comment decision | – |
| 1 | Index helpers → smesh only (`sshex8_{nxe,txe,lidx}`, `sstet4_{nxe,txe}`), delete `smesh_sshex.cuh` | ~330 |
| 2 | `operators/sfem_crs_kernels.hpp`: `linear_search`/`find_col`/`find_cols`/`local_to_global` for N nodes, host-device binary search, `sfem_atomic_add` | ~250 (21 definitions → 4) |
| 3 | Geometry → smesh: `hex8_fff`, `hex8_adjugate_and_det`, `tet4_fff`, `tet4_adjugate_and_det`, `hex8_sub_adj_0`, `hex8_sub_fff_0`; plus `sstet4_sub_fff_{0..5}` and macro `tet4_sub_fff_0` in `<elem>_kernels.hpp` | ~600 |
| 4 | `hex8_ref_shape_grad{,_x,_y,_z}` | ~140 |
| 5 | Laplacian: hex8 rows 1-6, tet4 rows 25-26, tet10 row 28, spectral row 32, lagrange rows 29-31 | ~700 |
| 6 | Elasticity: rows 14-15 (A) at once. Rows 16/27 (apply_adj) and 17 (whole matrix vs blocks) after the ptxas and timing comparison decides which shapes stay | ~350 certain; up to ~3,000 more if one matrix form serves both |
| 7 | quadshell4 row 40, macro tet4 row 39 | ~60 |
| G1 | Generator: one qualifier, restrict prelude, guards, residual `static` | – (correctness) |
| G2 | Generator: single support headers (scalar host-device + host-only lanes) | ~1,900 (the `.cuh` support headers) |
| G3 | Generator: scalar spelling in the same header as the lane spelling, `.cuh` local/hessian/element trees deleted, CUDA builder classes merged | ~98k files' worth moved; net generated lines change little; generator code shrinks |
| G4 | Experiment: Route 2 on one kernel; extend only if Grace and GH200 show no regression | up to ~98k generated lines |

Handwritten total: about 2.4k lines certain (the 48 element-level pairs plus smesh and lookup copies); up to about 5.5k if step 6 unifies the elasticity matrix.

Verification for every handwritten step:

1. **Host bitwise.** The affected operators' apply/diag/assembly output is checksummed single-threaded by the old-tree build and the new-tree build (old-tree build per the prove-by-running-the-prior-commit rule; single-threaded per the fixed-order rule for atomics).
   - Identical for A/C rows.
   - B/D rows: the stated tolerance.
   - Plus the existing ctests that touch the operator (`sfem_PackedLaplacianTest`, `sfem_PoissonTest`, `sfem_ElasticityCaseTest`, `sfem_SSHex8*`, `sfem_SSTet4*`, `sfem_MultiBlockOpTest`).
2. **Host-device agreement.** A new `sfem_MicroKernelParityTest` (`.cu`, CUDA builds only) instantiates every shared function on the host and in a one-thread kernel, on random and on distorted-element inputs, and compares with a stated relative tolerance (`-use_fast_math` applies on the device). The instantiation also catches host-only calls in host-device bodies.
3. **Device registers.** ptxas `-v` register and spill lines of the touched kernels are diffed against the previous build.
4. **Timings unchanged.**
   - `drivers/bench/bench_op.exe.cpp` (prints `#elements`, `#nodes`, rate in MDOF/s for `Laplacian`, `PackedLaplacian`, `em:Laplacian`, `LinearElasticity`, `em:LinearElasticity`, ... at `SFEM_BASE_RESOLUTION`, `SFEM_ELEMENT_REFINE_LEVEL`, `SFEM_USE_PACKED`).
   - Grace: 72 threads, `OMP_PROC_BIND=true`. GH200: one GPU, `SFEM_EXECUTION_SPACE=device`.
   - Packed and standard side by side, at a size past saturation; element and dof counts with every number.
   - Accept when the difference is within the run-to-run spread.
5. **CUDA build and ctest on Alps** (the local build has no CUDA): build on the login node, then `ctest -R "Cuda|Playground|BSRTangent|MGSDF"` on debug.

Verification for generator steps:

1. The `python/codegen/framework` tests.
2. Regenerate everything and commit the generated diff.
3. The normalised-diff check of section 3.2 becomes a generator unit test while both spellings exist; it should be 0 for all 110 pairs.
4. The host ctest locally.
5. The CUDA ctest subset on Alps (op wrappers change in both trees).
6. Timings as above, plus `GeneratedNeoHookeanOgden` and the generated `laplace`/`linear_elasticity` packed and standard.
