# Paper to code: compressed partial assembly

Each section and paragraph of `main.tex`, with the code that implements or produces it. Line numbers
are at sfem **a70598f39** (main, 2026-11-17); paths are relative to sfem's root. `b60449f5f:<path>` is
the retired spike (`spikes/inexact_apply_compare/qp/`, deleted from main in 09f63d7a6; read it with
`git show b60449f5f:<path>`). The paper's measured numbers were taken with the spike; the library
(`src/frontend/ops/inexact/`) is its successor, and where the two differ the entry says so.

## Where the paper and today's code differ

| Paper | Code at a70598f39 |
|---|---|
| Which point to drop (main.tex:502): the second point of each line on either element | HEX8 drops the **first** (`DROP = 0`, `src/frontend/ops/generated/reference/inexact_hex8_q8.hpp:348-349`); HEX27 the second (`inexact_hex27_q64.hpp:753-754`). Rule: smallest largest weight, ties to the lowest index (`python/codegen/framework/plans/inexact_element.py:436-450`). Same numbers on HEX8 (weight 1 either way). |
| Encodings: int8 per element, int8 per component, fp16 mean and deviation | Not offered by the library (`src/frontend/ops/inexact/sfem_InexactOptions.hpp:37-38`, `make_points` at `sfem_InexactStore.hpp:116-134`: f64, f32, f16, i8md). Per component is spike only (`b60449f5f:spikes/inexact_apply_compare/qp/qp_store.hpp:9,35,171-179`). |
| fp16 scale "just below the top of the half-precision range" | The largest value maps to at most 2^14, two binades below fp16's maximum (`src/frontend/ops/prony/sfem_HistoryStorage.hpp:148-150,162-166`). |
| Full (upper-triangle) element matrix, eq:elementmatrix stored densely | Library has only the difference form; the full form is spike only (`b60449f5f:.../qp/ea_store.hpp:10,39,54`). |
| Full quantized BSR | Spike only (`b60449f5f:.../qp/bsr_quant.hpp:5,60`); the library offers Difference and Upper (`src/frontend/ops/inexact/sfem_QuantBSR.hpp:22-23,64`). |
| Quantized matrix on the GPU (main.tex:1305): SELL-32-1024, a thread per row, columns in the record | Records in the node graph's CSR order, columns in the CSR index, 4 or 8 threads per row, transposed lower records, an f16 scale per window (`src/frontend/ops/inexact/cuda/sfem_cuda_QuantBSRStore.cu:109-110,171-176,201-214,321`; changed in a3147b0cb). SELL is spike only (`b60449f5f:.../qp/cuda/qbsr_kernels.hpp:4-5,34,54-79`). |
| A thread per point (HEX27, main.tex:1290): gradients by sweeps shared between threads | Each thread computes its own gradient from the 1D tables over all 27 nodes; only the transposed sweeps are shared (`sfem_cuda_InexactStore.cuh:422-452`, `:470-497`). |
| A thread per element (main.tex:1283): block-local scatter for TET4 and half-precision HEX8 | Also for the TET10 point store in half precision (`sfem_cuda_InexactStore.cuh:39-51`, `:46-47`). |
| GPU stores keep the host layouts (main.tex:1278) | HEX27 device stores are element-major (`POINT_MAJOR` `sfem_cuda_InexactStore.cuh:812`, `ELEMENT_MAJOR` `:894`). |
| Data layout: 45 arrays with a padded stride, or blocked | Library: blocked only (`sfem_PointStore.hpp:308-319`); the 45-array layout is spike only (`b60449f5f:.../qp/qp_store.hpp:36,86,105,115`). |
| Arithmetic precision: τ independent of the storage type, both measurable | The library fixes τ per encoding: float for f32/f16/i8md/i8pair, double for f64 (`sfem_InexactStore.hpp:120-127,140-148`). |
| Error model (eq:eps, eq:codecerror) | No code computes ε or σ/μ; `bench_inexact` reports the relative l2 distance (`drivers/bench/bench_inexact.exe.cpp:120`). The spike measured the energy error (`b60449f5f:.../qp/bench_qp.cpp:7,14-15,217`). |
| Newton forcing η+ε | η is a fixed `rtol` (`src/frontend/nonlinear/sfem_UnconstrainedStep.hpp:26`); nothing ties it to ε. |
| Quantized BSR "set-up and memory" (main.tex:894): f32 values stay allocated | As the text says on the host: the store keeps the f32 assembly (`sfem_QuantBSRStore.hpp:31,101-106,121`), 374 B/dof on HEX8 against the spike's 92. |
| Timing protocol: best of three averages of ten applications | Spike: `b60449f5f:.../qp/bench_problem.inc:59-69`, `bench_qp.cpp:201`. `bench_inexact` averages one run of `repeats` (default 50; `:92-99`, `:165`). |
| Results tables and figures | Generated from the spike's logs by `results_tables.py`/`paper_figures.py`; today's `bench_inexact` and playground print other formats (`bench_inexact.exe.cpp:205-216`; `src/frontend/mech/sfem_Playground.cpp:1508`, `sfem_Observers.cpp:850`), so the generators cannot read them as they are. |

---

## Introduction (main.tex:127-197)
Motivates the formats (BSR, element matrix, matrix-free, partial assembly), the exact and quantized reductions, and the η+ε result.
- `BSRTangent` — `src/frontend/nonlinear/sfem_BSRTangent.hpp:27`: the assembled BSR reference (`storage_bytes` 8/4/2 at `:22-26`).
- `create_linear_operator` — `src/frontend/sfem_LinearOperatorFactory.cpp:480,484-485,489`: matrix-free against the stored-tangent (INEXACT) operator.
- `InexactOptions` — `src/frontend/ops/inexact/sfem_InexactOptions.hpp:39`: the stored formats and encodings (doc `:18-38`).
- `LinearSolverOptions::rtol` — `src/frontend/nonlinear/sfem_UnconstrainedStep.hpp:26`: the Krylov forcing η (1e-3).
- `\exactQpPct…` — `spikes/inexact_apply_compare/qp/cost_model.py:513`.

## Background and related work (main.tex:200-261)
- Partial assembly and quadrature-point data (203-222): no code (background).
- Reduced-precision storage (224-247): no code; the choices it motivates are under Encodings.
- Newton with an approximate Jacobian (249-261): no code (background).

## Operator formats (main.tex:264-298)
Four formats share one tangent; exact reductions first, then encodings.
- Matrix-free: `create_parallel_matrix_free_operator`, returned at `src/frontend/sfem_LinearOperatorFactory.cpp:484-485`.
- `InexactOptions::Store {AUTO, POINTS, ELEMENT_MATRIX, BSR}` — `sfem_InexactOptions.hpp:40`; `Encoding` `:41`.
- `inexact::create_store<M>` — `src/frontend/ops/inexact/sfem_InexactStore.hpp:158`: point or element-matrix store, folded or not.
- `OpState::update_bsr` — `sfem_InexactStore.hpp:205`: the assembled (bsr) store.
- `InexactOptions::validate` / `resolve` / `from_strings` — `src/frontend/ops/inexact/sfem_InexactOptions.cpp:104,134,171`; `resolve` called from `Function::set_inexact_options`, `src/frontend/sfem_Function.cpp:583,593`.
- Playground: `inexact_options_from_yaml` — `sfem_InexactOptions.cpp:185` (keys `solver.linear.inexact.store/encoding/fold/traversal`, `:185-196`), called at `src/frontend/mech/sfem_Playground.cpp:1077`; allowed keys `src/frontend/mech/sfem_ElasticityCase.cpp:127-130`; documented at `docs/playground/case-reference.md:290-293`.
- Set-up once per Newton step: `UnconstrainedStep::linearize`, INEXACT branch — `src/frontend/nonlinear/sfem_UnconstrainedStep.cpp:157-172` (`inexact_update` `:169`).
- All formats benchmarked: `drivers/bench/bench_inexact.exe.cpp:243-246` (format list), loop `:248-261`.

## Quadrature-point stores (main.tex:301-302)

### The tangent action at the quadrature points (main.tex:304-339)
eq:pullback (weighted pulled-back tangent), eq:action, major symmetry (45 against 81 numbers).
- eq:pullback, symbolic: `point_tangent` — `python/codegen/framework/plans/point_tangent.py:207` (Σ A·adj·adj/det, `:250-257`; formula in the docstring `:7`).
- Emitted point function, e.g. `neohookean_ogden_hex8_inexact::point` — `src/frontend/ops/generated/neohookean_ogden/d3/hex8/neohookean_ogden_hex8_inexact.hpp:28` (`NC = 45` `:16`), emitted by `_point_lines`, `python/codegen/framework/emitters/inexact_codegen.py:873`.
- w_q folded in at set-up: `setup_points` — `src/frontend/ops/inexact/sfem_InexactSetup.hpp:103` (`w = QW[q]` `:178`, `M::point` `:205`, accumulate `:207`; `adjugate_and_determinant` `:18`).
- eq:action: `PointStore::block` — `src/frontend/ops/inexact/sfem_PointStore.hpp:212` (`stage_in` `:219`, `contract_lanes` `:283-284`, `stage_out` `:287`); no geometry read.
- Generated stages, e.g. `inexact_hex8_q8::stage_in/stage_out` — `src/frontend/ops/generated/reference/inexact_hex8_q8.hpp:28,199` (`DT` `:22`).
- 9×9 contraction: `inexact_contract_d3<true>::contract_lanes` — `src/frontend/ops/generated/reference/inexact_contract_d3.hpp:32` (`SLOT` `:14`), emitted by `contract_header_source`, `inexact_codegen.py:809`.
- 45 against 81: `PointTangentPlan::tangent_index` — `point_tangent.py:55`; `flux_tangent_is_symmetric` `:93`; `packed_slot`/`symmetric_components`/`full_components` — `plans/inexact_element.py:71,60,66`; the 81-number packing `inexact_contract_d3<false>`, `inexact_contract_d3.hpp:389-390`.
- The spatial-frame 21+6 alternative: not implemented (text only).

### What the quadrature can see (main.tex:341-404)
eq:moments, tab:stores (r, dim Π, ρ, the minimum store).
- `cost_model.py` (tracked at HEAD, `spikes/inexact_apply_compare/qp/`): `gradient_span` `:170`, `product_space_dim` `:180`, `evaluation_rank` `:197` (`matrix_rank` `:209`), `moment_store` `:213`, `analyse` `:368` (`assert store == minimal` `:393`), `stores.tex` written `:431-437`; run by `make tables` (`docs/compressed_partial_assembly/Makefile:18-19`).
- The library holds the minimum: TET4 `NV = 1` (`inexact_tet4_q1.hpp:13`); TET10 `NV = 10` against `NQ = 11` (`inexact_tet10_q11.hpp:13-14`); HEX8 `NS = 288` (`inexact_hex8_q8.hpp:356`); HEX27 `NS = 2592` (`inexact_hex27_q64.hpp:761`); selection `inexact_element_plan`, `plans/inexact_element.py:545`.
- Negative weight of the 11-point rule: `QW[0]`, `inexact_tet10_q11.hpp:23`.

### Moments on quadratic tetrahedra (main.tex:406-441)
eq:tet10moments; vertex derivatives, 16 block contractions, transposed expansion.
- Plan: `_moments_plan` — `plans/inexact_element.py:349` (vertex derivatives `:356-371`, pairs `:375`, λ_aλ_b `:376-381`, multiplicity `:398`; `_barycentric` `:341`).
- Tables: `inexact_tet10_q11` — `src/frontend/ops/generated/reference/inexact_tet10_q11.hpp:8` (`moments` `:15`, `DT` `:20`, `PA/PB/MULT` `:24-26`, `FACTOR` `:27`).
- Set-up: `PointStore::accumulate` — `sfem_PointStore.hpp:82` (`:84`); moments zeroed `sfem_InexactSetup.hpp:141-146`.
- Schedule: `stage_in` `inexact_tet10_q11.hpp:32`, `stage_out` `:107`; each moment decoded once, applied to (a,b) and (b,a), `sfem_PointStore.hpp:283-284`; `ElementTables::pa/pb`, `src/frontend/ops/inexact/sfem_InexactElement.hpp:38,45`.
- Emitter: `_simplex_stage_lines` — `inexact_codegen.py:182`.

### The element mean and the deviations (main.tex:443-454)
- `PointStore::encode_block` — `sfem_PointStore.hpp:91`: mean `:100-105` (stored f32), deviation `T -= X*mean` `:106`.
- `ElementTables::mult` — `sfem_InexactElement.hpp:53` (multiplicity 2 for a≠b).
- X: point stores X = w_q (`plans/inexact_element.py:519,330`); TET10 X_ab Gram (`reference_weight` `:382-385`); emitted `X[NV]` (`inexact_tet10_q11.hpp:22`).
- |K̂|: `KVOL` (`inexact_tet10_q11.hpp:18`, `inexact_hex8_q8.hpp:18`). Means stored: `means_`, `sfem_PointStore.hpp:62`.

### The fold on the hexahedra (main.tex:456-500)
eq:fold; a point keeps 45−6d_q numbers; 9 fewer multiply-adds per dropped direction.
- Plan: `_fold` — `plans/inexact_element.py:430` (masks, sources, coefficients `:459-474`); `_lagrange_weight` `:421`; `slot_direction` `:87`; `fold_layout` `:230` (keep `:244-251`, offsets `:252-254`, folded weights `:255-265`); `line_coefficients` `:218`; `Fold` dataclass `:128`.
- Emitted: `_fold_lines`/`_fold_struct_lines` — `inexact_codegen.py:604,574`. HEX8: `Q1/DROP/COEF1` `inexact_hex8_q8.hpp:347-349`, `fold_symmetric` `:350`, `NKEEP = {45,39,39,33,39,33,33,27}` `:353`, `NS = 288` `:356`. HEX27: `inexact_hex27_q64.hpp:752-761`.
- `Fold<El,Tables>` — `sfem_InexactElement.hpp:97` (`mask` `:110`, `source` `:115`, `coefficient` `:118`; `FoldOf` `:127-141`).
- At set-up: `PointStore::fold` — `sfem_PointStore.hpp:145` (`:150`); kept slots only `:122-125` (`fold_index` `:312`).
- Masked application: `sfem_PointStore.hpp:243-265` → `contract_lanes_masked`, `inexact_contract_d3.hpp:374` (per mask `:73`); emitter `_scalar_masked_contraction_lines`, `inexact_codegen.py:779`.
- Option `fold auto|on|off` — `sfem_InexactStore.hpp:167-176`. Device: HEX8 only, reason at `src/frontend/ops/inexact/cuda/sfem_cuda_InexactStore.cuh:11,17-18`.
- Counts in the paper: `fold(name)` — `cost_model.py:265`.

#### Which point to drop (main.tex:502-508)
- Rule: `_fold` — `plans/inexact_element.py:436` (docstring), `:440-450` (search; module docstring `:31-32`).
- HEX27: `DROP = 1`, `COEF1 = {0.3948, 0, 1, -0.3948}` — `inexact_hex27_q64.hpp:753-754` (as the paper).
- HEX8: `DROP = 0` — `inexact_hex8_q8.hpp:348-349`, `inexact_proteus_hex8_q8.hpp:346-347` (**the first point**, see the table above).
- Spike original: `b60449f5f:spikes/inexact_apply_compare/qp/make_qp_kernels.py:454` (`fold_tables`, search `:467-473`).

## Encodings and their error (main.tex:511-512)

### Encodings (main.tex:514-562)
- Fused encode: `setup_points` → `store.encode_block` — `sfem_InexactSetup.hpp:211`; `PointStore::encode_block` `sfem_PointStore.hpp:91,129-139`.
- `Scheme {Plain, MeanDeviation}` — `src/frontend/ops/inexact/sfem_InexactCodecs.hpp:22`; `CodeTraits<half_t>`/`<int8_t>` `:34-46`.
- fp16 scaled: `ScaledHalfCodec` — `src/frontend/ops/prony/sfem_HistoryStorage.hpp:152` (`scale_for` `:162-166`).
- int8: `ScaledIntegerCodec<T,127>` — `sfem_HistoryStorage.hpp:171` (`scale_for` `:193-196`, `encode` `:181,188-191`); `Int8Codec` `:217`.
- Scale on the outputs: `PointStore::scale_on_output` — `sfem_PointStore.hpp:49`, applied `:289-295`.
- int8 per element: not offered (`sfem_InexactOptions.hpp:37-38`, `sfem_InexactStore.hpp:117-134`); spike `b60449f5f:.../qp/qp_store.hpp:59`.
- int8 per component: spike only, `b60449f5f:.../qp/qp_store.hpp:9,35,171`.
- int8 mean and deviation (`i8md`): `PointStore<M,int8_t,MeanDeviation>` — `sfem_InexactStore.hpp:126-128`; encoding `sfem_PointStore.hpp:100-108,129-139`.
- fp16 mean and deviation: not offered; spike `--qp f16md` (`b60449f5f:.../qp/newton_qp.exe.cpp:3`); the history store has `Float16MeanDeviationCodec`, `sfem_HistoryStorage.hpp:218`.
- Shared with the history variables: `sfem_InexactCodecs.hpp:8,24-26`; `MeanDeviationCodec`, `sfem_HistoryStorage.hpp:208-219`.
- Encoding names: `encoding_from_string` — `sfem_InexactOptions.cpp:17`. Bytes in the paper: `codecs` — `cost_model.py:353`.

#### The fold and the encodings (main.tex:564-571)
- Order: mean (`sfem_PointStore.hpp:100-108`), `fold` (`:111`), scale from the kept values (`:112-121`); comment `:144`.
- Folded weights `XC`: `folded_weight`, `plans/inexact_element.py:256-265`; emitted `inexact_hex8_q8.hpp:358`; used `sfem_PointStore.hpp:253,258,171`.

### Error model (main.tex:573-615)
- No code computes ε or σ/μ. Nearest: `relative_difference` — `drivers/bench/bench_inexact.exe.cpp:120`.
- Spike energy error and ν sweep (`QP_NU`): `b60449f5f:.../qp/bench_qp.cpp:7,14-15,217`.
- σ per codec = the scale in `encode_block` (`sfem_PointStore.hpp:131-133`).

#### Rotations (main.tex:617-635)
- No code (analysis); its consequence, translations kept exactly through differences, is the difference form below.

#### Mean and deviation in the application (main.tex:637-640)
- Per-block means and scales: `PointStore::block` — `sfem_PointStore.hpp:229-239`; decode `S = xj*A + sc*v` unfolded `:276-278`, folded `:257-258`; host reference `PointStore::decode` `:160` (`:164-166`).
- Device: `PointCodes<…>` — `sfem_cuda_InexactStore.cuh:178`; `i8md` store `:983-984`.

## Effect on the Newton iteration (main.tex:643-687)
- η fixed: `LinearSolverOptions::rtol` (`sfem_UnconstrainedStep.hpp:26`), `set_rtol` at `sfem_UnconstrainedStep.cpp:45,51,226`.
- Newton loop: `NewtonSolver::solve` — `src/frontend/nonlinear/sfem_NewtonSolver.cpp:111` (`iterate` `:175-210`, `linearize` `:197`, solve `:203`); nonlinear rtol `sfem_NewtonSolver.hpp:41`.
- Encoded tangent in the Krylov solve: `UnconstrainedStep::linearize` — `sfem_UnconstrainedStep.cpp:146,157-172`; Jacobi diagonal `inexact_diag` `:178-181`; CG refused for an unsymmetric material `:162-167`.
- Newton study driver: spike `b60449f5f:.../qp/newton_qp.exe.cpp:1-9`; library: the playground with `solver.linear.format: INEXACT` (`sfem_Playground.cpp:1077`).

## Element matrices (main.tex:690-710)
eq:elementmatrix.
- `ElementMatrixStore::encode_block` — `src/frontend/ops/inexact/sfem_ElementMatrixStore.hpp:96` (both passes for a≠b `:106-108`; T = S·DT `:109-120`; K += DT·T `:121-133`; formula `:22`). Fed by `setup_points` (`sfem_InexactSetup.hpp:211`); `accumulate` `:84`.
- Full upper-triangle storage: spike only, `b60449f5f:.../qp/ea_store.hpp:10,39,54,266`.
- tab:elementmatrix: `cost_model.py:423`.

#### The difference form (main.tex:712-741)
- `ElementMatrixStore` — `sfem_ElementMatrixStore.hpp:33` (doc `:17-31`; `NB` `:48-50`); `NodePairs<N>`, `sfem_InexactElement.hpp:21`; `block_nodes` `sfem_ElementMatrixStore.hpp:75-79`.
- eq:difference: `ElementMatrixStore::block` — `:174` (differences `:188-191`; y_p += K d `:201-203`; y_q −= Kᵀd `:204-208`).
- Implied diagonal: `diagonal_block` — `:149-170`.
- Differences first in the point stages: `inexact_tet10_q11.hpp:29-30`. Device: `MatrixStore`, `sfem_cuda_InexactStore.cuh:891`.

#### A scale per node pair (main.tex:743-761)
eq:pairscale, E5M3 factor.
- `PairFactor::value` — `sfem_InexactCodecs.hpp:54`; `encode` `:62` (rounding up `:67-73`; doc `:48-53`).
- `encode_values` — `sfem_ElementMatrixStore.hpp:238` (s_e `:242-247`; ratio and factor `:248-259`).
- Apply: factor on the differences `:192-198`, s_e on the outputs `:217`; factors `[block][pair][lane]` (`factors_` `:59,226`).
- Option `i8pair` — `sfem_InexactStore.hpp:146-148`. Device: `sfem_cuda_InexactStore.cuh:626-628`, apply `:664,719`. Spike: `b60449f5f:.../qp/ea_store.hpp:15-17,72,79`.

## Quantized assembled matrices (main.tex:807-827)
- Single-precision assembly then encode: `BsrStoreImpl::update` — `src/frontend/ops/inexact/sfem_QuantBSRStore.hpp:34` (`SMESH_FLOAT32` `:43`, `encode_in_place` `:51`); `AssembleBSR` — `src/frontend/ops/inexact/sfem_InexactInterface.hpp:67`; supplied by e.g. `GeneratedNeoHookeanOgden::inexact_update` → `hessian_bsr_typed`, `src/frontend/ops/generated/neohookean_ogden/op/sfem_GeneratedNeoHookeanOgden.cpp:1348-1353`.
- Codes: `QuantBsr::encode_into` `put` — `sfem_QuantBSR.hpp:275-286`; f16 matrix scale `:264-272`.
- `create_bsr_store` — `sfem_QuantBSRStore.hpp:173`. Format overview: `sfem_QuantBSR.hpp:1-41`.

#### Difference form and symmetry (main.tex:829-841)
- `BsrForm {Difference, Upper}` — `sfem_QuantBSR.hpp:64` (doc `:5-20`; no full form `:22-23`).
- Upper stores j>i: `in_sell` `:106-108`, `stored_blocks` `:109-118`, encoded `:319-320`; symmetrisation for the two-sided form `:321-327`; unsymmetric case `sfem_QuantBSRStore.hpp:184-188`.
- Implied diagonal: `QuantBsr::add_diagonal` — `sfem_QuantBSR.hpp:373-401`.
- Rows that do not sum to zero (inertia): exact f32 correction, `BsrStoreImpl::correction` — `sfem_QuantBSRStore.hpp:127` (applied `:86-96`); not in the paper.
- Full form: spike only, `b60449f5f:.../qp/bsr_quant.hpp:5,60`. Device: transposed records, `transpose_kernel` `sfem_cuda_QuantBSRStore.cu:171-176`.

#### Layout and application, Algorithm alg:qbsr (main.tex:843-892)
- C and σ: `VS = 16` (`sfem_QuantBSRStore.hpp:176`), `SIGMA = 32*VS_` (`sfem_QuantBSR.hpp:69`); 16-bit offsets `col_t = uint16_t` (`sfem_QuantBSRStore.hpp:179,181`), `reach` (`sfem_QuantBSR.hpp:74`), written `:214`.
- Structure: `QuantBsr` constructor — `sfem_QuantBSR.hpp:121` (parts `:128-137`; sort in windows `:143-146`; padding `:150-157`; remainder and halo slots `:162-207`; padding with the own column `:215-217`; slot order `:221-230`).
- alg:qbsr: `QuantBsr::team_apply_upper` — `sfem_QuantBSR.hpp:484` (z ← 0 `:501`; a ← 0 `:513`; d `:522-529`; a, t `:533-538`; y_j `:542-547`; y_i `:549-555`; remainder `:558-574`; barrier `:577`; slots `:578-582`); wrapper `team_apply` `:416-423`; two-sided form `team_apply_difference` `:426`.
- Records `step_codes`/`step_scales` `:234-235`; `R` `:85-86`.
- Device: rows, G threads per row (not SELL) — `apply_kernel`, `sfem_cuda_QuantBSRStore.cu:209-214`.

#### Set-up and memory (main.tex:894-899)
- 13 B (int8) / 18 B (fp16) per block: `CODE_BYTES`/`R` — `sfem_QuantBSR.hpp:85-86` (doc `:36-41`).
- In place: `encode_in_place` — `sfem_QuantBSR.hpp:256` (remainder first `:287-303`; rounds `:344-367`).
- f32 values stay allocated: `values_` — `sfem_QuantBSRStore.hpp:31,121`; `bytes()` `:101-106`.
- Device set-up in windows chosen by memory: `sfem_cuda_QuantBSRStore.cu:299-302,350-352`.

## Kernels on the CPU (main.tex:920-929)
Blocks of V elements, a lane per element, lane loop innermost, reference constants as literals.
- `HOST_VS = 16` — `sfem_InexactStore.hpp:29`.
- `contract_lanes` with `#pragma omp simd` — `inexact_contract_d3.hpp:31-69`.
- Generated constants (`DT`, `X`, `QW`, `PA/PB/MULT/FACTOR`): `inexact_tet10_q11.hpp:18-27`, `inexact_hex8_q8.hpp:18-25`, `inexact_hex27_q64.hpp:18-25`.
- Emitter `element_header_source` — `inexact_codegen.py:668`; plans `_simplex_points_plan` `plans/inexact_element.py:317`, `_moments_plan` `:349`, `_tensor_product_plan` `:483`.
- The quantized matrix's traversal: `QuantBsr::team_apply` `sfem_QuantBSR.hpp:416`, `team_apply_difference` `:426`, `team_apply_upper` `:484`.

#### Notation (main.tex:931-960)
- slot(i,k,m,n): `inexact_contract_d3<true>::SLOT`/`slot()` — `inexact_contract_d3.hpp:14-15` (81 at `:390-393`); `packed_slot`, `plans/inexact_element.py:71`; `detail::packed_slot`, `sfem_InexactSetup.hpp:32-35`.
- Storage types: `Encoding`, `sfem_InexactOptions.hpp:41`; `make_points`, `sfem_InexactStore.hpp:116-134`.
- X_j: `Element::X` (`inexact_tet10_q11.hpp:22`, `inexact_hex8_q8.hpp:24`); X_jα: `fold_symmetric::XC` (`inexact_hex8_q8.hpp:358`, `inexact_hex27_q64.hpp:763`).
- A_j: `Fold::mask/source/coefficient` (`sfem_InexactElement.hpp:96-119`); `KEEP/NKEEP/OFF/NS` (`inexact_hex8_q8.hpp:353-356`).
- eq:decode: `PointStore::block`, `sfem_PointStore.hpp:229-283` (mean case `:258,277`; plain `:260,279`; output scale `:289-295`). Per-component case: spike only, `b60449f5f:.../qp/qp_store.hpp:9,35,70,171-179`.

### Data layout (main.tex:962-974), fig:fmtpoints, fig:fmtfold (976-1006)
- Blocked index: `PointStore::code_index` — `sfem_PointStore.hpp:308-311`; `fold_index` `:312-315`; `mean_index` `:316-319`; buffers `:58-66`.
- Narrow load then widen: `:271-282`; folded `:250-263`.
- 45 arrays with a padded stride: spike only, `b60449f5f:.../qp/qp_store.hpp:36,86,105,115`.
- NS = 288 / 2592: `inexact_hex8_q8.hpp:356`, `inexact_hex27_q64.hpp:761`.
- Figures: TikZ sources `figures/formats/fmt_*.tex` (Makefile pattern rule).

### Set-up, Algorithm alg:setup (main.tex:1008-1063)
- `setup_points<Store,PROJECT>` — `sfem_InexactSetup.hpp:102-214`: blocks of VS `:125,137-139`; per-thread scratch `codegen::thread_scratch` `:127-135`; moments zeroed `:141-146`; gather `El::setup_in` `:148-175`; geometry `:177-193` (`adjugate_and_determinant` `:18-29`); material `M::point` `:205` (e.g. `src/frontend/ops/generated/continued_mooney_rivlin_prony/d3/hex8/continued_mooney_rivlin_prony_hex8_inexact.hpp:28-29`); projection `:206`; weight and accumulate `:178,207`; encode `:211`.
- `PointStore::accumulate` — `sfem_PointStore.hpp:81-88`.
- `PointStore::encode_block` — `:91-141`: mean `:100-108` (computed from the tensors, not inside the q loop; same result); fold `:110-111`, `fold()` `:145-153`; scale and codes `:112-125`; unfolded `:129-139`.
- Scales: `ScaledHalfCodec::scale_for` (`sfem_HistoryStorage.hpp:162-166`), `ScaledIntegerCodec::scale_for` (`:193-196`). Per-component σ_α: spike only.
- Element matrices: `ElementMatrixStore::encode_block` — `sfem_ElementMatrixStore.hpp:95-142` (float work type `:97`; p<q only, `NB` `:49`, `NodePairs` `sfem_InexactElement.hpp:20-32`; `encode_values` `:238-262`, `PairFactor` `sfem_InexactCodecs.hpp:53-76`).
- Device: `setup_kernel` — `src/frontend/ops/inexact/cuda/sfem_cuda_InexactSetup.cuh:409-478`.

### Application schedules and tab:flops (main.tex:1065-1108)
- Only the selected schedule per element is generated: `_simplex_stage_lines` (`inexact_codegen.py:182`), `_linear_hex_stage_lines` (`:234`), `_tensor_stage_lines` (`:363`).
- 81-multiply-add contraction: `contract_header_source` (`inexact_codegen.py:809-856`) → `contract` (`inexact_contract_d3.hpp:17-28`), `contract_lanes` (`:31-69`).
- Direct and modal schedules and their counts: cost model only — `count_direct` `cost_model.py:127`, `count_modal_per_point` `:152`, `count_modal` `:229`, `count_sumfact` `:255`, `fold` `:265`; `analyse` `:368-385`; `flops.tex` `:438`.

#### Algorithm alg:applymoments (main.tex:1110-1147)
- Gather: `team_apply_atomic`, `src/frontend/ops/inexact/sfem_InexactTraversal.hpp:46-54`, or `packed_pack` `:107-120`.
- P = G h then τ: `inexact_tet10::stage_in`, `inexact_tet10_q11.hpp:29-104`.
- Moment loop: `PointStore::block`, `sfem_PointStore.hpp:241-285` (pairs `Tables::pa/pb`, `sfem_InexactElement.hpp:38-51`; decode `:267-282`; (a,b) `:283`, (b,a) `:284`).
- Expansion: `stage_out`, `inexact_tet10_q11.hpp:105-179` (called `sfem_PointStore.hpp:287`); output scale `:289-295`; scatter atomic `sfem_InexactTraversal.hpp:57-65`, packed `:123-127,130-133`.
- TET4: `inexact_tet4_q1.hpp:12-13,27,57`.

#### Algorithm alg:applypoints (main.tex:1149-1187)
- Gradients by sweeps: `inexact_hex8_q8::stage_in`, `inexact_hex8_q8.hpp:27-197` (rounding e.g. `:63-65`).
- Point loop: `sfem_PointStore.hpp:241-285` (folded decode `:243-263`; `contract_lanes_masked` `:264`; unfolded `:267-283`).
- Transposed sweeps: `stage_out` `inexact_hex8_q8.hpp:198-338`; HEX27 `inexact_hex27_q64.hpp:27,483`.

#### Algorithm alg:applyelement (main.tex:1189-1215)
- `ElementMatrixStore::block<tau>` — `sfem_ElementMatrixStore.hpp:173-220` (pairs `:182-210`; differences `:188-191`; factor `:192-198`; K `:199-200`; y_p `:201-203`; y_q `:204-208`; s_e `:211-219`).
- Scatter: `team_apply_packed`, `sfem_InexactTraversal.hpp:151-186`.

#### The folded kernels (main.tex:1217-1228)
- Set-up: `sfem_PointStore.hpp:110-126`, `fold()` `:145-153`.
- Eight masked contractions: `contract_lanes_masked<MASK>`, `inexact_contract_d3.hpp:72-371`; run-time switch `:373-385`, called `sfem_PointStore.hpp:264`; emitter `_lane_contraction_lines`, `inexact_codegen.py:729-777`.
- Derivation: `_fold`, `plans/inexact_element.py:430-482`; tables `_fold_struct_lines` `inexact_codegen.py:574`, `_fold_lines` `:604`. Default on when foldable: `create_store`, `sfem_InexactStore.hpp:167-176`.

### Arithmetic precision (main.tex:1230-1241)
- τ a template parameter: `StoreImpl<Store,tau>` (`sfem_InexactStore.hpp:31-32`), `make<Store,tau>` (`:110-114`), `PointStore::block<tau>` (`sfem_PointStore.hpp:211-215`), `ElementMatrixStore::block<tau>` (`sfem_ElementMatrixStore.hpp:173-174`).
- τ chosen per encoding: `sfem_InexactStore.hpp:120-127,140-148` (see the table above).
- Differences in h_t then cast: `inexact_tet4_q1.hpp:25-26`, `inexact_tet10_q11.hpp:29-30`, `inexact_hex8_q8.hpp:63-65`; element matrices `sfem_ElementMatrixStore.hpp:188-191`.
- Outputs in real_t: `stage_out<tau,real_t>` (`sfem_PointStore.hpp:287`); `bout` (`sfem_InexactTraversal.hpp:55,121`).
- Device: `Code<__half>::tau`, `Code<int8_t>::tau` = float — `sfem_cuda_InexactSetup.cuh:121,132`.

### The packed two-pass traversal, fig:packed (main.tex:1243-1272)
- `team_apply_packed` — `sfem_InexactTraversal.hpp:151-186` (scratch slots `:165-166`; packs `schedule(dynamic,1)` `:167-169`; barrier `:171`; second pass in fixed order `:173-185`).
- `packed_pack` — `:72-134` (node-major gather `:94-99`; blocks of VS with `uint16_t` connectivity `:100-120`; accumulate `:123-127`; owned written `:131`; ghosts parked `:132`).
- Packed or atomic: `StoreImpl::team_apply` (`sfem_InexactStore.hpp:60-75`); `PackedView` (`sfem_InexactInterface.hpp:47-63`); `OpState::packed_view` (`sfem_InexactStore.hpp:259-280`); chosen by `InexactOptions::Traversal`.
- smesh: `PackedMesh` (`external/smesh/src/frontend/smesh_packed_mesh.hpp:12-62`, ghost reduce `:42-49`); `Block::pack` (`smesh_packed_mesh.cpp:300`); owned ranges `:220,292`; `build_ghost_reduce` `:359-399`; max nodes per pack `:403-404`; pack size `SMESH_ELEMENTS_PER_PACK` `:420-421`; `FunctionSpace::PackedIdxType = uint16_t` (`src/frontend/sfem_FunctionSpace.hpp:25-27`).
- Scratch pool `codegen::thread_scratch` — `src/frontend/ops/generated/packed_thread_scratch.hpp:38-42`.
- Operator wiring, e.g. `GeneratedMooneyRivlinProny::team_prepare` (`src/frontend/ops/generated/mooney_rivlin_prony/op/sfem_GeneratedMooneyRivlinProny.cpp:979-992`), `team_inexact_apply` (`:1510-1550`).
- `SFEM_PACKED_TWO_PASS` switches only the exact operators' packed apply (`python/codegen/framework/package/op_wrappers.py:1439`; e.g. `sfem_GeneratedNeoHookeanOgden.cpp:648`).

## Kernels on the GPU (main.tex:1275-1281)
- `THREADS = 128`, `V = 32` — `sfem_cuda_InexactSetup.cuh:36-37`; blocked device index `PointCodes::code_index` (`sfem_cuda_InexactStore.cuh:193-195`), `MatrixCodes::code_index` (`:542-548`); HEX27 element-major (`:812,894`; `point_index` `:196-198`).
- Device set-up: `setup_kernel` (`sfem_cuda_InexactSetup.cuh:409-478`), `element_max` (`:483-499`), `launch_setup_kernel` (`:502-534`); encoders `PointCodes::encode` (`sfem_cuda_InexactStore.cuh:217-276`), `MatrixCodes::encode` (`:560-632`); `Code<>` (`sfem_cuda_InexactSetup.cuh:95-138`).
- Host/device agreement: `src/frontend/tests/cuda/sfem_CudaInexactOperatorTest.cpp:6-7`.

#### A thread per element (main.tex:1283-1288)
- `apply_points_kernel` — `sfem_cuda_InexactStore.cuh:314-384` (folded `:347-360`, `contract_masked` `src/frontend/ops/generated/reference/cuda/inexact_contract_d3.cuh:32`; moments both pairs `:372-373`; scale `:377-381`); `apply_matrices_kernel` `:635-683`.
- `Traversal<El,BLOCK>` — `:101-161` (block-local `:115-154`; atomic `:156-158`); choice `shape<…>()` `:39-51`; `BlockMap::build` `:71-84`; host `block_local_map` (`src/frontend/ops/inexact/sfem_InexactBlockMap.hpp:16-27`, `.cpp:8`); fold on the device for HEX8 only (`create_store` `:1013-1032`, `:1024`).

#### A thread per point (main.tex:1290-1299)
- `apply_points27_kernel` — `sfem_cuda_InexactStore.cuh:386-512` (`EPB = 2` `:400`; increment in shared memory `:414-420`; per-thread gradient `:422-452`; decode and contract `:453-467`; shared transposed sweeps `:470-497`; atomics `:499-511`); launch `:848-850`; line tables `inexact_hex27_q64.cuh:765-767`; no fold `:187`, `:1024-1029`.
- Spike original: `b60449f5f:spikes/inexact_apply_compare/qp/cuda/qp_device.cu:1007-1127`.

#### A warp per element (main.tex:1301-1303)
- `apply_matrices27_kernel` — `sfem_cuda_InexactStore.cuh:685-738` (shared h `:697-707`; row p `:709-733`; pair factor `:718-721`; atomics `:734-737`); launch `:929-930`; `MatrixCodes` `:520-555`.
- Spike original: `b60449f5f:.../qp/cuda/qp_device.cu:1218`.

#### The quantized matrix (main.tex:1305-1311)
- Library (differs from the text): `sfem_cuda_QuantBSRStore.cu` — records `RecI8`/`RecF16` `:22-33` (columns in CSR `ci`, `:340-341`); full difference form, transposed lower records `transpose_kernel` `:171-199`, `encode_kernel` `:145-168`; layout comment `:201-207`; `apply_kernel<I8,G,FUSED>` `:209-267` (shuffle reduction `:239-246`; rows written once, no atomics `:247-266`); `g_ = mean <= 16 ? 4 : 8` `:321`; launch `:436-446`; interface `sfem_cuda_QuantBSRStore.hpp:12-17`.
- SELL-32-1024, as the paper: spike only — `b60449f5f:.../qp/cuda/qbsr_kernels.hpp:4-5,34,54-79,87`; `qbsr_kernels.cu:93-148,150-152,261`; choice `qp_device.cu:1385-1394`, encode `:1450`; variant sweep `qbsr_shape.cpp:567-571`.

#### BSR, fig:gpu (main.tex:1313-1328)
- Device assembly (generated), e.g. `mooney_rivlin_prony_tet10_hessian_i_msoa_assemble_impl` — `src/frontend/ops/generated/mooney_rivlin_prony/d3/tet10/cuda/mooney_rivlin_prony_tet10_operator.cu:1445` (entry `:1528`, PSD `:1591`); helpers `src/frontend/ops/cuda/sfem_cuda_CooperativeAssembly.cuh`.
- Dispatch `GPUGeneratedMooneyRivlinProny::hessian_bsr_typed` — `src/frontend/ops/generated/mooney_rivlin_prony/op/cuda/sfem_GeneratedMooneyRivlinProny_cuda.cpp:1030-1092`; into the store `OpState::update_bsr` (`sfem_InexactStore.hpp:205-223`).
- Product: `bsr_spmv_group_kernel` — `src/algebra/bsr/cuda/sfem_BSR_cuda.cu:118-162` (fixed-order shuffles `:151-156`; group 16 `:11-15`, `:217`; dispatch `:243-245`).
- Figure panels: (a) `apply_points_kernel`/`Traversal`; (b) `apply_points27_kernel`; (c) `apply_matrices27_kernel`; (d) SELL, spike only.

## Cost model against BSR (main.tex:1331-1447)
- `cost_model.py` (`spikes/inexact_apply_compare/qp/`): `main` `:518`, `analyse` `:368`, `write_tables` `:419`; constants `BANDWIDTH` `:339`, `NOMINAL_BANDWIDTH` `:341`, `PEAK_F64/F32` `:342-343`, `VEC_BYTES_PER_DOF` `:345`, `HEX8_DOF` `:348`, `BSR_HEX8_SECONDS` `:350` (typed in from `spikes/inexact_apply_compare/RESULTS.md`, comment `:335-337`).

### Operator memory (main.tex:1334-1396)
- `mesh_graph` `:293`, `bsr_blocks` `:328`, `moment_store` `:213`, `fold` `:265`, `count_modal` `:229`, `count_sumfact` `:255`, `count_dense` `:278`, `codecs` `:353`.
- `measures.tex` (`measure_rows` `:456`, written `:471-476`); `footprint.tex` `:444-452`; `constants.tex` `:488-515` (`\blocks*` `:504`, `\ratioint*` `:509`, `\ratiohalf*` `:510`).
- Library layouts counted: `PointStore` (`sfem_PointStore.hpp:33`), `ElementMatrixStore` (`sfem_ElementMatrixStore.hpp:32`), `QuantBsr` (`sfem_QuantBSR.hpp:67`); BSR bytes from the ledger (`bench_inexact.exe.cpp:236-240`).

### Application throughput (main.tex:1397-1435)
- `roofline.tex` `cost_model.py:477-487`; `\MachineBandwidth` `:489`, `\MachinePeak*` `:492-493`, `\MeasuredDof` `:494`, `\BsrMeasured*`/`\BsrBound*` `:496-498`.
- Library BSR product: `bsr_spmv_rows`, `src/frontend/nonlinear/sfem_BSRTangent.cpp:261,269`.

### Time to solution (main.tex:1436-1446)
- `\ratiointHexEight` — `cost_model.py:509`.

## Experimental setup (main.tex:1449-1504)
- fig:twiststates (`figures/twist_states.png`): no generator writes that name; nearest `spikes/inexact_apply_compare/qp/viz/twist_movies.sh:39-41` (`twist_all_elements.png`), frames `viz/render.py:99`.

#### Machine (main.tex:1452-1461)
- `\MachineNominalBandwidth` `cost_model.py:490`; `\gpuDevice` `results_tables.py:508`; `\gpuNominalBandwidth` `:591` (`GPU_NOMINAL_BANDWIDTH` `:448`).
- Binding as run: `spikes/inexact_apply_compare/qp/ab_library.sbatch:27`, `ab_library_gpu.sbatch:23`.

#### Operators (main.tex:1463-1469)
- Pack size: spike `b60449f5f:.../qp/bench_qp.cpp:388-395`; `ab_library.sbatch:26-28`; library `team_apply_packed`/`packed_pack` (`sfem_InexactTraversal.hpp:151,72`), smesh `smesh_packed_mesh.cpp:420-421`; BSR rows `sfem_BSRTangent.cpp:261-269`; quantized matrix `sfem_QuantBSRStore.hpp:23`.

#### Operators on the GPU (main.tex:1471-1474)
- cuSPARSE rows: spike `b60449f5f:.../qp/cuda/bench_qp_cuda.cpp:301-303,511`; generator `GPU_CUSPARSE`/`GPU_CUSPARSE_CSR`, `results_tables.py:443-450`. The library has no cuSPARSE reference.
- Library device stores: `sfem_cuda_InexactStore.cuh:316,638`, `sfem_cuda_QuantBSRStore.cu:214`, device BSR `sfem_BSRTangent.cpp:185-215`.

#### Problems (main.tex:1476-1504)
- `\benchdof*`, `\benchn*` — `results_tables.py:195-199`; `\nqHexTwentySeven` — `cost_model.py:501-503`.
- Cases (`spikes/inexact_apply_compare/qp/cases/`): `*_stretch_shear.yaml` (dt: HEX8 and TET4 0.25 `:19`, TET10 0.0625 `:20` with `promote_to_p2` `:6`, HEX27 0.125 `:21` with `refine: 2` `:6`; Newton rtol 1e-8, unpreconditioned CG rtol 1e-3 at `:21-24`); `*_twist.yaml` (HEX8 `:14,17`, HEX27 `:16,19`, TET4 `:14,17`, TET10 `:15,18`). 10⁷-dof steps through `--set time.dt` (`ab_library.sbatch:37`).
- Protocol: spike `best_seconds`, `b60449f5f:.../qp/bench_problem.inc:59-69`, `bench_qp.cpp:201-202`; library `bench_inexact.exe.cpp:180-195` (state), `:92-99` (timing), `:141-151` (cubes).
- Newton driver: spike `newton_qp.exe.cpp`, `cuda/newton_qp_cuda.exe.cpp`; library playground → `UnconstrainedStep::linearize` (`:146`) → `NewtonSolver::solve` (`:111`).

## Results on the Grace CPU (main.tex:1506-1797)
Generators (`spikes/inexact_apply_compare/qp/`): `results_tables.py` (`main` `:727`; `write` `:168`; `write_macros` `:223`; `bench_rows` `:103`; `newton_case` `:116`; `operator_name` `:136`; sweeps `:746,768`), `paper_figures.py` (`main` `:282`; `bars` `:107`; keys `:72,84,90,95`). Data: the Grace logs (the campaign, its rerun and its fold, see the move's README).

### Correctness (main.tex:1513-1524)
- `\corrQpSixtyFour*` — `write_macros` `:272-280`; spike row `b60449f5f:.../qp/bench_qp.cpp:369`; library row `points f64` (`bench_inexact.exe.cpp:244,260`; relative difference `:120-127`).

### Application throughput, memory and set-up (main.tex:1526-1588)
- `throughput.pdf` `paper_figures.py:303-309`; `memory_footprint.pdf` `memory_figure` `:166-243`; `setup_error.pdf` `:316-328`.
- Macros `write_macros` `:246-255` (keys `:228-234`); `\blocksHexTwentySeven` `cost_model.py:504`.
- Spike rows: `bench_qp.cpp:264-284,330-331,487-488,506-509`. Library: `bench_inexact.exe.cpp:226-261`; set-up through `GeneratedNeoHookeanOgden::inexact_update` (`sfem_GeneratedNeoHookeanOgden.cpp:1345`) and `BSRTangent::assemble` (`sfem_BSRTangent.cpp:126`).

### Accuracy and the Newton iteration (main.tex:1590-1628)
- `newton_iterations.pdf`, `newton_twist.pdf` — `paper_figures.py:330-355`; macros `write_macros` `:296-315`.
- Spike `newton_qp.exe.cpp:107-152` (`--qp` variants); parsed by `parse_newton` (`results_tables.py:48-87`, steps `:67-70`).
- Library: the playground (`sfem_InexactOptions.cpp:185-196`, `sfem_Playground.cpp:1077`), step line `sfem_Observers.cpp:850`. The int8-per-element and full element-matrix counts are spike only.

### Time to solution and problem size (main.tex:1630-1708)
- `sweep_throughput.pdf` `paper_figures.py:389-422`; `sweep_newton.pdf` `:424-475`; `time_to_solution.pdf` `:357-387`.
- `summary.tex` — `write_summary` `results_tables.py:355-373`; `summary_gpu.tex` — `write_gpu` `:491-503`; `\time*` `write_macros` `:322-330`.
- Library time: `NewtonSolver::solve` trace (`sfem_NewtonSolver.cpp:112`), read by `spikes/inexact_apply_compare/qp/ab_summary.py`.

### Element matrices (main.tex:1710-1755)
- Macros: rates and ratios `write_macros` `:248-255`; errors `:281-284`; pair slowdown `:285-288`; set-up `:240-245`; Newton `:301-305`; twist `:311-315`.
- Spike rows `bench_qp.cpp:500-509`. Library: `ElementMatrixStore`, `PairFactor`; `bench_inexact` rows `element_matrix f16`, `element_matrix i8pair` only.

### Quantized assembled matrices (main.tex:1757-1795)
- Macros: `write_macros` `:248-255,258-268,301-315,325-333`; peak memory `qbsr_macros` `:403-415` (`ipnewton_*` logs).
- Spike: rows `bench_qp.cpp:324-333`, `bsr_quant.hpp`, peaks `newton_qp.exe.cpp:178,235,240`, `qp_op.hpp:62`.
- Library: `QuantBsr` (`sfem_QuantBSR.hpp:67`), `BsrStoreImpl` (`sfem_QuantBSRStore.hpp:23`; trace scopes `:40,46,49`), `inexact::make_bsr` (`sfem_InexactStore.hpp:208`); peaks `MemoryProbe` (`src/frontend/mech/sfem_Observers.cpp:678-727`), the `memory` probe (`sfem_Playground.cpp:1405-1413`); store bytes `MemoryLedger::TANGENT_VALUES/STRUCTURE` (`src/frontend/sfem_MemoryLedger.hpp:22-23`).

## Results on the H100 GPU (main.tex:1798-1930)
- `h100_throughput.tex` `write_gpu` `results_tables.py:465-469`; `h100_cusparse.tex` `:470-478`; GPU figures `paper_figures.py --machine gpu` (`:285-287`, `figure_name` `:40-41`, `file_gpu_rows` `:246-255`).
- Macros in `write_gpu` `:507-604` (bench dof `:507`; GB/s and ratios `:517-519`; against cuSPARSE `:522-528`; CSR `:529-537`; 10⁷ times, peaks, set-up shares, apply against bench `:548-561`; meshes `:564-581`; sweep `:582-589`; percentages `:590-604`); name lists `GPU_BENCH` `:426`, `GPU_NEWTON` `:431`, `GPU_TAGS` `:439`.
- Spike drivers: `cuda/bench_qp_cuda.cpp:476,478`, `cuda/newton_qp_cuda.exe.cpp:195-197`.
- Library: `bench_inexact` with `SFEM_EXECUTION_SPACE=device` (`:167-168`, held and scratch memory `:250-254`); the playground with `execution_space=device` (`ab_library_gpu.sbatch:32`); `MemoryProbe` device peaks.

## Application meshes (main.tex:1932-2032)
- `appmesh.tex`, `appmesh_tet10.tex` — `real_tables`, `results_tables.py:671-724`; `results_macros_real.tex` — `write_real` `:660-668` (`REAL_MESHES` `:622`, `REAL_ELEMENTS` `:633`, `real_rows` `:637`, `CPU_BENCH` `:626`); figures `paper_figures.application_figure` `:258-279`.
- Mesh preparation: `spikes/inexact_apply_compare/qp/real_meshes.py` (`read` `:20`, `compact` `:41`, `volumes` `:46`, `refine` `:51`, `blocks_per_row` `:77`, `main` `:83`); TET10 by smesh `mesh_promote`.
- Spike: `QP_MESH`, `b60449f5f:spikes/inexact_apply_compare/element_mesh.inc:362-365`. Library: `bench_inexact MESH <folder>` (`:5,134-140`).

## Discussion and limitations (main.tex:2034-2094)
- The choice of format (2037-2045): `write_summary` (`results_tables.py:355-373`), `write_gpu` (`:491-503`), `qbsr_macros` (`:403-415`).
- Where the compressed operators pay (2047-2056): `paper_figures.py:389-475`.
- On the GPU (2058-2068): `write_gpu` `:590-604`, `:553-554`.
- Points or element matrices (2070-2078): `cost_model.py` `analyse` `:368`, `element_matrix.tex` `:423-430`, `flops.tex` `:438-442`.
- What the error measure must see (2080-2085): `err*` macros (`write_macros` `:281-284`), Newton counts (`:296-320`); codecs `sfem_InexactCodecs.hpp:28-53`.
- Limitations (2087-2093): pack size (spike `bench_qp.cpp:388-395`; `ab_library.sbatch:26-28`; smesh `smesh_packed_mesh.cpp:420-421`); unpreconditioned CG (`cases/*.yaml`).

## Conclusions (main.tex:2096-2129)
- `\exactQpPct*`, `\exactEaPct*`, `\exactBsrPct*` — `cost_model.py:511-515`; the rest restates the results macros.
