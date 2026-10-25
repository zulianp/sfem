# Retire sub-optimal implementations

Item from TIGHT_API.md: "Retire sub-optimal implementations".

Surveyed at `717c5cfbf` (branch `tight-api`) by reading the code. Nothing was built, run or timed. Every "Result" cell says *to be measured*.

## Ground rules

- **"Retire" means:**
  1. Remove the factory name, or redirect it to the keeper.
  2. Move the class out of the default build.
  3. Record the measurement that justified the retirement.
- **Handwritten kernels stay as references.** Hand-written kernels are the reference that generated ones are measured against (standing rule), so a retired *operator* keeps its kernels compiled for the reference tests and benchmarks unless DEAD.md marks them dead. Dead code (unreachable from everything, including benchmarks and tests) goes to DEAD.md, not here.
- **Compare against the real competitor.** Every retirement compares the candidate against the variant it actually competes with: same element, same layout (packed vs standard), same geometry assumption (affine vs isoparametric), same discretisation.
- **Different discretisations on a semi-structured mesh.** On `PROTEUS_HEX{(L+1)^3}` the hand-written `sshex8` kernels compute L^3 trilinear micro elements. `SpectralElementLaplacian` and the generated PROTEUS paths may compute one order-L element per macro element. These are not competitors until the discretisation is confirmed equal (standing note: proteus-is-high-order-not-micro).
- **Function before speed.** A keeper must cover every method the retired operator's users call. Section 7 lists the capability gaps that block swaps today.

## Benchmark protocol (all families)

**Hardware and run shape**
- **Host:** one Grace socket (GH200 node), one MPI rank, `OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores`. `bench_op` rejects more than one rank.
- **Device:** one GH200 GPU, `SFEM_EXECUTION_SPACE=device`.
- Jobs run on the Alps debug partition; compile on the login node.

**Sizes and meshes**
- **Sizes at saturation.** Sweep `SFEM_BASE_RESOLUTION` until MDOF/s stops climbing, and report the plateau.
  - Starting sweep for HEX8 cubes: n = 64, 96, 128, 160, 192 (0.26M to 7.1M elements).
  - Starting sweep for TET4/TET10: n = 48, 64, 96, 128.
  - Semi-structured: `SFEM_ELEMENT_REFINE_LEVEL` L in {2, 4, 8}, with the base resolution chosen so the fine dof count matches the standard runs.
- **Distorted meshes.** Before timing, check each variant's apply against the family's oracle (the generated isoparametric operator, or plain HEX8 on the same micro elements) on a distorted mesh. Record the maximum relative difference in the "Agreement" column.

**What every row reports**
- Element count, node count and dof count. `bench_op` prints `#elements`/`#nodes` of the *base* mesh (bench_op.exe.cpp:344-345), so on semi-structured runs record the fine counts separately.
- The geometry assumption (affine / isoparametric).
- The layout: packed and standard side by side (`SFEM_USE_PACKED=1/0` in bench_op; the `SFEM_RUN_*` packed variants in bench_laplace and bench_linear_elasticity).
- Time per apply, MDOF/s and setup time.

**Drivers that can measure the variants**

| Driver | Variants it can time | Prints |
| --- | --- | --- |
| `drivers/bench/bench_op.exe.cpp` | Built in: `Laplacian` (MF, CRS, ALIGNEDCRS, SPLITDACRS, SELL), `PackedLaplacian`, `em:Laplacian`, `EMWarpOp`, `LinearElasticity` (MF, BSR, BSR_SYM), `em:LinearElasticity`, `EMMultiVectorWarpOp`, `NeoHookeanOgden` (MF, BSR), `NeoHookeanOgdenPacked`, `GeneratedNeoHookeanOgden`. Any other name via `SFEM_EXTRA_OP`, `SFEM_EXTRA_OP_TYPE` (e.g. `MF`, `BSR`), `SFEM_EXTRA_OP_BLOCK_SIZE` | `#elements`, `#nodes`, time, MDOF/s, BW, setup, (rows, cols) |
| `drivers/bench/bench_laplace.exe.cpp` | `GeneratedLaplace`, `Laplacian` (TET4/TET10/HEX8), generated packed, `PackedLaplacian`, packed two-pass; `SFEM_CODEGEN_GEOMETRY` | `#elements`, `#nodes`, `#dofs`, MElem/s, MDOF/s, FLOP and bytes per element, agreement with the baseline |
| `drivers/bench/bench_linear_elasticity.exe.cpp` | `GeneratedLinearElasticity` (standard, packed, two-pass), `LinearElasticity` baseline | as bench_laplace |
| `drivers/bench/bench_hyperelasticity.exe.cpp` | `NeoHookeanOgden` (baseline, :230), `GeneratedNeoHookeanOgden` (:229, :231), `NeoHookeanOgdenPacked` (:294) | element and dof counts, rates |
| `drivers/bench/bench_merit.exe.cpp` | `Function::value` of `SFEM_OPERATOR` on host and `gpu:` device | `#elements`, `#nodes`, `#dofs`, MElem/s |
| `drivers/bench/bench_solver.exe.cpp`, `bench_bgs_vs_cheb.exe.cpp` | Smoother and solver use of `SFEM_OPERATOR` (diag and block-diag paths) | MDOF/s |
| `bench/cuda/bench_packed_laplacian.{cpp,cu}` | Host `Laplacian` and `PackedLaplacian` vs raw and packed CUDA prototypes | `#elements`, `#nodes`, `#dofs`, `#packs` |

**Drivers that cannot be used as they stand**
- `workflows/poisson/bench.sh` is broken: `drivers/simulations/poisson.exe.cpp:119` creates op `"MF"` and never uses `SFEM_OPERATOR`, so it dereferences a null op.
- No driver times the sshex8 kernels at L=1 (`bench_op` builds a semi-structured mesh only for `SFEM_ELEMENT_REFINE_LEVEL > 1`, :249). This is needed for section 6.

Example run (host, packed):

```sh
OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
SFEM_ELEM_TYPE=HEX8 SFEM_BASE_RESOLUTION=128 SFEM_USE_PACKED=1 SFEM_REPEAT=20 \
SFEM_EXTRA_OP=GeneratedLaplace SFEM_EXTRA_OP_TYPE=MF SFEM_EXTRA_OP_BLOCK_SIZE=1 \
./bench_op
```

## Registries

- **Host:** `frontend/ops/sfem_OpFactory.cpp:53-85`. `private_register_op` silently overwrites an existing name (:90-92).
- **Generated host:** `frontend/ops/generated/sfem_generated_ops_registration.cpp:17-32`. Every material registers `Generated<X>` and `ss:Generated<X>`, which map to the same `create`.
- **Device:** `register_device_ops`, `frontend/cuda/sfem_Function_incore_cuda.cpp:1959-1969`, plus `generated/cuda/sfem_generated_device_ops_registration.cpp:19-34` (`gpu:Generated<X>`, `gpu:ss:Generated<X>`).
- **Name resolution:** `create_op(..., DEVICE)` adds the `gpu:` prefix (sfem_OpFactory.cpp:98-122). There is no switch between handwritten and generated; callers choose by name.

---

## 1. Laplacian

### Implementations

| # | Name (registry line) | Class / file | Elements | Methods | Host/device, layout | Users |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | `Laplacian` (OpFactory:63) | `Laplacian`, frontend/ops/sfem_Laplacian.cpp; kernels operators/laplacian.cpp | TRI3, TRI6, TET4, TET10, MACRO_TET4, MACRO_TRI3, HEX8, QUAD4, SS hex/quad/tet (laplacian.cpp:29-48) | apply/gradient/value with `ElementScope` (MPI). `hessian_crs` (no SS, no QUAD4), `hessian_crs_sym` (HEX8), `hessian_diag` (all). No BSR. FFF cache for TET4/TET10/HEX8/SS. SS-tet stencil (`SFEM_SSTET4_LAPLACIAN_USE_STENCIL`). Ignores `k` in CRS | host, standard | ctests: PoissonTest, MultiBlockOpTest, MultiBlockGMGDataTest, PackedLaplacianTest, SSTet4CachedFFFTest, MatrixFromatsTest, SSGMGTest, AMGTest, ParallelLaplacianApplyTest, ParallelCGTest, ParallelSSGMGTest. Drivers: assemble, mgsolve, steady_state_sim, test_galerkin_assembly, obstacle, multiblock_ssgmg_poisson. Scripts: tests/steady_state, tests/hex8_laplacian, workflows/poisson. Python: tests/test_poisson.py, tests/obstacle, tests/amg_test |
| L2 | `PackedLaplacian` (OpFactory:81) | frontend/ops/sfem_PackedLaplacian.cpp | Kernels for TET4, TET10, HEX8 only (:565-609) | `apply` only; crs, diag, gradient, value and clone all error. `SFEM_PACKED_TWO_PASS` | host, packed | ctest PackedLaplacianTest; bench_op:93, bench_laplace:232, bench_packed_laplacian; workflows/poisson (broken driver) |
| L3 | `em:Laplacian` (OpFactory:65) | `SemiStructuredEMLaplacian`, frontend/ops/sfem_SemiStructuredEMLaplacian.cpp | SSHEX8 | apply (27-point stencil from per-macro element matrices: affine per macro element), `hessian_diag`. No crs, gradient, value or `k` | host, SS | ctest MultiBlockOpTest:142; bench_op:87 (SS) |
| L4 | `SpectralElementLaplacian` (OpFactory:68) | frontend/ops/sfem_SpectralElementLaplacian.cpp; operators/spectral_hex | SS hex, single block | apply only; derefine → `Laplacian` | host, SS (high order) | **none by name** |
| L5 | `GeneratedLaplace`, `ss:GeneratedLaplace` (laplace/op/sfem_GeneratedLaplace_registration.cpp:6-7) | frontend/ops/generated/laplace | affine: TET4, TRI3, TET10, HEX8, PROTEUS_HEX8. iso: HEX8, PROTEUS_HEX8, TET10, QUAD4, PROTEUS_QUAD4. Packed: TET4, TET10, HEX8, PROTEUS_HEX8 | apply, gradient, value, value_steps, `hessian_crs`, `hessian_bsr(_typed)`. **No `hessian_diag`; block-diag errors**. `ASSUME_AFFINE*`, `PACKED_TWO_PASS`, flops and traffic, YAML. Parameter `kappa` | host, standard + packed | ctests MatrixFromatsTest, GeneratedElementRoutingTest; bench_laplace (default) |
| L6 | `gpu:Laplacian` (incore_cuda:1950) | `GPULaplacian` incore_cuda.cpp:520-787 | TET4, TET10, MACRO_TET4, HEX8, SS hex | apply/gradient (`ElementScope`), `hessian_crs` (TET4, MACRO_TET4), `hessian_crs_sym` (HEX8), `hessian_diag` (TET10 commented out). No value. **Ignores `k`** | device, standard | ctest CudaMultiBlockOpTest:134; bench_op on device |
| L7a | `gpu:em:Laplacian` = `gpu:EMOp` (incore_cuda:1951, :1953) | `GPUEMOp` :1211-1347 | HEX8, SSHEX8 | apply (8×8 SoA element matrices), gradient = apply. No derefine (cannot sit inside SS GMG) | device, SS | ctest CudaMultiBlockOpTest:142; bench_op `em:Laplacian` on device |
| L7b | `gpu:EMWarpOp` (incore_cuda:1954) | `GPUEMWarpOp` :1349-1488 | SS only | apply (AoS element matrices, Cartesian ordering). No derefine | device, SS | bench_op:89 only |
| L7c | `gpu:EMMultiVectorWarpOp` (incore_cuda:1955) | `GPUEMMultiVectorWarpOp` :1490-1642 | SS only | applies the scalar Laplacian matrix per component: a *vector Laplacian*, not elasticity | device, SS | bench_op:125 (listed among vector ops) |
| L8 | `gpu:GeneratedLaplace`, `gpu:ss:…` | laplace/op/cuda | affine: HEX8, PROTEUS_HEX8, TET10, TET4, TRI3. iso: HEX8, PROTEUS_HEX8, PROTEUS_QUAD4, QUAD4, TET10 | apply, gradient, value, BSR. crs and block-diag error; no diag; no packed device path | device, standard | none outside codegen tests |

Not a factory entry: `bench/cuda/bench_packed_laplacian.cu` holds packed CUDA prototypes (atomic and two-pass) that live only in the benchmark.

### Proposed keepers (pending measurement)

| Element / layout | Candidates | Proposed keeper | Condition |
| --- | --- | --- | --- |
| TET4, TET10, HEX8, host, standard | L1, L5 | L5 (`GeneratedLaplace`) | L5 needs `hessian_diag` (smoothers) and `k`→`kappa` mapping first. L1 stays for TRI6, MACRO_*, SS tet/quad, `hessian_crs_sym`, `ElementScope` |
| TET4, TET10, HEX8, host, packed | L2, L5 packed | L5 packed | L5 must be ≥ L2 at saturation. L2 is apply-only, so nothing else depends on it |
| SSHEX8 micro, host | L1 (FFF stencil), L3 | the faster of L1 and L3 | L3 lacks gradient and value; its coarsest derefine dereferences null (section 7) |
| SS hex high order, host | L4, L5 on PROTEUS_HEX(L+1)^3 | L5, if the discretisations match | L4 has no users |
| TET4, TET10, HEX8, device | L6, L8 | L8 | L8 needs crs/diag. L6 ignores `k` |
| SSHEX8, device | L6 (SS), L7a, L7b | the fastest of the three | L7a/L7b lack derefine, so SS GMG needs L6 today |
| Vector Laplacian SS, device | L7c | move to the `VectorLaplacian` family | it is not a Laplacian competitor |

### Measurements

| Comparison | Driver | Element / sizes | Packed | Standard | Agreement | Result |
| --- | --- | --- | --- | --- | --- | --- |
| L1 vs L5 vs L2, apply, host | bench_laplace (`SFEM_RUN_*`), bench_op + `SFEM_EXTRA_OP=GeneratedLaplace` | TET4, TET10, HEX8; saturation sweep | L2 vs L5 packed (both reductions) | L1 vs L5 | max rel. diff vs L1 | to be measured |
| L1 vs L3, SS apply, host | bench_op L∈{2,4,8} | SSHEX8 | – | yes | vs L1 | to be measured |
| L4 vs L5 PROTEUS, host | bench_op `SFEM_EXTRA_OP=SpectralElementLaplacian` and `GeneratedLaplace` | SS hex L∈{2,4} | – | yes | discretisation check first | to be measured |
| L6 vs L8, device | bench_op `SFEM_EXECUTION_SPACE=device`, `SFEM_EXTRA_OP=GeneratedLaplace` | TET4, HEX8 | – | yes | vs host L1 | to be measured |
| L6 vs L7a vs L7b, device SS | bench_op device L∈{2,4,8} | SSHEX8 | – | yes | vs host L1 | to be measured |

### Checklist

- [ ] retire `PackedLaplacian` (keep `GeneratedLaplace` packed) — evidence: L5 packed ≥ L2 on TET4/TET10/HEX8 at saturation, both reductions; PackedLaplacianTest ported to L5.
- [ ] retire `Laplacian` for TET4/TET10/HEX8 apply only (keep `GeneratedLaplace`; `Laplacian` stays for the other elements and methods) — evidence: bench_laplace L1 vs L5, standard. Blocked on L5 `hessian_diag` and the `k`/`kappa` mapping.
- [ ] retire `SpectralElementLaplacian` (keep `GeneratedLaplace` on PROTEUS_HEX) — evidence: no users; discretisation equality confirmed; L5 ≥ L4 at L=2,4.
- [ ] retire one of `Laplacian`-SS and `em:Laplacian` (keep the faster) — evidence: bench_op SS rows at L=2,4,8 plus fixed derefine.
- [ ] retire `gpu:EMOp` name (keep `gpu:em:Laplacian`) — evidence: duplicate alias of the same class (incore_cuda:1951, :1953); no tests use `EMOp`.
- [ ] retire `gpu:EMWarpOp` or `gpu:em:Laplacian` (keep the faster) — evidence: bench_op device SS rows.
- [ ] retire `gpu:Laplacian` for TET4/HEX8 apply (keep `gpu:GeneratedLaplace`) — evidence: device rows; blocked on L8 crs and diag.
- [ ] move `gpu:EMMultiVectorWarpOp` to the vector-Laplacian family — evidence: it is a vector Laplacian (incore_cuda:1547-1589).

## 2. Linear elasticity

### Implementations

| # | Name (registry line) | Class / file | Elements | Methods | Host/device, layout | Users |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | `LinearElasticity` (OpFactory:62) | frontend/ops/sfem_LinearElasticity.cpp; operators/linear_elasticity.cpp | TRI3, TET4, TET10, MACRO_TET4, HEX8, SS hex/tet/quad (SS quad calls the *generated* C ABI, :271, :302) | apply/gradient (`ElementScope`), value, value_steps, `hessian_crs` (TRI3/TET4/TET10/MACRO_TET4), `hessian_bsr` (TET4, HEX8, SS hex affine), `hessian_bcrs_sym` (HEX8), block-diag(+soa), `hessian_diag`. `SFEM_HEX8_ASSUME_AFFINE` default 1 (:470) | host, standard | many ctests (PoissonTest, NewmarkTest, MaMALTest, LinearKelvinVoigtTest, BSRTangentTest, MultiBlockOpTest, ParallelSSGMGTest, SSGMGTest, MGSDFContactTest, SAVectorAMGTest, TwoBodyContactTest, Cuda*); drivers obstacle, sdf_obstacle, obs, bench_sparse, bench_solver, bench_bgs_vs_cheb, bench_op, bench_linear_elasticity; YAML tests/sdf_obstacle/cases, V&V `legacy_*` variants; python obs.py, tests/obstacle |
| E2 | `GeneratedLinearElasticity`, `ss:…` (linear_elasticity/op/…_registration.cpp:6-7) | frontend/ops/generated/linear_elasticity | affine: TET4, TRI3, HEX8, PROTEUS_HEX8, TET10, HEX27, PROTEUS_HEX27, QUAD4, PROTEUS_QUAD4. iso: + TRI6. Packed: HEX8, HEX27, PROTEUS_HEX8/27, TET10, TET4 (affine) | apply, gradient, value, value_steps, BSR(_typed), block_diag_sym, inexact (partial assembly). **No crs, no diag**. Parameter `lmbda` | host, standard + packed | ctests MatrixFromatsTest, NewtonSolverTest, StepperTest, TimeSchemeIndependenceTest, GeneratedElementRoutingTest, GeneratedDerefineTest, BSRTangentTest, PlaygroundCasesTest, MGSDFContactTest; bench_linear_elasticity, bench_merit; workflows/elastodynamics, V&V `generated_*` |
| E3 | `em:LinearElasticity` (OpFactory:66) | `SemiStructuredEMLinearElasticity` | SS of HEX8 (element-matrix stencil) or TET4 (stencil) | apply, diag, block-diag; no crs, gradient or value | host, SS | ctests MultiBlockOpTest:214, SSGMGTest:185; bench_op:123 |
| E4 | `LinearElasticityHyTeG` (OpFactory:67) | `SemiStructuredHyTeGLinearElasticity` | SSHEX8 | apply (HyTeG category stencils), affine diag and block-diag | host, SS | ctest SSHex8LinearElasticityHytegTest only |
| E5 | `gpu:LinearElasticity` (incore_cuda:1949) | `GPULinearElasticity` :789-1162 | TET4, TET10, MACRO_TET4, HEX8, TRI3, SS hex (exact route via corners when not affine) | apply, gradient, `hessian_bsr` (non-SS), diag, block-diag; crs and value IMPLEMENT_ME | device | ctests CudaMultiBlockOpTest, CudaEMVectorWarpOpTest, CudaContactTest; bench_op device |
| E6 | `gpu:em:LinearElasticity` = `gpu:EMVectorWarpOp` (incore_cuda:1952, :1956) | `GPUEMVectorWarpOp` :1644-1946 | SS hex, block size 3 | apply (24×24 AoS); delegates bsr/diag/block-diag/value to a companion E5 | device, SS | ctests CudaEMVectorWarpOpTest:193,241 (name `EMVectorWarpOp`), CudaMultiBlockOpTest:152; bench_op device |
| E7 | `gpu:GeneratedLinearElasticity`, `gpu:ss:…` | linear_elasticity/op/cuda | affine and iso as E2, no packed | BSR, block_diag_sym; crs errors | device | bench_merit:357 |
| E8 | Legacy SymPy scripts `python/codegen/linear_elasticity_op.py`, `gpu_linear_elasticity_op.py` | stdout C generators | TRI3…HEX8 by argv | n/a | not compiled | none (only `python/codegen/framework/retired/PRD.md:5`); provenance of `*_loperand` routines in the handwritten headers (unverified) |

### Proposed keepers (pending measurement)

| Element / layout | Candidates | Proposed keeper | Condition |
| --- | --- | --- | --- |
| TET4, TET10, HEX8, host, standard | E1, E2 | E2 for apply/gradient/value/BSR/block-diag | E1 stays for `hessian_crs`, `hessian_bcrs_sym`, `hessian_diag`, MACRO_TET4, SS tet/quad until E2 has them |
| Same, host, packed | E2 packed only | E2 | – |
| SSHEX8 micro, host | E1 (affine adjugate or exact), E3, E4 | the fastest that has diag + block-diag + derefine | E4 has one user; E1's affine route is 1.2% off on distorted meshes (standing note) |
| SSTET4, host | E1, E3 | measure | – |
| Device, standard | E5, E7 | E7 | E7 needs diag. E5 has no crs or value either |
| Device, SSHEX8 | E5, E6 | measure; E6 already delegates everything except apply to E5 | – |
| Legacy SymPy scripts | E8 | retire | no output consumed |

### Measurements

| Comparison | Driver | Element / sizes | Packed | Standard | Agreement | Result |
| --- | --- | --- | --- | --- | --- | --- |
| E1 vs E2, apply and BSR, host | bench_linear_elasticity; bench_op `SFEM_EXTRA_OP=GeneratedLinearElasticity SFEM_EXTRA_OP_BLOCK_SIZE=3` (MF and BSR) | TET4, TET10, HEX8; affine and iso | E2 packed vs E1 standard | yes | vs E2 iso | to be measured |
| E1 vs E3 vs E4, SS, host | bench_op L∈{2,4,8}, `SFEM_EXTRA_OP=LinearElasticityHyTeG` | SSHEX8 (+SSTET4 for E1/E3) | – | yes | vs HEX8 on micro elements | to be measured |
| Diag and block-diag as smoothers | bench_bgs_vs_cheb, bench_solver (`SFEM_OPERATOR`) | HEX8, SSHEX8 | – | yes | equal stopping accuracy | to be measured |
| E5 vs E7, device | bench_op device; bench_merit (value) | TET4, HEX8 | – | yes | vs host | to be measured |
| E5 vs E6, device SS | bench_op device L∈{2,4,8} | SSHEX8 | – | yes | vs host | to be measured |

### Checklist

- [ ] retire `LinearElasticity` apply/BSR for TET4/TET10/HEX8 (keep `GeneratedLinearElasticity`) — evidence: E2 ≥ E1 at saturation, packed and standard. Blocked on E2 `hessian_diag`/`hessian_crs` and the `lambda`/`lmbda` mapping.
- [ ] retire `LinearElasticityHyTeG` (keep the faster of `LinearElasticity`-SS and `em:LinearElasticity`) — evidence: bench_op SS rows; one test user.
- [ ] retire one of `LinearElasticity`-SS and `em:LinearElasticity` (keep the faster with derefine) — evidence: bench_op SS rows plus smoother timings.
- [ ] retire `gpu:EMVectorWarpOp` name (keep `gpu:em:LinearElasticity`) — evidence: alias of the same class; port CudaEMVectorWarpOpTest to the canonical name.
- [ ] retire `gpu:LinearElasticity` for standard elements (keep `gpu:GeneratedLinearElasticity`) — evidence: device rows; blocked on E7 diag.
- [ ] retire `python/codegen/linear_elasticity_op.py` and `gpu_linear_elasticity_op.py` (keep `python/codegen/framework/generators/linear_elasticity.py`) — evidence: not referenced by CMake, tests or scripts. Also list the sibling scripts `laplace_op.py`, `quadrature_laplace_op.py`, `gpu_lapl.py`, `gpu_lapl_apply.py` and `lapl.py` in DEAD.md.

## 3. NeoHookean (Ogden)

### Implementations

| # | Name (registry line) | Class / file | Elements | Methods | Host/device, layout | Users |
| --- | --- | --- | --- | --- | --- | --- |
| N1 | `NeoHookeanOgden` (OpFactory:73) | frontend/ops/sfem_NeoHookeanOgden.cpp; operators/neohookean_ogden.cpp | TET4, TET10, HEX8, SS hex | gradient, value(_steps), apply (partial assembly forced for HEX8/TET10/SS), `hessian_diag` (HEX8, SS), `hessian_bsr` (HEX8, TET4, TET10; SS rejected), `hessian_bcrs_sym` (HEX8). **`hessian_crs` is `IMPLEMENT ME`** (neohookean_ogden.cpp:251-263). `SFEM_HEX8_ASSUME_AFFINE` is read (:90-92) but never used | host, standard | ctests PartialAssemblyTest:181, MGSDFContactTest:703,731; bench_op:131,155; bench_hyperelasticity:230; YAML workflows/hyperelasticity/{hyperelasticity,_blade,_torus}.yaml, bone, cables, scripts/hyperelasticity_torus.yaml; V&V hyperelastic_modes_3d:179,187 |
| N2 | `NeoHookeanOgdenPacked` (OpFactory:74) | frontend/ops/sfem_NeoHookeanOgdenPacked.cpp | gradient: HEX8, TET4, TET10. apply: HEX8, TET10 | packed gradient and partial-assembly apply; crs not implemented; diag, BSR and bcrs_sym use the non-packed dispatch (`mesh->n_elements()` in diag, :978-1012: suspected multi-block bug) | host, packed | bench_op:135; bench_hyperelasticity:294; workflows/hyperelasticity/hyperelasticity_hex8.yaml |
| N3 | `Hyperelasticity` (OpFactory:80) | frontend/ops/sfem_Hyperelasticity.cpp | HEX8 only (:720); other types get a null kernel | NeoHookean via generic helpers; `hex8::MooneyRivlin` is a stub that is never registered; crs `IMPLEMENT ME`; reads env on every call; misspelled `SFEM_HYPERELAStICITY_USE_AOS` (:833) | host | **none** |
| N4 | `GeneratedNeoHookeanOgden`, `ss:…`, `gpu:…`, `gpu:ss:…` | frontend/ops/generated/neohookean_ogden | tri3, tri6, quad4, proteus_quad4, tet4, tet10, hex8, hex27, proteus_hex8, proteus_hex27 (HEX8 forwards to PROTEUS_HEX8 by column permutation) | gradient, apply, value(_steps), BSR(_typed), inexact (stored, compressed, packed two-pass tangents), affine/iso/packed/patch variants; **no crs, no block-diag, no diag** | host + device, standard + packed | ctests NewtonSolverTest, MatrixFromatsTest, TimeSchemeIndependenceTest, GeneratedElementRoutingTest, GeneratedOpWrapperCompileTest, MeritInterfaceTest, BSRTangentTest, CudaBSRTangentTest; bench_op:140, bench_hyperelasticity:229,231, neohookean_assemble; V&V hyperelastic_modes_{2d,3d}, inflated_spherical_shell_3d |
| N5 | legacy TET4 neohookean (no factory) | operators/tet4/tet4_neohookean.cpp, tet4_neohookean_principal_stresses.cpp; `tet4_neohookean_active_strain.{cpp,hpp}` are 0 bytes; a duplicate `neohookean_assemble_hessian` in vectorized/tet4_simd_neohookean.cpp:901 | TET4 | hessian, gradient, value, Cauchy stress, von Mises, principal stresses | host | drivers cauchy_stress, cprincipal_stresses, vonmises; workflows/mech/stress.sh, principal_stresses.sh |
| N6 | `NeoHookeanOgdenActiveStrainPacked` (OpFactory:77) | frontend/ops/sfem_NeoHookeanOgdenActiveStrainPacked.cpp | HEX8 | active strain; **no generated counterpart** | host | PartialAssemblyTest:186 (dead branch, section 7); workflows/hyperelasticity/active_strain_hex8.yaml; V&V active_strain_homogeneous_3d |

### Proposed keepers

- **HEX8, TET4, TET10 (host, standard and packed) and device: N4.**
  - N1 stays as the hand-written reference, and for `hessian_diag` and `hessian_bcrs_sym`, until N4 gains diag.
  - N2 is retired if N4 packed inexact is at least as fast.
- **SS hex:** N1's sshex8 path vs `ss:GeneratedNeoHookeanOgden`. Confirm the discretisation first (micro trilinear vs Qp).
- **N3:** retire.
- **N5:** keep until a generated stress post-processing path (`quadrature_output`) covers the three drivers. It is a different energy from Ogden and must be checked before any swap.
- **N6:** keep. No replacement exists.

### Measurements

| Comparison | Driver | Element / sizes | Packed | Standard | Agreement | Result |
| --- | --- | --- | --- | --- | --- | --- |
| N1 vs N2 vs N4, gradient and apply (PA / inexact) | bench_hyperelasticity; bench_op | HEX8, TET10 (TET4 gradient) | N2 vs N4 packed | N1 vs N4 | vs N4 iso | to be measured |
| N1 vs N4, BSR assembly | bench_op `NeoHookeanOgden` BSR vs `SFEM_EXTRA_OP=GeneratedNeoHookeanOgden SFEM_EXTRA_OP_TYPE=BSR`; neohookean_assemble | HEX8, TET4, TET10 | – | yes | vs N4 | to be measured |
| N1-SS vs `ss:GeneratedNeoHookeanOgden` | bench_op L∈{2,4} | SS hex | – | yes | discretisation check | to be measured |
| Newton solve end to end | playground on workflows/hyperelasticity/hyperelasticity.yaml with `material.type` swapped | as in the YAML | – | yes | equal stopping accuracy | to be measured |

### Checklist

- [ ] retire `Hyperelasticity` (keep `GeneratedNeoHookeanOgden`) — evidence: no users; HEX8 only; Mooney-Rivlin stub; crs `IMPLEMENT ME`.
- [ ] retire `NeoHookeanOgdenPacked` (keep `GeneratedNeoHookeanOgden` packed inexact) — evidence: bench_hyperelasticity packed rows, HEX8 and TET10; port hyperelasticity_hex8.yaml.
- [ ] retire `NeoHookeanOgden` from workflows (keep `GeneratedNeoHookeanOgden`; N1 stays registered as reference) — evidence: bench rows plus the Newton end-to-end at equal accuracy; blocked on N4 diag (smoothers) and on SS discretisation equality for MGSDFContactTest.
- [ ] fix or remove `NeoHookeanOgden::hessian_crs` (`IMPLEMENT ME`) — evidence: code read; no keeper offers CRS.
- [ ] retire legacy `tet4_neohookean` drivers (keep generated `quadrature_output`) — evidence: identical stresses on TET4 for the same energy; blocked on that path existing.

## 4. Mooney-Rivlin

### Implementations

| # | Name (registry line) | Class / file | Elements | Methods | Host/device, layout | Users |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | `MooneyRivlin` (OpFactory:78) **and** `MooneyRivlinActiveStrainPacked` (:79), same `create`; `name()` always returns the latter | frontend/ops/sfem_MooneyRivlinActiveStrainPacked.cpp; operators/hex8/hex8_mooney_rivlin.cpp, hex8_mooney_rivlin_local.hpp (4,053 lines) | HEX8 | gradient, value(_steps), diag, BSR (:421), bcrs_sym (:462); apply is non-packed partial assembly despite the name; needs field `Fa`; no crs | host | PartialAssemblyTest (both names, **never executed**, :180-187); V&V active_strain_homogeneous_3d:161,169 |
| M2 | `MooneyRivlinVisco` (**not registered**) | frontend/ops/sfem_MooneyRivlinVisco.cpp; operators/mooney_rivlin_visco.cpp → hex8_mooney_rivlin_visco_flexible.cpp, which includes `_unique_Hi_local.hpp` (4,825 lines; the `_standard.hpp` include is commented out at :19) | HEX8 | gradient, BSR, diag, update_history; `hessian_crs` forwards to BSR (suspect); apply and value fail | host | ctest GeneratedMooneyRivlinPronyTest (parity reference only) |
| M2s | `hex8_mooney_rivlin_visco_unique_Hi_standard.hpp` (4,483 lines) | not compiled | – | "Standard MR (matches Marc)" form; differs from `_local` in 3 function bodies (1,156 diff lines), ~3,900 lines identical; both regenerable from `python/codegen/sr_visco_hyper_unique_Hi.py` (`USE_STANDARD`, :2260) | – | none |
| M3 | `GeneratedModifiedMooneyRivlin`, `ss:`, `gpu:`, `gpu:ss:` | generated/modified_mooney_rivlin | tri3, tri6, quad4, proteus_quad4, tet4, tet10, hex8, hex27, proteus_hex8/27/64 | as N4 without inexact; no crs, no block-diag | host + device | ctest CudaGeneratedMooneyRivlinPronyTest:194; V&V cylindrical_pressure_vessel, hyperelastic_modes_{2d,3d}, inflated_spherical_shell_3d |
| M4 | `GeneratedMooneyRivlinProny`, `ss:`, `gpu:`, `gpu:ss:` | generated/mooney_rivlin_prony | tri3, quad4, proteus_quad4, tet4, tet10, hex8, proteus_hex8 | + history, `quadrature_output`; no crs, no block-diag | host + device | ctests MRPureElasticValidation, MRViscoExcelValidation, PlaygroundCasesTest, PlaygroundContactTest, ElasticityCaseTest, GeneratedMooneyRivlinPronyTest, MooneyRivlinViscoTest, MRHyperelastic(Excel)Validation, CudaEnergyBSRAssemblyTest, CudaGeneratedMooneyRivlinPronyTest; workflows prony_torsion (10 cases), mooney_rivlin_gravity, ldp_containment/transient |

### Proposed keepers

- **Viscoelastic Mooney-Rivlin: M4 everywhere.**
  - M2 stays only as the parity reference of GeneratedMooneyRivlinProny.
  - M2s goes to DEAD.md.
- **Hyperelastic Mooney-Rivlin without history: M3.** Measure whether M4 with zero Prony terms equals M3 and is as fast. If so, M3 is a candidate too; check that both use the same deviatoric form.
- **Active strain Mooney-Rivlin: M1.** No generated replacement exists. Drop the `MooneyRivlin` alias.

### Measurements

| Comparison | Driver | Element / sizes | Packed | Standard | Agreement | Result |
| --- | --- | --- | --- | --- | --- | --- |
| M2 vs M4, gradient and BSR | bench_op `SFEM_EXTRA_OP=GeneratedMooneyRivlinProny` (MF/BSR, `SFEM_EXTRA_OP_FIELDS=previous`); M2 needs a driver hook (unregistered) | HEX8 | – | yes | GeneratedMooneyRivlinProny parity tolerances | to be measured |
| M3 vs M4 with zero Prony terms | bench_op with `SFEM_EXTRA_OP` | HEX8, TET10 | yes | yes | vs each other | to be measured |

### Checklist

- [ ] retire the `MooneyRivlin` factory name (keep `MooneyRivlinActiveStrainPacked`) — evidence: duplicate registration (OpFactory:78-79); its only user is the dead test branch.
- [ ] fix `sfem_PartialAssemblyTest` (`return A; +B;`, :180-187) so the Mooney-Rivlin and active-strain cases run — evidence: code read. This must happen before M1 is used as evidence.
- [ ] retire `hex8_mooney_rivlin_visco_unique_Hi_standard.hpp` (keep the generator script `sr_visco_hyper_unique_Hi.py`) — evidence: not compiled; regenerable.
- [ ] retire `MooneyRivlinVisco` (keep `GeneratedMooneyRivlinProny`) — evidence: unregistered; one test user; parity already established by GeneratedMooneyRivlinProny. Decide whether the reference role continues.
- [ ] retire `GeneratedModifiedMooneyRivlin` or keep it (decide on M3 vs M4-zero-Prony) — evidence: equal energy and equal or better rate.

## 5. InertiaPotential

| Name | Line | Users by string |
| --- | --- | --- |
| `InertiaPotential` | OpFactory:54 | none (users construct the class directly: sfem_NewmarkScheme.cpp:38, sfem_BDF2Scheme.cpp:44, sfem_DerefinedTimeScheme.cpp:46-47) |
| `BDF2InertiaPotential` | :60 | none |
| `NewmarkInertiaPotential` | :61 | none |

- One class: frontend/ops/sfem_InertiaPotential.cpp, with device kernels in frontend/ops/cuda/sfem_InertiaPotential_cuda.cu.
- The comment at :55-59 says the old keys stay "because drivers and env vars resolve operators by string". A grep of `*.cpp/*.hpp/*.py/*.yaml/*.sh` finds no such string user.

- [ ] retire `BDF2InertiaPotential` and `NewmarkInertiaPotential` names (keep `InertiaPotential`) — evidence: no string users (grep). Measurement: none (same class).

## 6. HEX8 vs SSHEX8: what HEX8 → PROTEUS_HEX8 delegation needs

**How delegation would work**
- `PROTEUS_HEX8` (100008) is the level-1 semi-structured hex.
- Delegating means permuting element columns 2↔3 and 6↔7 (no copy). The generated code already does this: `neohookean_ogden_hex8_inexact_apply_operator.cpp:101-111`, `pipeline/driver.py:2570`.

**Kernels by function**

| Function | HEX8 | SSHEX8 | Gap |
| --- | --- | --- | --- |
| Laplacian apply / FFF apply / diag | yes | yes (`sshex8_laplacian_apply`, `affine_sshex8_laplacian_stencil_apply_fff`, `affine_sshex8_laplacian_diag`) | diag is affine only |
| Laplacian CRS / CRS-sym | yes | **no** (element-matrix building blocks only); rejected at laplacian.cpp:170, :323 | assembly missing |
| Linear elasticity apply, BSR, diag, block-diag, objective_steps | yes | yes | – |
| Linear elasticity SoA adjugate apply | yes | no | missing |
| Linear elasticity CRS-sym | yes | `affine_sshex8_elasticity_crs_sym` exists (sshex8_linear_elasticity.hpp:114) but is **not wired** (linear_elasticity.cpp:534) | wiring |
| `l2_project_cauchy_stress` | yes (hex8_linear_elasticity.hpp:137) | **no** | missing (one user: drivers/postprocess/hex8_cauchy_stress.exe.cpp:39) |
| Mass apply | `hex8_apply_mass` | **no** (`sshex8_mass_apply` commented out, sshex8_mass.hpp:11-17); `apply_mass` has no SS branch (mass.cpp:275-330) | missing |
| Lumped mass | yes | `affine_sshex8_mass_lumped` (affine only) | iso missing |
| NeoHookean objective / gradient / PA / compressed PA / diag | yes | yes | – |
| NeoHookean PA diag, BSR, bcrs_sym | yes | **no** (neohookean_ogden.cpp:335, :667, :705) | missing |
| NeoHookean active strain (13 functions) | yes | **no** (sshex8_neohookean_ogden_active_strain.{cpp,hpp} entirely commented out) | missing |
| Mooney-Rivlin active strain (10), Mooney-Rivlin visco | yes | **no** | missing |
| CUDA | laplacian, linear elasticity | laplacian, linear elasticity, elemental matrix | – |

**Performance at L=1**
- Every sshex8 kernel allocates its scratch with `malloc`/`calloc` inside `#pragma omp parallel`, once per thread per call. Examples: sshex8_laplacian.cpp:68-70, 203-205, …; sshex8_linear_elasticity.cpp:82-86, …; sshex8_neohookean_ogden.cpp:76-85, ….
- One per-element `calloc` exists, debug builds only (sshex8_laplacian.cpp:499).
- At L=1 the per-call cost is not amortised over L^3 micro elements. This also conflicts with TODO.md P0 "Remove per-apply heap allocations".
- The HEX8 kernels use stack arrays.

**Affine environment variables**
- `SFEM_HEX8_ASSUME_AFFINE` (default 1) and `SFEM_HEX8_ASSUME_AXIS_ALIGNED` (default 0) are read on every call of `hex8_laplacian_apply` (operators/hex8/hex8_laplacian.cpp:17-20). The `Laplacian` op never reaches that function: it always uses the FFF `hex8_laplacian_apply_opt`. So they affect only `drivers/laplacian_apply.exe.cpp` and `lapl.exe.cpp`, and exporting them in tests/hex8_laplacian and tests/steady_state does nothing on the op path.
- `SFEM_HEX8_ASSUME_AFFINE` is also read by `LinearElasticity` (:470, gates the adjugate cache and is consulted only in `initialize`), by `GPULinearElasticity` (incore_cuda:804), and by `NeoHookeanOgden` (:90, no effect).
- Delegation must turn these into typed options (TODO.md P0 "Move env parsing to drivers").

**Measurement: HEX8 vs PROTEUS_HEX8 at L=1**

| Kernel | HEX8 rate | sshex8 L=1 rate | Agreement | Result |
| --- | --- | --- | --- | --- |
| Laplacian apply (FFF) | | | vs HEX8 | to be measured |
| Laplacian diag | | | | to be measured |
| Linear elasticity apply (affine / exact) | | | | to be measured |
| Linear elasticity BSR, diag, block-diag | | | | to be measured |
| NeoHookean gradient, PA apply | | | | to be measured |

- Driver: bench_op needs `to_semistructured(1, …)` for `SFEM_ELEMENT_REFINE_LEVEL=1` (it requires `> 1` today, bench_op.exe.cpp:249). That is a one-line driver change.
- Grace 72 threads and GH200; sizes and reporting as in the protocol.

### Checklist

- [ ] add sshex8 CRS assembly for Laplacian and NeoHookean (or route HEX8 CRS to BSR users) — prerequisite for delegation.
- [ ] wire `affine_sshex8_elasticity_crs_sym`, add the sshex8 SoA adjugate apply, NeoHookean PA diag, BSR and bcrs_sym, mass apply, iso lumped mass, `l2_project_cauchy_stress`, active strain (NeoHookean, Mooney-Rivlin) and Mooney-Rivlin visco — prerequisites.
- [ ] hoist the sshex8 per-call scratch into per-thread arenas allocated at `initialize` — evidence: L=1 rows match HEX8 before delegating.
- [ ] replace `SFEM_HEX8_ASSUME_AFFINE`/`_AXIS_ALIGNED` env reads in kernels and ops with typed options set before `initialize`.
- [ ] retire the duplicated hand-written HEX8 kernels behind PROTEUS_HEX8 delegation (keep them as references per the standing rule) — evidence: the L=1 table at parity or better on Grace and GH200.

## 7. Defects found while surveying (they block or bias the measurements)

1. `sfem_PartialAssemblyTest.cpp:238-245`: `return A; +B;`, so the Mooney-Rivlin and active-strain halves never run.
2. `drivers/simulations/poisson.exe.cpp:119` creates op `"MF"` (null) and ignores `SFEM_OPERATOR`. This breaks workflows/poisson.
3. `em:Laplacian` coarsest derefine calls `Laplacian::override_element_types` on an uninitialised op, which dereferences null (sfem_SemiStructuredEMLaplacian.cpp:95-98; sfem_Laplacian.cpp:490-492).
4. `LinearElasticity::hessian_{crs,bsr,bcrs_sym}` discard the `iterate` result and always return success (sfem_LinearElasticity.cpp:538-624).
5. `ASSUME_AFFINE` set after `initialize` does nothing in `LinearElasticity`/`GPULinearElasticity`. Their `derefine_op` and YAML paths do exactly that, so coarse SS levels always run affine.
6. `GPULaplacian` ignores `k`; host `Laplacian` CRS ignores `k`.
7. `NeoHookeanOgden(Packed)::hessian_diag` and `value_steps` pass `mesh->n_elements()` rather than the block count (suspected multi-block bug).
8. `GeneratedNeoHookeanOgden`, `GeneratedLaplace` and `GeneratedLinearElasticity` lack `hessian_diag`; the generated ops lack CRS (except Laplace) and block-diag (Laplace). These are the main swap blockers.
9. Parameter names differ between handwritten and generated: `k`/`kappa`, `lambda`/`lmbda`.
10. `register_GeneratedNavierStokes_generated_op` is never called on the host, while `gpu:GeneratedNavierStokes` is registered. `GeneratedNeumann(General)` is registered twice (OpFactory:75-76 and generated).
