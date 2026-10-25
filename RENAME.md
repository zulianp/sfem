# Renames: approval checklist

Tick `[x]` to approve an item. To pick another name, edit the new name in place before ticking. Each approved item is applied mechanically in one commit: every call site, the Python bindings (`python/bindings/pysfem.cpp`), the code generator when the name is generated (`python/codegen/`), and the docs. The commit then builds and runs ctest. Old names are not kept as aliases unless the item says **alias**. Names that also live in `external/smesh` are renamed on smesh `main` first.

Scope counts come from `rg` in this worktree at `717c5cfbf`, over C/C++/CUDA/Python/CMake/shell/YAML files, excluding `build*/`, `venv/`, `.git` and (unless stated) `external/`. "occ/files" means occurrences and files. Standing rules: a name describes the operation or kernel, not jargon; methods and functions are snake_case; classes are CamelCase. A **Considered, no change** list closes each group, so you can see what was checked and left as it is.

## 1. Factories

**Convention.** An object is created by a static member of its class: `X::create(...)`, returning `std::shared_ptr<X>`. The execution space is a parameter of `create`. There is no free wrapper whose only job is the `to_device` dispatch. Other sources keep the existing suffixes on the same class: `X::create_from_yaml`, `X::create_from_file`, `X::create_from_env`. Some builders assemble an object from types in another module, such as a solver built from a `Function`. Those stay free `create_<thing>` functions in the module that knows both types. A function that only wraps an existing object behind another interface is `as_<interface>(obj)`. `make_*` is retired. This is the layout the `sfem_API.hpp` reorganisation produces: the factories move next to their classes.

- [ ] F1 `create_cg(op, es)` → `ConjugateGradient<T>::create(op, es)`: 35 occ/23 files (`algebra/sfem_solver_factory.cpp:15`). The algebra class owns its construction.
- [ ] F2 `create_parallel_cg` → `ParallelConjugateGradient<T>::create`: 6/4 (`sfem_solver_factory.cpp:34`). Same as F1.
- [ ] F3 `create_bcgs` → `BiCGStab<T>::create`: 12/9 (`sfem_solver_factory.cpp:141`). Removes the `bcgs` contraction too.
- [ ] F4 `create_cheb3` → `Chebyshev3<T>::create`: 4/4 (`sfem_solver_factory.cpp:160`). Removes the `cheb3` contraction.
- [ ] F5 `create_mprgp` → `MPRGP<T>::create`: 7/7 (`sfem_solver_factory.cpp:177`).
- [ ] F6 `create_mg` → `Multigrid<T>::create`: 3/3 (`sfem_solver_factory.cpp:197`).
- [ ] F7 `create_stationary` → `StationaryIteration<T>::create`: 9/7 (`sfem_solver_factory.cpp:119`).
- [ ] F8 `create_shiftable_jacobi`, `create_shiftable_block_sym_jacobi`, `create_mixed_precision_shiftable_block_sym_jacobi` → `ShiftableJacobi<T>::create`, `ShiftableBlockSymJacobi<T>::create`, `MixedPrecisionShiftableBlockSymJacobi<HP, LP>::create`: 16/11, 8/5 and 4/3 (`sfem_solver_factory.cpp:52-95`).
- [ ] F9 `create_dirichlet_conditions(space, conds, es)` and `create_dirichlet_conditions_from_env(space, es)` → `DirichletConditions::create(space, conds, es)` and `DirichletConditions::create_from_env(space, es)`: 45/25 and 6/6 (`frontend/sfem_DirichletConditions.cpp:771`, `781`). The class already has `create` and `create_from_env`; the free wrappers only add the device dispatch. That is two names for one operation.
- [ ] F10 `create_neumann_conditions(space, conds, es)` → `NeumannConditions::create(space, conds, es)`: 4/4 (`frontend/sfem_NeumannConditions.cpp:1265`). Same as F9.
- [ ] F11 `create_contact_conditions_from_env(space, es)` → `AxisAlignedContactConditions::create_from_env(space, es)`: 2/2 (`frontend/sfem_ContactConditions.cpp:984`). Same as F9.
- [ ] F12 `create_contact_surface(space, ...)` → `ContactSurface::create(space, ...)`: 2/2 (`frontend/sfem_ContactSurface.hpp:116`). Its implementations `MeshContactSurface` and `SSMeshContactSurface` already use `create`/`create_from_file`.
- [ ] F13 class `sfem::Factory` → `sfem::OpFactory`. `Factory::create_op(space, name)`, `Factory::create_op_gpu(space, name)` and the free `create_op(space, name, es)` become one `OpFactory::create(space, name, es)`. Scope: 35/13, 3/2 and 136/61 (`frontend/ops/sfem_OpFactory.hpp:40,84,94,134`). The file is already `sfem_OpFactory.hpp`. Three spellings of one lookup collapse into one, and `es` replaces the `_gpu` twin.
- [ ] F14 `Factory::create_boundary_op` → `OpFactory::create_boundary(space, elements, name, es)`: 6/5 (`sfem_OpFactory.hpp:101`). Same as F13.
- [ ] F15 free `create_op_from_yaml(space, node, es)` → `OpFactory::create_from_yaml(space, node, es)`: 17/7 (`sfem_OpFactory.hpp:145-149`). This puts the YAML source on the factory, as for every other class. `Factory::register_op` keeps its name and only takes the new class qualifier (69/33).
- [ ] F16 `d_op_str(name)` → internal `device_op_name(name)` in `sfem_OpFactory.cpp`: 3/2 (`sfem_OpFactory.hpp:131`). It is used only inside the factory once F13 lands.
- [ ] F17 `make_op(rows, cols, fn, es)` → `LambdaOperator<T>::create(rows, cols, fn, es)` and `make_in_place_op` → `InPlaceLambdaOperator<T>::create`: 12/8 and 2/2 (`algebra/sfem_Operator.hpp:144,171`). `make_op` currently names two unrelated operations (F18 is the other).
- [ ] F18 `make_op(std::shared_ptr<smesh::AbstractRestrict>)` → `as_operator(restrict)` (`frontend/sfem_Transfer.cpp:171`). It wraps an existing object and creates no state.
- [ ] F19 `make_linear_op(f)` and `make_linear_op_variant(f, opts)` → `as_operator(f)` and `as_operator(f, opts)`: 6/6 and 1/1 (`frontend/sfem_LinearOperatorFactory.cpp:385,393`). Both are adapters (F18).
- [ ] F20 `make_parallel_op(...)` (2 overloads) → `LambdaParallelOperator<T>::create(...)`, and `make_team_op` → `TeamParallelOperator<T>::create`: 12/7 and 4/4 (`algebra/parallel/sfem_ParallelOperator.hpp:141-163`). Both construct the class they return.
- [ ] F21 `make_openmp_blas<T>()` and `make_cuda_blas<T>()` → `OpenMPBLAS<T>::create()` and `GPUBLAS<T>::create()` (class names from B1): 29/22 and 15/7. Callers that choose by space use the existing `blas<T>(es)` (134 occ).
- [ ] F22 member `make_buffer(n)` in `ShiftedPenalty` and `ShiftedPenaltyMultigrid` → member `create_buffer(n)`: 20 occ/2 files (`algebra/sfem_ShiftedPenalty.hpp:103`, `algebra/sfem_ShiftedPenaltyMultigrid.hpp:516`, used at `:244`). It allocates through the solver's BLAS. `create_buffer` is the library's word for this (smesh `create_buffer`, 478/92 via `frontend/sfem_aliases.hpp:43`). The test-local `static make_buffer(initializer_list)` helpers in three `algebra/*/tests` files stay; they are file-local.

**Considered, no change:**
- The free builders `create_ssmgc`, `create_ssgmg`, `create_ssmg`, `create_shifted_penalty`, `create_parallel_preconditioner`, `create_parallel_matrix_free_operator`, `create_hierarchical_{prolongation,restriction}[_from_graph]` and `create_inverse_diagonal_scaling` construct algebra objects from a `Function`/`FunctionSpace`. So do `create_energy_merit`/`create_residual_merit`/`create_merit`, `create_contact`, `create_relaxation_integrator`, `create_torque_probe`/`create_named_probe`, `create_obstacle_sdf` and `create_derefined_crs_graph`, or they return a plain `Operator<T>` built from a lambda. They follow the module-builder rule and move with the `sfem_API.hpp` split.
- `FunctionSpace::create`, `Function::create`, `SSMGCStep::create`, `BSRTangent::create`, every `X::create_from_file`/`create_from_yaml` (107/74 and 137/78), and smesh's `create_buffer`/`create_host_buffer`.

## 2. Counts and sizes

**Convention.** A size accessor is `n_<entity>[_<qualifier>]()`: the entity comes first, the qualifier after (`n_dofs_owned`, `n_dofs_global`). That is smesh's order (`n_nodes_owned`, `n_nodes_global`, `n_elements_owned`, `n_elements_global`, `n_nodes_owned_not_shared`). C++ class APIs use no abbreviated form (`ndofs`, `nnodes`, `nelements`). Linear-algebra shapes are `rows()`/`cols()`.

The public C++ methods already use `n_dofs`/`n_nodes`/`n_elements`: no `ndofs()`, `nnodes()` or `nelements()` method exists in a public header. What remains:

- [ ] C1 `FunctionSpace::n_owned_dofs()` → `n_dofs_owned()`: 35 calls/16 files (`frontend/sfem_FunctionSpace.hpp:72`). Its sibling is `n_dofs_global()` (`:73`), and smesh says `n_nodes_owned()`. Today sfem is the only place with the qualifier first.
- [ ] C2 `InPlaceOperator<T>::size()` → `rows()`: 2 files (`algebra/sfem_Operator.hpp:40,102,110`; `InPlaceLambdaOperator`). Composition then reads `left_->rows() == right_->rows()` throughout. `size()` is otherwise the buffer element count (1,585 `.size()` calls).
- [ ] C3 file-local op-data method `nelements()` → `n_elements()`: 2 declarations and 5 calls in 1 file (`frontend/cuda/sfem_Function_incore_cuda.cpp:302,354`). It is the only abbreviated size method in C++.

**Considered, no change:**
- `Op::n_dofs_domain()`/`n_dofs_image()` (`frontend/ops/sfem_Op.hpp:488`, 114 declarations/102 files, mostly generated): these already have the `n_<entity>_<qualifier>` shape. `Op` is a form on a `FunctionSpace`, not a matrix, so it keeps dof words rather than `rows`/`cols`.
- The legacy C struct `mesh_t` (`mesh/mesh_utils.hpp:20-49`: `nelements`, `nnodes`, `n_owned_nodes`, `n_global_nodes`; 60 occ/15 files) belongs in a retirement list (`DEAD.md`). Renaming it is not worth a commit.
- The C kernel parameters `nelements`/`nnodes` (25,573 and 13,637 occ; 21,338 and 10,061 of them in `frontend/ops/generated`) are generator-owned kernel signatures. A rename would mean a full regeneration with no API gain.

## 3. Abbreviations

**Convention.** Identifiers spell words out. An acronym is allowed only when it is the published name of the method or storage format: MPRGP, BiCGStab, BSR, CRS, COO, DIA, SELL, LOR, SSMGC. Invented contractions are not (`bcgs`, `cheb3`, `cg` in file names, `es`). A file that defines a class is named after the class (group 4).

- [ ] A1 parameter `es` → `execution_space`: 179 occ/80 files; 109/78 already spell it out. **Defer:** tick only if the ExecutionContext work does not remove the parameter altogether.
- [ ] A2 member `es_` → `execution_space_`: 49/7, against 118/32 that already spell it out. The accessor is `execution_space()` everywhere (253 calls).
- [ ] A3 `algebra/sfem_cg.hpp` → `algebra/sfem_ConjugateGradient.hpp`: 12 includes. The class inside is `ConjugateGradient`.
- [ ] A4 `algebra/sfem_bcgs.hpp` → `algebra/sfem_BiCGStab.hpp`: 12 includes. The class is `BiCGStab`.
- [ ] A5 `algebra/sfem_mprgp.hpp`, `sfem_mprgp_bounds.hpp`, `sfem_mprgp_control.hpp` → `sfem_MPRGP.hpp`, `sfem_MPRGPBounds.hpp`, `sfem_MPRGPControl.hpp`: 7, 2 and 3 includes. The classes are `MPRGP`, `MPRGPBounds` and `MPRGPControl`.
- [ ] A6 `algebra/parallel/sfem_parallel_cg.hpp` → `sfem_ParallelConjugateGradient.hpp`: 1 include.
- [ ] A7 `algebra/sfem_tpl_blas.hpp` → `algebra/sfem_BLAS.hpp`: 7 includes. The file defines class `BLAS`; `tpl` describes nothing.
- [ ] A8 host-callable device functions `d_*` → `cu_*`: 48 distinct names, 175 occ/32 files (`d_axpy`, `d_axpby`, `d_dot`, `d_nrm2`, `d_memcpy`, `d_memset`, `d_cg`, `d_bcgs`, `d_cheb3`, `d_bsr_spmv*`, `d_crs_spmv`, `d_buffer_alloc`, `d_ediv`, `d_max`, ...). The other 529 such functions (6,381 occ) use `cu_`, so one prefix would mean "launches CUDA work". Where the old name carries a contraction it is spelled out: `d_cg` → `cu_conjugate_gradient`, `d_bcgs` → `cu_bicgstab`, `d_cheb3` → `cu_chebyshev3`.

## 4. Backend classes and files

**Convention, classes.** The backend implementation of concept `X` is `<Backend>X`, in CamelCase with no underscore. `OpenMP` marks host-threaded code and `GPU` marks device code; the GPU prefix is already used 1,016 times (`GPULaplacian`, `GPUDirichletConditions`, `GPUGenerated*`). A table of backend kernels (a struct of `std::function` that a backend fills) is `XKernels`.

**Convention, files.** A file that defines a class is `sfem_<Class>.{hpp,cpp}`. A header of free functions or aliases is `sfem_<snake_topic>.hpp`; `sfem_aliases.hpp`, `sfem_context.hpp`, `sfem_mask.hpp` and `sfem_fixed_chunks.hpp` already conform. The backend counterpart of a host file sits in the backend folder with a backend suffix: `cuda/sfem_<Name>_cuda.{hpp,cu}` (39 files already follow this, including every generated op) or `openmp/sfem_<Name>_openmp.hpp`. An executable source is `<name>.exe.cpp`.

- [ ] B1 `CUDA_BLAS` → `GPUBLAS` (17/2) and `OpenMP_BLAS` → `OpenMPBLAS` (4/3).
- [ ] B2 `CG_Tpl` → `ConjugateGradientKernels` (6/5), `OpenMP_CG` → `OpenMPConjugateGradientKernels` (3/3), `CUDA_CG` → `GPUConjugateGradientKernels` (5/3) (`algebra/openmp/sfem_openmp_cg_impl.hpp:18,49`, `algebra/cuda/sfem_cuda_cg_impl.hpp:8`).
- [ ] B3 `MPRGP_Tpl` → `MPRGPKernels` (9/6), `OpenMP_MPRGP` → `OpenMPMPRGPKernels` (4/3), `CUDA_MPRGP` → `GPUMPRGPKernels` (6/4).
- [ ] B4 `ShiftedPenalty_Tpl` → `ShiftedPenaltyKernels` (8/6), `OpenMP_ShiftedPenalty` → `OpenMPShiftedPenaltyKernels` (6/4), `CUDA_ShiftedPenalty` → `GPUShiftedPenaltyKernels` (7/5).
- [ ] B5 `ShiftableBlockSymJacobi_Tpl` → `ShiftableBlockSymJacobiKernels` (7/6), `ShiftableBlockSymJacobi_OpenMP` → `OpenMPShiftableBlockSymJacobiKernels` (6/4), `ShiftableBlockSymJacobi_CUDA` → `GPUShiftableBlockSymJacobiKernels` (9/3). Also the file-local helpers `Invert3_OpenMP` → `OpenMPInvert3` and `Mask3_OpenMP` → `OpenMPMask3` (1 file). The SoA variants are in `DEAD.md`.
- [ ] B6 `algebra/cuda/sfem_cuda_<x>.*` and `algebra/bsr/cuda/sfem_cuda_BSR.*` (17 files) → `sfem_<Class>_cuda.*`. That gives `sfem_BLAS_cuda` (29 includes), `sfem_ConjugateGradientKernels_cuda` (2), `sfem_MPRGPKernels_cuda` (3), `sfem_ShiftedPenaltyKernels_cuda` (4), `sfem_ShiftableBlockSymJacobiKernels_cuda` (2), `sfem_crs_spmv_cuda` (5), `sfem_max_cuda` (3), `sfem_solver_cuda` (11) and `sfem_BSR_cuda` (4). Today three orders coexist: `sfem_cuda_X` (19 files), `sfem_X_cuda` (39) and `cu_X` (69).
- [ ] B7 `algebra/openmp/sfem_openmp_<x>.hpp` (5 live files) and `sfem_ShiftedPenalty_impl.hpp` → `sfem_BLAS_openmp` (22 includes), `sfem_ConjugateGradientKernels_openmp` (3), `sfem_MPRGPKernels_openmp` (2), `sfem_ShiftableBlockSymJacobiKernels_openmp` (2) and `sfem_ShiftedPenaltyKernels_openmp`.
- [ ] B8 kernel-layer `cu_<x>.{cu,cuh,hpp}` (69 files, in `cuda/` folders under `operators/`, `resampling/` and `algebra/`) → `<x>_cuda.*`. This only matters with B6. The CMake globs are per folder, so no CMake edit is needed. It is the most expensive file item; leave it unticked unless the kernel layer is being touched anyway.
- [ ] B9 unprefixed library files → `sfem_` prefix: `algebra/lumped_ptdp.{hpp,cpp}` (2 includes), `algebra/inverse.cpp` (2), `algebra/multiblock_crs_graph.{hpp,cpp}` (1), `algebra/sortreduce.{hpp,cpp}` (50; see G2 first). The class files `algebra/crs/{scrs,acrs,sdacrs}.hpp` and `algebra/ell/sell.hpp` take the class names from H4-H6: `sfem_SplitCRS.hpp`, `sfem_AlignedCRS.hpp`, `sfem_SplitAlignedCRS.hpp`, `sfem_sell_spmv.hpp`.
- [ ] B10 drivers with a `main` but no `.exe` suffix → `.exe.cpp`. These are `drivers/steady_state_sim.cpp`, `mgsolve.cpp`, `hex8_fix_ordering.cpp`, `nitsche_flow.cpp`, `sym_spmv_bench.cpp`, `test_galerkin_assembly.cpp`, `drivers/simulations/wall_mounted_hump.cpp`, `drivers/verification/generated_stokes_fvca8.cpp` and `drivers/metis/partition_mesh_based_on_operator.cpp`: 9 files, against 60 `.exe.cpp`. CMake builds every `.cpp` in a driver folder under its `NAME_WE`, so the targets keep their names. Only `SFEM_EXCLUDED_EXEC` (`sym_spmv_bench`) and the scripts that call these binaries need checking.

**Considered, no change:**
- The `GPU*` classes in `sfem_*_cuda` files (`GPUGenerated*` in `sfem_Generated*_cuda.hpp`, 838 occ/44 files) already conform: the class carries the GPU prefix and the file the `_cuda` suffix.
- `sfem_aliases.hpp` and `sfem_context.hpp` are free-function and alias headers, which is what the lowercase form is for.
- The kernel-layer files under `operators/` (321, named `<element>_<operation>`) keep the element as their namespace.

## 5. Hessian format suffixes

**Convention.** `hessian_<format>[_sym][_<layout>]`, where `<format>` is the lowercase name of the matrix class that stores the result (`crs`, `bsr`, `coo`, `dia`, `sell`) and qualifiers are spelled out. BCRS and BSR are the same storage, block compressed rows. The library's class is `BSR` (`algebra/bsr/sfem_BSR.hpp`), and the format string `op_type::BSR_SYM` is already assembled by `hessian_bcrs_sym` (`frontend/sfem_API.hpp:1629`), so the word `bcrs` goes.

- [ ] H1 `hessian_bcrs_sym` → `hessian_bsr_sym`: 31/16 (`frontend/ops/sfem_Op.hpp:173`, `sfem_LinearOperatorFactory.cpp:108`, op implementations, codegen).
- [ ] H2 kernels `*_bcrs_sym` (`linear_elasticity_bcrs_sym`, `neohookean_ogden_bcrs_sym`, `hex8_neohookean_ogden[_active_strain]_bcrs_sym`, `hex8_mooney_rivlin_active_strain_bcrs_sym`, `h_bcrs_sym`) → `*_bsr_sym`: 24 occ. Kernel names follow H1.
- [ ] H3 class `BCRSSymSpMV` in `algebra/sfem_bcrs_sym_SpMV.hpp` → `BSRSymSpMV` in `sfem_BSRSymSpMV.hpp` (5/1 plus 1 include). Also `algebra/sfem_crs_sym_SpMV.hpp` → `sfem_CRSSymSpMV.hpp` (class `CRSSymSpMV`, 3 includes).
- [ ] H4 `hessian_scrs` / class `SCRS` → `hessian_split_crs` / `SplitCRS`: 2/1 and 7/1 (`sfem_LinearOperatorFactory.cpp:340`, `algebra/crs/scrs.hpp:73`).
- [ ] H5 `hessian_acrs` / class `ACRS` → `hessian_aligned_crs` / `AlignedCRS`: 2/1 and 7/1 (`sfem_LinearOperatorFactory.cpp:320`, `algebra/crs/acrs.hpp:23`).
- [ ] H6 `hessian_sdacrs` / class `SDACRS` → `hessian_split_aligned_crs` / `SplitAlignedCRS`: 2/1 and 7/1 (`sfem_LinearOperatorFactory.cpp:299`, `algebra/crs/sdacrs.hpp:74`). Please confirm what "DA" expands to: the class keeps a separate diagonal block (`diag_rowptr`, `diag_colidx`, `diag_values`).
- [ ] H7 format strings `"SPLITCRS"`, `"ALIGNEDCRS"`, `"SPLITDACRS"` (`base/sfem_defs.hpp:18-25`) → `"SPLIT_CRS"`, `"ALIGNED_CRS"`, `"SPLIT_ALIGNED_CRS"`, matching `CRS_SYM`/`BSR_SYM`/`COO_SYM`. These are user input (`solver.linear.format`, `SFEM_OPERATOR`), so: **alias**, the old strings are still accepted, with a warning.

**Considered, no change:**
- `hessian_bsr_typed` (113/59) keeps its own name. As an overload of the virtual `hessian_bsr`, it would be hidden in every derived class that overrides only one of the two.
- `hessian_block_diag_sym_soa` already has the shape: `soa` is a layout qualifier.
- `hessian_crs`, `hessian_crs_sym`, `hessian_coo_sym`, `hessian_dia`, `hessian_sell`, `hessian_diag` and `hessian_block_diag_sym` conform.

## 6. Verbs and word order

**Convention.** A function is `<verb>_<object>`. Moving an object between memory spaces is the overloaded free `to_device(x)`/`to_host(x)` (smesh's convention; 154 and 114 occ). A raw byte copy is `copy_host_to_device(nbytes, src, dst)` or `copy_device_to_host(nbytes, src, dst)`, source before destination. An operation that produces the same kind of derived object has one method name in every class that offers it. A noun accessor returns the `std::shared_ptr`.

- [ ] V1 `buffer_host_to_device(n, h, d)` / `buffer_device_to_host(n, d, h)` → `copy_host_to_device` / `copy_device_to_host`: 13/11 and 6/5 (`algebra/cuda/sfem_cuda_blas.hpp:20-21`). These copy bytes, not buffers.
- [ ] V2 `crs_graph_host_to_device` → `to_device(graph)`: 5/4. smesh already has this overload (`external/smesh/src/frontend/smesh_crs_graph.hpp:31`), so sfem's copy is a second path and goes.
- [ ] V3 `boundary_conditions_host_to_device` and `elements_to_device` → `to_device(...)` overloads: 4/4 and 2/2.
- [ ] V4 `Op::lor_op(space)` → `Op::lor(space)` and `Op::derefine_op(space)` → `Op::derefine(space)`: 69/38 and 177/98, mostly generated (`python/codegen/framework/package/op_wrappers.py`), declared at `frontend/ops/sfem_Op.hpp:451,458`. `FunctionSpace::lor()`/`derefine()` (`sfem_FunctionSpace.hpp:92-93`) and `Function::lor()`/`derefine()` (`sfem_Function.hpp:63-73`) already name the same operation without the suffix; the return type says it is an `Op`.
- [ ] V5 `FunctionSpace::mesh()` (returns `Mesh&`) plus `mesh_ptr()` (returns `shared_ptr`) → one `mesh()` returning `const std::shared_ptr<Mesh>&`. This takes two commits. First, the at most 144 `.mesh()`/`->mesh()` calls in 30 files (the count includes other classes' `mesh()`) become `mesh_ptr()->`. Then `mesh_ptr()` (704/117) is renamed to `mesh()`. `Function::space()` (288 calls) already returns the shared pointer under the noun.

**Considered, no change:**
- `write_crs(path, graph, values)` (`algebra/crs/sfem_CRS.cpp:9`) already conforms.
- `crs_write_folder`/`crs_read_folder` are smesh's C functions; renaming them belongs in smesh.
- `copy_to_host`/`copy_to_device` and `copy_quadrature_rule_to_device` conform.

## 7. Enums

**Convention.** An sfem enum is an `enum class`, nested in the class whose state it describes, with UPPER_SNAKE enumerators that carry no type prefix. `ElementScope::ALL`, `LineSearch::Rule::ARMIJO`, `ContactConditions::Gap::NODAL`, `LoadProfile::Type::LINEAR_RAMP` and `FunctionSpace::SemiStructuredBasis::MICRO` already do this. smesh's plain `ExecutionSpace`, `MemorySpace`, `ElemType` and `PrimitiveType` stay plain: they cross the C ABI as runtime type tags and belong to smesh. sfem stops writing them as `enum ExecutionSpace`.

- [ ] E1 `Multigrid::CycleType { V_CYCLE, W_CYCLE }` and `ShiftedPenaltyMultigrid::CycleType { V_CYCLE, ... }` → `enum class CycleType { V, W }`: 7/2 (`algebra/sfem_Multigrid.hpp:34`, `algebra/sfem_ShiftedPenaltyMultigrid.hpp:135`).
- [ ] E2 `CycleReturnCode { CYCLE_CONTINUE, CYCLE_CONVERGED, CYCLE_FAILURE }` → `enum class CycleResult { CONTINUE, CONVERGED, FAILED }`: 15/2 (`sfem_Multigrid.hpp:39`, `sfem_ShiftedPenaltyMultigrid.hpp:140`). The enum is declared identically in both classes; one shared declaration in `sfem_Multigrid.hpp` would do.
- [ ] E3 `NonlinearReport::Status` → `enum class`, enumerators unchanged (`NonlinearReport::Status::CONVERGED`): 43/6 (`frontend/nonlinear/sfem_NonlinearSolver.hpp:20`).
- [ ] E4 `LineSearchResult::Status` → `enum class`, enumerators unchanged: 31/4 (`frontend/nonlinear/sfem_LineSearch.hpp:34`). Its sibling `LineSearch::Rule` (`:17`) is already an enum class.
- [ ] E5 `MPRGP::ExpansionType { EXPANSION_TYPE_ORGINAL, EXPANSION_TYPE_PROJECTED_CG }` → `enum class Expansion { ORIGINAL, PROJECTED_CG }`: 5/2 (`algebra/sfem_mprgp.hpp:32`). This also fixes the `ORGINAL` typo.
- [ ] E6 free `MPRGPMove { MPRGP_MOVE_* }` and `MPRGPDecision { MPRGP_DECIDE_* }` → `enum class MPRGPControl::Move { STEP, EXPAND, UNBOUNDED }` and `MPRGPControl::Decision { NONE, CG, PROPORTIONING }` with `: int`: 31/5 (`algebra/sfem_mprgp_control.hpp:11,14`). The explicit `: int` keeps the values usable in device code.
- [ ] E7 `MPRGPRunResult::Exit { PROPORTIONING = 1, NOT_A_STEP = 2, DONE = 3 }` (`algebra/openmp/sfem_openmp_mprgp_impl.hpp:46`) and `MPRGP::TeamExit` (`algebra/sfem_mprgp.hpp:296`, already an enum class with the same values) → one shared `enum class MPRGPExit`. The two are twins.
- [ ] E8 `typedef enum { SFEM_ACCELERATOR_TYPE_* } AcceleratorsType` → `enum class Accelerator { CPU, CUDA, OPENCL, OPENACC, HIP }`: 19/4 (`base/sfem_defs.hpp:35`). If nothing outside its 3 files uses it, retire it instead.
- [ ] E9 `typedef enum { ADJOINT_BASE, ADJOINT_REFINE_* } AdjointRefineType` → `enum class AdjointRefine { BASE, ONE_STEP, ITERATIVE, HYTEG_REFINEMENT }`: 24/4 (`base/sfem_defs.hpp:43`, resampling).
- [ ] E10 elaborated `enum ExecutionSpace` / `enum MemorySpace` in sfem → `ExecutionSpace` / `MemorySpace`: 42/19 and 3/3. The elaborated form is C syntax and hides that these are smesh types.

## 8. Return codes for unimplemented virtuals

**Rule.** There are two codes, `SFEM_SUCCESS` and `SFEM_FAILURE`. A base-class default of a virtual follows one of three cases:
- (a) It returns `SFEM_SUCCESS` when doing nothing is correct. That covers optional hooks (`Op::initialize`/`update`/`report`, `LinearizedStep::begin_step`/`end_step`, the `Stepper`/`Observer` callbacks) and contributions that are exactly zero. Example: `NeumannConditions::hessian_block_diag_sym` (`frontend/sfem_NeumannConditions.hpp:104`), because a dead load has no Hessian. Each such default gets a one-line comment saying why nothing is the right answer.
- (b) It returns `SFEM_FAILURE`, without aborting, when the class lacks the capability (`Op::hessian_*` formats, `inexact_update`/`inexact_apply` at `frontend/ops/sfem_Op.hpp:111,123`). The caller can then fall back, as `create_linear_operator` already falls back to CRS. The caller that cannot fall back is the one that reports and aborts.
- (c) `SFEM_ERROR` (abort) is reserved for broken invariants.

- [ ] R1 the 13 `Op::hessian_*` defaults that call `SFEM_ERROR(...)` and then `return SFEM_FAILURE` (`frontend/ops/sfem_Op.hpp:136-262`; the return is unreachable because `SFEM_ERROR` aborts) → `return SFEM_FAILURE` with no abort (rule b).
- [ ] R2 the zero-contribution defaults (`NeumannConditions::hessian_block_diag_sym`; the `NoOp` gradient/apply/value at `sfem_Op.hpp:524-534`) stay `SFEM_SUCCESS` and gain the comment (rule a).
- [ ] R3 `SFEM_OK` (`base/sfem_base.hpp:102`, defined, used nowhere else) → removed in favour of `SFEM_SUCCESS`.
- [ ] R4 `frontend/ops/generated/kernel_diagnostics.hpp:11` redefines `SFEM_SUCCESS` under `#ifndef` → include `sfem_base.hpp` instead. This is a generator change.

## 9. Global C-style utilities

**Rule.** A C-linkage function (`extern "C"`, no namespace) carries the `sfem_` prefix; `sfem_abort` (`base/sfem_base.hpp:108`) already does. Anything called only from C++ moves into `namespace sfem` and drops `extern "C"`. The kernel-layer C functions (`<element>_<operation>`) are out of scope: the element is their namespace.

- [ ] G1 `logger_t` with `log_init`, `log_is_empty`, `log_write_double`, `log_create_memstream`, `log_create_file`, `log_destroy` (`base/sfem_logger.hpp:11-23`, global names) → `sfem::Logger` with `open_file(path, mode)`, `open_memstream()`, `write(double)`, `empty()` and a destructor: 16/5 and 27/5. Its only users are C++ (`frontend/sfem_Function.cpp`, `drivers/taylor_hood_navier_stokes.exe.cpp`, `examples/run_convection_diffusion.cpp`).
- [ ] G2 `sortreduce`, `psortreduce`, `sort_idx`, `find_idx_binary_search` (`algebra/sortreduce.hpp`, global, `extern "C"`): 65/52, 3/3 and 31/31. smesh has `find_idx` and its own `sortreduce` (`external/smesh/src/graph/smesh_graph.hpp:216`). Use smesh's and delete sfem's, since two paths do the same operation. If they differ, move sfem's into `namespace sfem`.
- [ ] G3 `lumped_ptdp_crs`, `lumped_ptdp_crs_v` (`algebra/lumped_ptdp.hpp`, global, 4/2) → `sfem::lumped_galerkin_crs`, `sfem::lumped_galerkin_crs_vector`. The name should describe the operation, a lumped Galerkin product P^T D P, rather than its formula.
- [ ] G4 `dinvert1..4` (`algebra/inverse.cpp`, global; 8/2, generated by `python/codegen/inverse.py`, used by `drivers/bgs.exe.cpp` and `solver/constrained_gs.cpp`) → `sfem::invert_node_blocks_<N>`.

## 10. Typos

- [ ] T1 `create_mulitbody_contact` → `create_multibody_contact`: 3/3 (`frontend/contact/sfem_SelfContact.hpp:43`).
- [ ] T2 environment variable `SFEM_HYPERELAStICITY_USE_AOS` → `SFEM_HYPERELASTICITY_USE_AOS` (`frontend/ops/sfem_Hyperelasticity.cpp:833`). Nobody can set it by its correct spelling today. **Alias:** the misspelt name is still read, with a warning, for one release.
- [ ] T3 `frontend/tests/sfem_MatrixFromatsTest.cpp` → `sfem_MatrixFormatsTest.cpp`.
- [ ] T4 `cmake/SFEMOptions.cmake:47` "Eneable ... (Override with SFEM_TRACE_FILE)" → "Enable ... (override with SMESH_TRACE_FILE)". Nothing reads `SFEM_TRACE_FILE`; the tracer reads `SMESH_TRACE_FILE` (`external/smesh/src/profile/smesh_tracer.cpp:28`).

---
Items: 22 factories, 3 sizes, 8 abbreviations, 10 backend/file, 7 hessian, 5 verbs, 10 enums, 4 return codes, 4 C utilities, 4 typos: **77**.
