# Dead code: removal checklist

Tick `[x]` to approve removing an item. Leave it unticked to keep it. Edit the line to add a note.
After approval, the ticked items under each heading are removed together, one commit per heading.
Evidence verified with grep/find on commit `717c5cfbf` (branch tight-api; build*/, external/, venv, .git and wip/ excluded).
"N lines, 0 code" means every line is a comment or blank. "Not compiled" means no SFEM_MODULES, SFEM_DRIVER_MODULES, unit-test or per-folder CMake entry globs the file.

## Empty or fully commented-out files

- [ ] `frontend/ops/sfem_PackedLaplacian_trials.cpp`: old packed-Laplacian trials. 645 lines, 0 code, still compiled (frontend/ops module); the name is referenced nowhere.
- [ ] `operators/tet4/tet4_viscous_power_density_curnier.{cpp,hpp}`: viscous power density (Curnier). The cpp has 1377 lines, 0 code. Nothing includes the hpp, and its symbol `tet4_viscous_power_density_curiner_assemble_hessian` appears nowhere else.
- [ ] `operators/sshex8/sshex8_neohookean_ogden_active_strain.{cpp,hpp}`: 753 + 111 lines, 0 code. Nothing includes the header.
- [ ] `operators/macro_tet4/macro_tet4_laplacian_simd.cpp`: 606 lines, 0 code. The name is referenced nowhere.
- [ ] `operators/sstet4/sstet4_sampling.{cpp,hpp}`: 442 + 22 lines, 0 code. Only its own cpp includes the header.
- [ ] `operators/boundary_conditions/neumann.{cpp,hpp}`: 455 + 34 lines, 0 code. `#include "neumann.hpp"` still appears in 13 files: drivers stokes, stokes_check, taylor_hood_navier_stokes, cvfem_assemble; frontend sfem_ContactConditions, sfem_Function, sfem_NeumannConditions; boundary_condition{,_io}.cpp; examples/*. Those lines go too, and they bring in nothing.
- [ ] `mesh/sfem_hex8_mesh_graph.cpp`: 1637 lines, 0 code. The name is referenced nowhere.
- [ ] `mesh/sfem_clustering.{cpp,hpp}`: 370 + 74 lines, 0 code. Only its own cpp includes the header.
- [ ] `mesh/sfem_sshex8_skin.{cpp,hpp}`: 231 + 31 lines, 0 code. The only other reference is a commented-out include at `frontend/sfem_ContactSurface.cpp:6`.
- [ ] `algebra/multiblock_crs_graph.{cpp,hpp}`: 289 + 50 lines, 0 code. The live version is `smesh::create_multiblock_crs_graph` (`frontend/sfem_FunctionSpace.cpp:208`).
- [ ] `algebra/sfem_mask.cpp`: 22 lines, 0 code. `sfem_mask.hpp` stays (15 includers).
- [ ] `drivers/contact/self_contact.exe.cpp`: 521 lines. The only live line is `int main() { return 0; }`, so it builds a no-op exe. No script uses it.
- [ ] `drivers/mesh_self_intersect.exe.cpp`: a 1-line stub `int main() { return 0; }` that builds a no-op exe. No script uses it.
- [ ] Zero-byte C++: `algebra/openmp/sfem_openmp_blas.cpp` (the hpp stays), `frontend/sfem_SelfContact.cpp`, `operators/tet4/tet4_neohookean_active_strain.{cpp,hpp}` (that name is referenced nowhere).
- [ ] `python/sfem/__main__.py`: zero bytes, so `python -m sfem` does nothing.
- [ ] `frontend/sfem_SelfContact.hpp`: a 6-line forwarder to `contact/sfem_SelfContact.hpp`. Both includers still work without it: `frontend/contact/sfem_MaMAL.cpp:10` finds the header in its own folder, and `frontend/distances/tests/sfem_TwoBodyContactTest.exe.cpp:17` is built only with SSDF, which puts frontend/contact on the include path.
- [ ] `frontend/distances/sfem_MeshDistances.{cpp,hpp}`: declares an empty `class MeshDistances {}`. The cpp holds only two includes, and `MeshDistances` appears nowhere else.
- [ ] `resampling/data_structures/sfem_queue.hpp`: an include guard and nothing else. No includer.

## Not compiled

- [ ] `operators/tet20/` (`tet20_laplacian.{cpp,hpp}`): not in SFEM_MODULES. Nothing outside the folder uses the `tet20_laplacian_*` symbols. Keep `python/codegen/tet20.py`: `laplace_op.py` and 11 other codegen scripts import it (see the codegen section).
- [ ] `operators/hex8/dg/`, `operators/quad4/dg/`, `operators/spectral_hex/dg/`: their module entries are commented out at `CMakeLists.txt:91-96` ("FIXME DG is yet to be implemented"). Nothing outside these folders includes them. `dg_hex8_symmetric_interior_penalty.hpp` is zero bytes, `dg_quad4_symmetric_interior_penalty_inline.hpp` has no includer at all, and `spectral_hex_lax_friedrichs_flux.cpp` is a single include.
- [ ] `operators/tet4/drafts/tet4_cuda_laplacian{,_2,_3}.cu`: operators/tet4 is globbed non-recursively, so these never build, and the name is referenced nowhere. `_2.cu` also holds a 326-line `#if 0` (353-678).
- [ ] `drivers/metis/partition_mesh_based_on_operator.cpp`: `metis` is missing from SFEM_DRIVER_MODULES (`CMakeLists.txt:351`). Its only caller, `workflows/decomposition/partition_mesh_based_on_operator.sh:33`, is therefore already broken (the call in `test/test.sh:46` is commented out).
- [ ] `examples/` (run_poisson.cpp, run_poisson_cuda.cpp, run_convection_diffusion.cpp and 3 .sh): the driver glob covers `drivers/examples`, which does not exist, and nothing globs top-level `examples/`. Only its own .sh files use these.
- [ ] `viz/` (`quad4_viz.exe.cpp`, `run.sh`): in no CMake list. Only its own run.sh uses it.
- [ ] `cuda/shuffles/width.cu`, `cuda/tensorcore/{mma_fp64_test.cu,mma_fp64_test.py,wmma_example.cu}`: standalone micro-tests. No CMake or script references them.
- [ ] `resampling/test_grid/`: a standalone experiment built by its own Makefile (`CC = g++-13`), outside CMake. It also commits `perf.data` and `perf.data.old`.
- [ ] `notebook/symbtrafo/main.cpp`, `matlab/projectedSSOR.cpp`: loose sources that no build or script references.
- [ ] CMake entries for directories that do not exist: SFEM_MODULES `operators/hierarchical` (`:69`) and `graphs` (`:142`, METIS); SFEM_UNIT_TESTS `operators/ssquad4/tests`, `operators/sshex8/tests`, `mesh/tests` (`:450-453`); SFEM_DRIVER_MODULES `mesh examples mesh_functions sdf` (`:351`).
- [ ] The cuSPARSE and cuBLAS branches: `SFEM_ENABLE_CUSPARSE` (`configuration/sfem_config.h.in:10`) and `SFEM_ENABLE_CUBLAS` have no CMake option, so both are undefined in every build (the Alps CUDA build's `sfem_config.h` has `/* #undef SFEM_ENABLE_CUSPARSE */`). Never compiled: `algebra/cuda/sfem_cuda_crs_SpMV.cpp:13-489` (the cuSPARSE CRS, symmetric-COO and BSR SpMV, with a static handle and no stream) and `algebra/cuda/sfem_cuda_blas.cu:29-215` plus the `#ifdef` arms at 565 and 626 (cuBLAS). What is compiled instead: `d_bsr_spmv` runs the library's kernel; `d_crs_spmv` and `d_sym_coo_spmv` are `SFEM_ERROR("Not implemented!")` stubs, reached from `frontend/sfem_LinearOperatorFactory.cpp:417,452`, `drivers/bench/bench_spmv.exe.cpp` and `drivers/sym_spmv_bench.cpp`. Removing the branches also removes the `#warning "No CUSPARSE installation!"` every CUDA build prints. The alternative, an option that turns them on, is Stage H.
- [ ] Stale SFEM_EXCLUDED_EXEC names: `heat_equation` (listed twice, `:364`), `lapl_matrix_free` and `linear_elasticity_matrix_free` (`:387`) have no source. `run_poisson_cuda` lives only in the unglobbed examples/.

## Backups and deprecated copies

- [ ] `resampling/deprecated/` (`sfem_resample_field_V4.c_dep`, `sfem_resample_field_V8.c_dep`): the `.c_dep` extension is never globbed, and neither name is referenced.
- [ ] `resampling/tet10/deprecated_bk/` (`tet10_resample_field_DEP.{c,h}_dep`): same.
- [ ] `python/codegen/framework/retired/` (9 docs, about 210 KB): superseded design documents. Its own README says they are kept for their reasoning. They are cited by `framework/README.md:43`, `ARCHITECTURE.html` and a docstring in `tests/test_plans_are_consumed.py`. Tick only if you no longer want them.

## Large `#if 0` blocks

Each line removes only the dead `#if 0` branch. Where an `#else` branch exists it stays, with its guards dropped.

- [ ] `operators/tet4/tet4_neohookean_ogden.cpp:423-437`: "Old implementation" of the hessian apply. The `#else` branch at 438-478 stays.
- [ ] `algebra/crs/sfem_CRS.hpp:365-380`: plain CRS SpMV loop. The `#else` branch at 381-419 ("20-27% faster on M1") stays.
- [ ] `operators/tet10/tet10_laplacian_inline_cpu.hpp:43-67`: reference-gradient loop. The `#else` branch at 68-91 stays.
- [ ] `operators/hex8/hex8_neohookean_ogden_active_strain.cpp:637-647` (centroid sampling) and `:731-738` ("Slower than other variant"). The `#else` branches at 648-682 and 739-748 stay.
- [ ] `operators/hex8/hex8_neohookean_ogden.cpp:334-344` (centroid sampling) and `:426-433` ("Slower than other variant"). The `#else` branches at 345-377 and 434-443 stay.
- [ ] `operators/tet10/tet10_neohookean_ogden.cpp:389-398` (centroid sampling) and `:461-468` ("Slower than other variant"). The `#else` branches at 399-411 and 469-478 stay.
- [ ] `operators/sshex8/sshex8_interpolate.cpp`: eight debug `printf` dumps, 12 lines each, at 319-330, 357-368, 389-400, 421-432, 459-470, 497-508, 535-546 and 575-586.

## Headers nothing includes

No `#include` of the basename exists anywhere in C/C++/CUDA/CMake/python/bindings.

- [ ] `operators/tet4/tet4_partial_assembly_pa_inline.hpp` (636 lines)
- [ ] `operators/tet4/tet4_neohookean_ogden_active_strain_local.hpp` (925 lines)
- [ ] `operators/tet4/tet4_partial_assembly_neohookean_ogden_active_strain_inline.hpp` (1359 lines)
- [ ] `operators/tet4/cuda/cu_tet4_linear_elasticity_inline.hpp` (90 lines)
- [ ] `operators/hex8/hex8_mooney_rivlin_visco_unique_Hi_standard.hpp` (4523 lines): the only include, at `hex8_mooney_rivlin_visco_flexible.cpp:19`, is commented out. It is output of `sr_visco_hyper_unique_Hi.py`, whose live sibling `_unique_Hi_local.hpp` stays.
- [ ] `operators/stencil/sshex8_skeleton_offdiag_stencil.hpp` (1718 lines): the only include, at `operators/sshex8/sshex8_laplacian.cpp:8`, is commented out.
- [ ] `algebra/openmp/sfem_openmp_ShiftableJacobi_SoA.hpp` (231 lines): the name `ShiftableJacobi_SoA` appears nowhere else.
- [ ] `algebra/sfem_MPIType.hpp` (31 lines): `MPIType` appears nowhere else.
- [ ] `algebra/sfem_SparseBlockVector.hpp` (22 lines): a stale duplicate of `SparseBlockVector`. The live class is in `algebra/sfem_MatrixFreeLinearSolver.hpp:26`.
- [ ] `mesh/point_triangle_distance.hpp` (202 lines): `point_triangle_distance` appears nowhere else.
- [ ] `resampling/rule35.{hpp,csv}`, `resampling/rule56.{hpp,csv}`: `rule35` and `rule56` appear nowhere else.

## Functions with no caller

Each name appears only at its own declaration and definition, checked across C/C++/CUDA, python/bindings/pysfem.cpp, drivers/, spikes/, workflows/ and tests.

- [ ] `operators/tet4/cuda/cu_tet4_fff.{cu,hpp}` (`cu_tet4_fff_allocate`, `cu_tet4_fff_fill`, `elements_to_device`): the whole file is dead. Also drop the unused include at `frontend/cuda/sfem_Function_incore_cuda.cpp:34`.
- [ ] `operators/hex8/cuda/cu_hex8_fff.{cu,hpp}` (`cu_hex8_fff_allocate`, `cu_hex8_fff_fill`): the whole file is dead. Also drop the include at `sfem_Function_incore_cuda.cpp:28`.
- [ ] `operators/tet4/cuda/cu_tet4_adjugate.{cu,hpp}` (`cu_tet4_adjugate_allocate`, `cu_tet4_adjugate_fill`): the whole file is dead. Also drop the include at `sfem_Function_incore_cuda.cpp:33`.
- [ ] `operators/hex8/cuda/cu_hex8_adjugate.{cu,hpp}` (`cu_hex8_adjugate_allocate`, `cu_hex8_adjugate_fill`): the whole file is dead. Also drop the include at `sfem_Function_incore_cuda.cpp:27`.
- [ ] `operators/tet4/tet4_adjugate.{cpp,hpp}` (`tet4_adjugate_{fill,create,destroy}`, `tet4_adjugate_t`): nothing outside the pair uses them.
- [ ] `operators/hex8/hex8_jacobian.{cpp,hpp}`: its only function, `hex8_adjugate_and_det_fill`, has no caller. Also drop the unused includes in `frontend/sfem_DirichletConditions.cpp`, `sfem_Function.cpp` and `sfem_NeumannConditions.cpp`.
- [ ] `operators/sstet4/cuda/` (`cu_sstet4_laplacian.{cu,hpp}`, `cu_sstet4_laplacian_apply`): the whole module is dead. Also drop `CMakeLists.txt:166`.
- [ ] `operators/spectral_hex/spectral_hex_mass.{cpp,hpp}` (`spectral_hex_mass_apply`, `lagrange_hex_mass_apply`): only its own cpp includes the header.
- [ ] `operators/quadshell4/cuda/cu_quadshell4_integrate_values.{cu,hpp}` (`cu_quadshell4_integrate_value`, `cu_quadshell4_integrate_values`): only its own cu includes the header.
- [ ] Helpers with no caller: `create_hierarchical_restriction_from_graph` (`frontend/sfem_Transfer.cpp:477-526`), `create_hierarchical_prolongation_from_graph` (`frontend/sfem_Transfer.cpp:528-583`), `make_linear_op_variant` (`frontend/sfem_LinearOperatorFactory.cpp:393-401`), `sshex8_derefine_element_connectivity` (`frontend/sfem_API.hpp:123-186`), `create_skin_sideset` (`frontend/sfem_API.hpp:189-191`), `create_zero_constraints_op` (`frontend/sfem_LinearOperatorFactory.cpp:619-622`). `sshex8_derefine_element_connectivity` is superseded by `smesh::derefine_elements` and `smesh::sshex8_device_elements_view`.
- [ ] `frontend/sfem_API.hpp:91-121`: four commented-out `create_buffer` overloads.

## Environment-driven factories (create_from_env) and their remaining users

- [ ] `ContactConditions::create_from_env` (`frontend/sfem_ContactConditions.cpp:584`, `.hpp:87`): no caller.
- [ ] `NeumannConditions::create_from_env` (`frontend/sfem_NeumannConditions.cpp:289`, `.hpp:48`): its only user is `drivers/assemble.exe.cpp:40`, which scripts still use (see "Checked, still used"), so assemble.exe must be migrated first.
- [ ] `DirichletConditions::create_from_env` (`frontend/sfem_DirichletConditions.cpp:283`, `.hpp:47`): used by `drivers/assemble.exe.cpp:37` and by `create_dirichlet_conditions_from_env` (`frontend/sfem_DirichletConditions.cpp:781`). That wrapper is called by drivers obstacle.exe, sdf_obstacle.exe, test_galerkin_assembly, mgsolve and steady_state_sim, which must be migrated first.
- [ ] `AxisAlignedContactConditions::create_from_env` (`frontend/sfem_ContactConditions.cpp:162`, `.hpp:18`), called through `create_contact_conditions_from_env` (`frontend/sfem_ContactConditions.cpp:984`): the only caller is `drivers/obstacle.exe.cpp`.
- For information only: `smesh::SFC::create_from_env` lives in smesh. It is used by 7 drivers/bench, `bench/cuda/bench_packed_laplacian.cpp`, `drivers/simulations/poisson.exe.cpp`, `frontend/tests/sfem_MatrixFromatsTest.cpp` and 5 spike drivers.

## Legacy implementations and drivers

- [ ] Legacy tet4 NeoHookean assembly: `neohookean_assemble_{hessian,gradient,value,hessian_soa}` in `operators/tet4/tet4_neohookean.cpp`, declared in `.hpp`. None has a caller; the only call is commented out at `drivers/assemble3.exe.cpp:102`. `neohookean_assemble_hessian` is defined twice (`tet4_neohookean.cpp:921` and `vectorized/tet4_simd_neohookean.cpp:901`), and both compile under the default `SFEM_ENABLE_EXPLICIT_VECTORIZATION=ON`. Remove the 4 functions and the whole `tet4_simd_neohookean.cpp`, whose name is referenced nowhere. The stress and von Mises functions stay: drivers cauchy_stress, vonmises and cprincipal_stresses use them.
- [ ] `drivers/assemble4.exe.cpp` together with `operators/tet4/tet4_isotropic_phasefield_for_fracture.cpp` and `operators/isotropic_phasefield_for_fracture.hpp`: no script or doc names assemble4, and only assemble4 uses the phase-field code. Also drop `assemble4` from `CMakeLists.txt:364`.
- [ ] `drivers/bgs.exe.cpp`: no script or doc reference.
- [ ] `drivers/crs_apply_dirichlet.exe.cpp`: no script or doc reference.
- [ ] `drivers/sym_spmv_bench.cpp`: referenced only by the exclusion list at `CMakeLists.txt:387`.
- [ ] The dynamic plug-in path: `operators/generated/CMakeLists.txt`, `frontend/ops/sfem_PlugInOp.{cpp,hpp}` and the `"plugin:"` branch at `frontend/ops/sfem_OpFactory.cpp:131-136`. Its `add_subdirectory` is commented out (`CMakeLists.txt:339`), the CMake globs a `tpl/hyperelasticity` directory that does not exist at the repo root, and no `plugin:` op name is used anywhere.

## Legacy codegen scripts (python/codegen/*.py, outside framework/)

The framework imports none of these. Its `fields`, `linear_elasticity`, `navier_stokes` and `stokes` imports resolve to its own `symbolic/`, `materials/` and `generators/` modules. No CMake or shell script runs any of them, apart from the dead `operators/generated` CMake.

- [ ] 75 leaf scripts that nothing imports and nothing names as `X.py`: PhaseField boundary_mass bspline bspline5 cdiv cgrad cgrad_op clagrangian_strain cu_ssquad4_interp cu_ssquad4_restrict cvfem_quad4_convection cvfem_quad4_laplacian cvfem_tri3_convection div_op dsfe ediv fe_taylor_expansion gpu_lapl gpu_lapl_apply gpu_linear_kv_integral_cuda_op grad_and_project_op hyperelasticity inverse isotropic_phase_field isotropic_phase_field_generic lagrange_legendre_gauss_lobatto lapl linear_ela_test_op linear_elasticity_op linear_kv_op lumped_boundary_mass lumped_mass macro_simplex macro_tet4 macro_tri3 make_fe mass neohookean nitsche_op p0_to_p1 p0_u_dot_grad_q p1_u_dot_grad_q parallel_eval phase_field_for_fracture_op projection_op proteus3D quad4_cvfem_flow quadrature_laplace_op rigid_body_modes simple_linear_elasticity sip_dg spectral_hex sshex8_resampling sstet4 sstet4_to_hex8_resampling stokes2 surface_integral_element surface_outflux surface_outflux_op surface_p0_to_p1 test_stencil tet10_lapl tet4_cvfem_flow tet4_lapl tri10 tri3_cvfem_flow u_dot_grad_q_op voigt_mapping volume wssmag. Five more appear only in prose, never imported: linear_elasticity navier_stokes stokes, gpu_linear_elasticity_op (in `framework/retired/PRD.md`) and cvfem_tet4_convection (in `spikes/cvfem/python/README.md`, which says the spike does not use it).
- [ ] 16 helper modules imported only by the scripts in the previous item: PhaseFieldBase bubble div fe_material fields lagrange_triangle mass_op mini phase_field_for_fracture_op_base quadrature real1 shear symbolic_fe tri1 viscous_power_density_curnier vonmises.
- [ ] `sr_hyperelasticity.py`, `pa_hyperelasticity.py`, `neohookean_partial_assembly.py`, `tpl_hyperelasticity.py` (zero bytes) and `tpl/hyperelasticity/symbolic_regression.pdf` (396 KB, the only file in tpl/): their only consumer is the dead `operators/generated/CMakeLists.txt`.
- [ ] `franetg/` (`__init__.py`, `franetg_gen.py`, `sfem_symbolic.py`), `next/looper.py`, `intersect/tri3_line2.py`: nothing outside their own folder references them.

## Stray artifacts in source folders

- [ ] `resampling/test_grid/perf.data`, `perf.data.old` (tracked, 28 KB each). They are already gone if the test_grid/ item above is ticked.
- [ ] `smesh.trace.csv` (repo root) and `spikes/inexact_apply_compare/smesh.trace.csv`: tracked trace output.
- [ ] `spmg_stats.csv` (repo root): tracked runtime output, written to `./spmg_stats.csv` by default (`algebra/sfem_ShiftedPenaltyMultigrid.hpp:1157`).
- [ ] `docs/partial_assembly.aux`: tracked LaTeX build product of `docs/partial_assembly.tex`.

## Checked, still used (not for removal)

- `drivers/assemble.exe.cpp` is run by `benchmark/initialize.sh:83,150` and `gpu_test/test_assemble.sh:36`.
- `drivers/assemble3.exe.cpp` is run by `benchmark/initialize.sh:84,151` and `tests/macro_fem/macro_fem_elast.sh:56-57`.
- `drivers/cvfem_assemble.exe.cpp` is run by `tests/cvfem3/cvfem3.sh:33` and `data/benchmarks/4_rectangle/run_cvfem_rectangle.sh:28`.
- `drivers/lapl.exe.cpp` is run by `gpu_test/test_apply.sh:39` and `data/benchmarks/1_darcy_cube/post_process.sh:29`. `laplacian_apply.exe` is run by `tests/hex8_laplacian/hex8_laplacian.sh`, so both stay until those scripts are migrated.
- `operators/tet4/vectorized/vtet4_laplacian.*` is used by `tet4_laplacian.cpp`; only `tet4_simd_neohookean.cpp` is listed above.
- `operators/generic/generic_hyperelasticity.cpp` is a 1-line include of a header-only file, which `frontend/ops/sfem_Hyperelasticity.cpp` uses.
- Codegen keep set:
  - `sr_visco_hyper_unique_Hi.py` generates the live `hex8_mooney_rivlin_visco_unique_Hi_local.hpp` and is cited in `workflows/prony_torsion/SPIKE.md:418`.
  - `laplace_op.py` is named as the generator in the headers of the live tri3, tri6 and tet10 laplacians.
  - Their import closure stays with them: aahex8 canon edge2 fe hex27 hex8 lagrange quad4 sfem_codegen stensor tet10 tet20 tet4 tri3 tri6 weighted_fe.
- `benchmark/baseline.cu` is built by `benchmark/measure_perf.sh`.
- `workflows/ldp_containment/omp/*.c` are standalone libgomp measurements driven by their own sbatch files.
- `spikes/cvfem/docs/*.pdf` are built from tracked .tex beside them and are kept as reports.
- `spikes/cvfem/subpar/` is the intentional quarantine (gated by the `cvfem_subpar_compiles` ctest). Keep it.
- These exist only in the main checkout (`../sfem`, untracked) and are absent from this worktree: `algebra/accelerate/`, `frontend/contact/*.bak_20260831_200801` (3 files), the root files `-h` and `arning:p`, `python/sfem/CMakeCache.txt`, `data/vtk/sfem.trace.csv`. Delete them there by hand if wanted.
