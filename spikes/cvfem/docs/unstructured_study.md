# Public irregular-HEX8 operator comparison

This experiment compares atomic and packed first-order and Darwish residuals
and their consistent Jacobian actions on a public, procedurally generated irregular mesh. It extends
the affine-cube experiments to non-affine geometry and irregular connectivity.
It does not measure a nonlinear solve or establish spatial accuracy.

## Mesh and reproduction

`scripts/unstructured_hex_mesh.py` uses Gmsh 4.15.0 to subtract a cylindrical
through-hole of radius 0.18 from the unit box. Delaunay tetrahedra are subdivided
into HEX8 cells with `Mesh.SubdivisionAlgorithm=2`. The target size varies as
`h*(0.55+0.45*x)`. Meshing is single-threaded with seed 314159.

The small numerical fixture uses `h=0.3` (2,864 cells); the two timing fixtures
use `h=0.085` (74,664 cells) and `h=0.055` (265,636 cells). Their interior node
incidence varies, so these are not Cartesian meshes with perturbed coordinates.
The geometry has no private input. Gmsh's algorithm is described in its
[official manual](https://gmsh.info/doc/texinfo/gmsh.html).

Generate each directory with Python, NumPy and `gmsh==4.15.0`, for example:

```sh
python scripts/unstructured_hex_mesh.py mesh-small --h 0.3
python scripts/unstructured_hex_mesh.py mesh-medium --h 0.085
python scripts/unstructured_hex_mesh.py mesh-large --h 0.055
```

The mesh is rounded to the production float32 coordinates and int32 connectivity
before validation. The generator checks node uniqueness within each cell,
manifold face incidence and irregular interior connectivity. It evaluates the
triquadratic Jacobian determinant's tensor-product Bernstein coefficients and
requires a positive lower bound with a relative margin of `1e-10`. This is a
floating-point geometry check, not an interval-arithmetic proof. The complete
arrays, settings, counts and SHA256 hashes are saved. Transfer those arrays for
measurement; a different Gmsh build is not assumed to reproduce identical bytes.

## Operators and numerical checks

The unknowns are three velocity components and physical pressure. Density is
1.3, dynamic viscosity 0.027, and the Rhie–Chow scale is one with the steady
combined time scale. Both first-order upwinding and Darwish reconstruction are tested. Only interior sub-control
surface fluxes are applied; boundary closure, boundary elimination and transient
terms are excluded. There is no Reynolds-number or physical flow claim.

Every face uses its trilinear geometric mapping and the physical edge
`d = x_j - x_i`. The pressure correction is
`C = p_j - p_i - 0.5*(grad(p)_i + grad(p)_j) dot d`, with mass-flux correction
`-c*C`. Its directional derivative includes both the reconstructed direction
gradient and the velocity derivative of `c`. The packed SIMD kernel uses the
existing coefficient and derivative helpers. The scalar isoparametric gather
now uses physical edges instead of the affine-column shortcut. A regression
checks zero correction for physical linear pressure on a distorted cell and
requires the old shortcut to fail that same check.

Both layouts start from the same SFC ordering. Packing then permutes node
storage. A checked bijection and exact coordinate match transfer the nonlinear
state and three directions without recomputing their floating-point values.
The state pressure is `0.3*sin(2*x+3*y+5*z)+0.17*x*y`; directions are smooth,
nonzero in all four fields and differ from one another. Before timing:

- every cell has positive determinant at all 27 reference samples;
- the reconstructed gradient of `x+2*y+3*z` matches `(1,2,3)` within `1e-11`;
- all four equations pass finite-output and interior-flux cancellation checks;
- packed and atomic outputs agree within `1e-11`, using each equation's own
  infinity-norm scale;
- for first order, both layouts' full Jacobian actions match centered differences
  at `1e-6` and `1e-7`, with relative error below `1e-6`, for all three directions;
- omitting the reconstructed pressure-direction gradient must produce relative
  error above `1e-5` in every difference check.

For Darwish, each velocity component is reconstructed from its physical nodal
gradient to the mapped SCS point. The Jacobian differentiates the limited
increment, including the direction gradient, and its mass-flux weight, including
both Rhie--Chow sensitivities. Fixed-state velocity and pressure gradients are
reused; direction gradients remain live. The affine coefficient cache is not
used on these non-affine cells.

The Darwish derivative test retains a coarse `1e-6` diagnostic, then requires
whole-vector relative error below `1e-6` at **both** `1e-7` and `1e-8`. It also
requires no sign crossing of either Darwish limiter argument at those two steps.
The initial large-mesh local check crossed 6--7 limiter switches per direction at
`1e-6`; that failed check is retained separately. No components or timing samples
are removed. Omitting either reconstructed pressure-direction gradients or
velocity-direction gradients must produce relative error above `1e-5`.
The element test independently checks the physical reconstruction point, its
affine limit, scalar/SIMD agreement and directional finite differences.

These tests cover steady interior configurations. They do
not validate arbitrary limiter switches, curved high-order elements, assembled
Jacobians, general boundary conditions, time integration or solver convergence.

The complete Grace campaigns for both schemes have been replayed and reviewed.
The paper now reports both schemes; the affine MMS already includes Darwish
and remains a separate accuracy study.

## Timing and storage

`cvfem_hex8_unstructured_bench` retains both layouts in one process. A residual
includes state-gradient reconstruction (pressure alone for first order, pressure and velocity for Darwish); a Jacobian action reuses the fixed
state gradient and reconstructs the current direction gradient. Neither arm
caches direction gradients. Geometry is evaluated in the operator, and neither
arm builds an unused affine Rhie–Chow coefficient table.

The Grace recipe measures both timing meshes at 1 and 72 threads with pack sizes
256 and 1024. Each operator has ten initial warmups, twelve paired blocks, ten
warmups before each arm and twelve timed applications per arm. Atomic/packed
order alternates by block. Actions cycle through all three directions. The
steady clock measures only application calls; post-capture output checks are
outside the interval. Every sample is retained. Speedup is the ratio of mean
normal times, with individual ranges and paired-ratio ranges also reported.
There is no speedup threshold for acceptance and no PMU instrumentation.

Both mesh objects and all validation allocations coexist; this is a warm
repeated-application experiment, not a cold-memory or isolated-process result.
On Grace, the existing process-local wrapper disables transparent huge pages
for each benchmark worker and verifies the setting before exec. System settings
are unchanged. The CPU allocation, compiler, binary hashes and launch commands
are retained. The larger pack on the smaller grid leaves only 73 packs for 72
threads; the smaller pack is included to expose that parallelism tradeoff.

The logs retain separate SFC-ordering and packing times. Storage is an explicit
array ledger: original coordinate/connectivity arrays, additional packed
connectivity/ownership/ghost/reduction/map arrays, ghost output buffer, resident
field/vector capacities, and the packed thread arena. The arena uses 25 scalars
per maximum pack node per thread for first order and 55 for Darwish after
residual/action validation. State backups and test directions are listed separately as harness allocations. Shared result
references, temporary validation vectors, allocator metadata, stacks and other
library/runtime storage are excluded. Thus this ledger is not peak RSS or a
complete solver footprint, and retaining the original connectivity must not be
presented as a total-memory saving.

## Running and inspecting

Build the driver, `cvfem_isoparam_rc_test` and `cvfem_isoparam_darwish_test`
through the standalone CVFEM CMake project.
`scripts/unstructured_study.py --local-check` runs four small
numerical fixtures at 1/2 threads and a short paired timing smoke test. Production
collection requires the 72-core Slurm allocation and the hashed mesh directories.
Inside a 72-core allocation, invoke the controller from the repository root:

```sh
python3 spikes/cvfem/scripts/unstructured_study.py \
  --binary /absolute/build/cvfem_hex8_unstructured_bench \
  --meshes /absolute/meshes --out /absolute/new-results
```

The mesh parent contains `mesh-small`, `mesh-medium` and `mesh-large`.
Use `--local-check` with the small mesh for one-/two-thread numerical checks
and a short smoke test; those timings are not paper performance evidence.

The controller refuses missing cases, changed mesh bytes, wrong scopes or teams,
nonfinite values, failed differences/negative controls and changed timed
outputs. A failed job's logs must be kept. `records.json` preserves identities,
commands, setup, storage and every accepted capture; `summary.md` reports the
paired times. No manuscript result is updated until the complete Grace run has
been reviewed.

## Historical first-order Grace run

Job 5052301 on nid006545 passed the complete campaign on 2026-10-14. The source
and mesh hashes, all twelve case logs and the generated timing table were
rechecked locally. With pack256 at 72 threads, packed residual speedups are
1.537x and 1.715x on the medium and large meshes; Jacobian-action speedups are
1.392x and 1.510x. Pack1024 on the medium mesh is also retained: the Jacobian is
about 8.3% slower than atomic. This is evidence that the pack choice matters.

The [saved result and replay instructions](../../../wip/paper/README.md)
include the complete one-/72-thread table, sample ranges, setup measurements,
explicit storage ledger and limits. The current manuscript replaces this
historical table with the two-scheme comparison below; the historical study remains in the original external archive.


## First-order and Darwish extension

The updated source was tested in two separate Grace allocations on 9 November
2026: first-order job 5306375 on nid005717 and Darwish job 5305511 on nid005379.
Both passed ten CTests, baseline comparisons at one and 72 threads, four small
numerical cases and eight paired timing cases. Each scheme retains 384 timing
samples, with 12 pairs per operator and configuration. The earlier Darwish
job 5305251 exceeded its time limit; none of its samples is pooled into these runs.

On 72 cores and the large mesh, packed residual speedups are 1.60–1.69x for
first order and 1.66–1.69x for Darwish; consistent Jacobian speedups are
1.38–1.47x and 1.33–1.36x, respectively. These are within-run layout comparisons,
not a cross-allocation measurement of the cost of changing schemes.
On the medium mesh, pack1024 makes the Jacobian 12.6% slower for first order
and 17.9% slower for Darwish. Both slow cases persist in all 12 pairs.
Pack256 benefits both operators for both schemes. All samples remain in the report.

The numerical kernels match the frozen source snapshot used in these runs.
Table regeneration checks the external archive, source and mesh hashes,
commands, binary identities, full numerical/timing coverage and saved summaries.
The archive and regeneration instructions are in the paper README linked above.
No nonlinear solve, irregular-mesh PDE convergence or boundary/transient
validation is claimed by these operator studies.
