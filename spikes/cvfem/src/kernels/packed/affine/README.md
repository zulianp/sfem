# `packed/affine/` — the packed and store layouts' affine sweeps

Every sweep here reads **one adjugate and determinant per element** from a table built once at
setup. That is what makes the variant affine: all of its geometry comes from that constant
Jacobian. True per-element geometry belongs to `../isoparametric/`.

| sweep | operation |
|---|---|
| `apply_residual_packed_affine_range` | the first-order residual, 16-wide over elements |
| `apply_jacobian_action_packed_affine_range` | the matrix-free Jacobian action |
| `assemble_jacobian_packed_affine_range` | the assembled Jacobian, pack-local then drained |
| `assemble_jacobian_store_affine_range` | the same assembly with the store layout's drain |
| `apply_residual_packed_defcor_range` | the deferred-correction higher-order residual |
| `apply_jacobian_action_packed_pa_range` | the partially assembled action (quarantined) |

The last two were never templated on the geometry — both read the adjugate table, so neither
has an isoparametric form. They are here because this is where the affine kernels live, not
because they were split from a twin.

## What this folder used to say, and why it was wrong

It said: *"empty, because the separation is the template parameter"*. The four sweeps were each
one function templated on `bool ISO`, and the argument was that DESIGN.md's clause — "`affine`,
`iosparametric` and `axis_aligned` kernels logically separated (now they are mixed in with enum
and booleans)" — was satisfied once the enum and the booleans were gone, because the caller then
picked the instantiation and the geometry was fixed at the lane loop.

The correction: **"I indicated separate folders for geometry affine vs isoparametric. This
implies that the kernels should be separated. Quite obvious isn't it? GeomKind is used at the
front end level to dispatch based on the type of elements in the block (now smesh also provides
such enums) and it can be overriden at runtime."**

So the structure was the request, and a compile-time parameter making the same distinction was
an argument about intent rather than the thing asked for.

## What the split bought, beyond the structure

It is not only a move. A sweep templated on the geometry has to take the **union** of both
geometries' inputs, so DESIGN.md's other clause — "the signatures of the functions are
lean-and-mean only arguments that are acually used are passed" — could not hold for either half:

* the isoparametric residual sheds six parameters,
* the isoparametric Jacobian action sheds twelve,
* the isoparametric assemblies shed the adjugate table.

All of that was dead in the isoparametric branch. The isoparametric SIMD kernels carry no
Rhie-Chow term — it was never put into them, which is why the driver refuses `--rhie-chow` on
this geometry for any pack-based layout — so `with_rc`, the three pressure-gradient arrays, the
three direction-gradient arrays, the coefficient table, the scale and the config were all
parameters that half could not use and had to accept.

## One thing that is deliberately NOT shared, and the measurement that decided it

The residual's lane loop is written out in all four sweeps -- the contiguous packed pair here and
in `../isoparametric/`, and the pack-coloured pair beside them -- although the four differ only
in their drain. That is the one-path rule being overruled by a measurement, and the numbers are
in the code beside each copy so it is not re-shared by someone applying the rule without them.

Two A/B runs against the same reference, each reproduced within its own allocation
(`jobs/ab_refactor.sbatch` 4983280 and 4983377):

| row | first run | second |
|---|---|---|
| `residual_packed_sumfact` | −9.4% | −9.7% |
| `residual_packed_sumfact_big` | −8.2% | −8.4% |
| `residual_colored_sumfact` | −17.3% | −20.0% |

Everything else was inside its band. Two facts narrow the cause: every row carrying Rhie-Chow or
the higher-order correction was clean, so the cost is fixed per pack and only the cheapest lane
loop notices it; and the **Jacobian action's** lane loop, which *is* still shared, measured +0.4%
and +0.2% — its packs are larger and its arithmetic per pack far greater. Hoisting the lane
scratch into one object per thread, so the packs were not re-materialised per pack, did not
recover it either; that is what the second run measured.

Worth keeping from this: the numerical gates cannot see it. All 66 flat fingerprints and the
semi-structured one were unchanged across the regression, because the arithmetic was identical.
A kernel restructuring needs `scripts/perf_regression.sh --against` even when every oracle is
silent.

## Where the shared machinery is, and why it had to move first

Nothing is duplicated between the two folders. Splitting would have copied whatever the two
geometries shared, so that went into the layout's own files first:

* `../cvfem_pack_scratch.hpp` — `Hex8PackCoords` and `cvfem_hex8_pack_coords` (slot 3),
  `Hex8PackQGrad` (slot 4), `Hex8PackExtent`, and the two pack drains over one ghost stager.
* `../cvfem_hex8_best_packed.hpp` — `Hex8PackElement` and `cvfem_hex8_stage_pack_element`, the
  scalar assemblies' shared element preamble.

Each had been copied into four or five sweeps; the split would have taken that to eight. It is
one definition now, and the two layout headers are 84 and 23 lines of it.

## Where the geometry is chosen

In `frontend/staging/cvfem_hex8_packed_launch.hpp` and `cvfem_hex8_store_launch.hpp`, from a
`GeomKind` argument, which `--geom` sets at run time. Nothing below the launcher tests it.
