# `packed/isoparametric/` — the packed and store layouts' isoparametric sweeps

Every sweep here derives the Jacobian **per sub-control surface from the element's node
coordinates**, which the pack stages for it. None of them takes an adjugate table.

| sweep | operation |
|---|---|
| `apply_residual_packed_isoparam_range` | the first-order residual, 16-wide over elements |
| `apply_jacobian_action_packed_isoparam_range` | the matrix-free Jacobian action |
| `assemble_jacobian_packed_isoparam_range` | the assembled Jacobian, pack-local then drained |
| `assemble_jacobian_store_isoparam_range` | the same assembly with the store layout's drain |

There is no higher-order or partially assembled sweep here: both read the adjugate table and
exist only in `../affine/`.

## What these take that the templated sweeps could not

Six parameters fewer on the residual and twelve on the Jacobian action, because a sweep
templated on `bool ISO` has to accept the union of both geometries' inputs. The isoparametric
SIMD kernels carry no Rhie-Chow term — it was never put into them, which is why the driver
refuses `--rhie-chow` on this geometry for any pack-based layout — so `with_rc`, the pressure
and direction gradients, the coefficient table, the scale and the Rhie-Chow config were all
dead in this half and had to be in the signature regardless.

The assemblies are the exception on the term: they are scalar per element and the
isoparametric assembly kernel does carry Rhie-Chow, so those two keep it.

See `../affine/README.md` for the correction that produced this split, for what this folder
used to claim instead, and for where the shared pack machinery lives.
