# Affine manufactured-solution refinement

The paper retains twenty FP64 solves across n=16,24,32,40 and five convection
schemes: first-order upwind, unlimited reconstruction, clip, Venkatakrishnan
with K=0 and Darwish. These are affine, steady manufactured-solution checks,
separate from the irregular-mesh operator timings.

The domain is [0,2]^3, density one and viscosity 0.01. Manufactured velocity
is imposed on all faces and pressure is compared after removing its mean.
FGMRES uses node-block Jacobi, restart 480, final nonlinear relative tolerance
1e-10 and linear relative tolerance 1e-6. Convection is unfrozen and the
Rhie–Chow derivative is consistent. Eight OpenMP threads are configured.

Jobs 5077627 (n=16,24,32) and 5077906 (n=40) use an identical solver binary.
All final nonlinear targets pass. Only the four reconstructed n=40 cases pass
the fixed 0.01 maximum velocity-error gate; all sixteen failed accuracy gates
and original exit codes remain in the report. The clip n=16 case accepts an
intermediate continuation residual floor before meeting the final target.
Finest-interval velocity-L2 rates are approximately 2.04, 2.07, 1.49 and 1.36
for the reconstructed schemes, and 0.73 for first-order upwind. They are
observations, not a common asymptotic-order guarantee.

The [paper README](../../../wip/paper/README.md#irregular-meshes-and-affine-refinement)
identifies the external archive and complete replay. The archive preserves
the actual wrapper, settings, source hashes and binary identity. Use those
recorded inputs to reproduce the historical study; applying the wrapper to a
different source revision is a new study and must be labelled accordingly.

For fresh logs with the same output format:

```sh
python3 spikes/cvfem/scripts/mms_refinement_summary.py /path/to/study --grids 16 24 32
python3 spikes/cvfem/scripts/mms_refinement_summary.py /path/to/fine-study --grids 40
python3 spikes/cvfem/scripts/test_mms_refinement_summary.py
```

The summary checks final nonlinear and spatial-accuracy targets separately.
It does not turn a failed accuracy gate into a successful solve exit.
