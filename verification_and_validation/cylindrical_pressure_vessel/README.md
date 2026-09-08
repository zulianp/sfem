# Hyperelastic Cylindrical Pressure Vessel

This case reproduces the solids4foam thick cylindrical pressure-vessel benchmark with a self-generated 20 x 20 `QUAD4` quarter-annulus mesh.

- Undeformed radii: `Ri = 7 m`, `Ro = 18.625 m`
- Plane-strain modified Mooney-Rivlin material: `c10 = 80 MPa`, `c01 = 20 MPa`
- Near-incompressible bulk modulus: `K = 9933.333333333334 MPa` (`nu = 0.49`)
- Internal pressure: `100 MPa`, applied in 20 equal increments
- Symmetry constraints on the two radial boundaries

The pass/fail oracles are the published analytical radial and hoop Cauchy-stress profiles. The verifier reconstructs the deformation gradient from SFEM's final displacement, computes the Cauchy stress from the same strain-energy definition as `GeneratedModifiedMooneyRivlin`, samples both sides of the undeformed 45-degree radius, and compares profile errors against the tolerances in `case.yaml`.

The report also records the minimum and maximum deformation Jacobian at element Gauss points as diagnostics. These values are not compared with the tutorial's cell-centered finite-volume Jacobian interval because the sampling locations and discretizations differ.

Run only this case from the repository root:

```bash
verification_and_validation/run_all.py --case cylindrical_pressure_vessel
```

References:

- https://www.solids4foam.com/tutorials/more-tutorials/solid-mechanics/hyperelasticity/cylindricalPressureVessel.html
- https://github.com/solids4foam/solids4foam-tutorials-benchmark-data/tree/6917f439243d7d782d42aacf1227b12b2300d5b2/tutorials/solids/hyperelasticity/cylindricalPressureVessel/fe41
