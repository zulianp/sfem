# Pressure-Vessel Oracle Provenance

The radial and hoop tables are the `radial_stress.dat` and `hoop_stress.dat`
reference values from
[`solids4foam-tutorials-benchmark-data` at revision `6917f439243d7d782d42aacf1227b12b2300d5b2`](https://github.com/solids4foam/solids4foam-tutorials-benchmark-data/tree/6917f439243d7d782d42aacf1227b12b2300d5b2/tutorials/solids/hyperelasticity/cylindricalPressureVessel/fe41).
The upstream repository attributes these data to Bijelonja, Demirdzic, and
Muzaferija (2005). The checked-in CSV files change only the whitespace field
separator to a comma and add descriptive headers; their numerical values are
unchanged.

The first column is **undeformed radius in metres**. The second column is
**radial or hoop Cauchy stress in MPa** at the final `100 MPa` inner pressure.
The [tutorial](https://www.solids4foam.com/tutorials/more-tutorials/solid-mechanics/hyperelasticity/cylindricalPressureVessel.html)
plots stress along the undeformed radius and describes its plane-strain,
quarter-cylinder `20 x 20` setup with `Ri = 7 m`, `Ro = 18.625 m`,
`c10 = 80 MPa`, and `c01 = 20 MPa`. Its displayed curves use `nu = 0.49`.

For the canonical SFEM mesh, the verifier samples the two element edges
adjacent to the **undeformed 45-degree ray**, at the midpoint of each radial
cell, and averages the two Cauchy stress tensors after radial/hoop projection.
The published tables are linearly interpolated at those undeformed radii.
Only sample radii inside each table's support enter the canonical relative-L2
and maximum-absolute-error checks; there is no extrapolation. The independent
refinement diagnostic uses a common fixed radius grid and linear interpolation
of each SFEM profile and the published tables on their mutual support.
