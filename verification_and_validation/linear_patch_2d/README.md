# Two-Dimensional Linear Patch

This exact verification case uses a unit square with all boundary displacement
components prescribed from three affine fields. The legacy `TRI3` operator and
generated `TRI3` and `QUAD4` operators are covered; the legacy linear operator
does not provide a `QUAD4` kernel. `TRI3` variants use the fixed skew transform
recorded in generated `generation.yaml`, while `QUAD4` remains axis aligned. All
meshes and boundary values are generated at run time.

The independent oracle evaluates the plane-strain small-strain law

`epsilon = sym(F - I)`,
`sigma = 2 mu epsilon + lambda tr(epsilon) I`, and
`W = 0.5 sigma:epsilon`.

For each mode, the verifier compares the nodal displacement, the normalized
interior material residual, the driver-exported material objective, and the
driver-exported resultant on the reference `x = 1` face. Parameters and all
reported quantities use one consistent nondimensional unit system.
