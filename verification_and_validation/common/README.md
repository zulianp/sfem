# Shared Case Infrastructure

The modules in this directory support self-contained verification and
validation cases. They depend only on NumPy and PyYAML from the repository
virtual environment. They do not call SFEM operators, so constitutive oracle
equations remain independent and visible inside each case.

## Modules

| Module | Purpose |
| --- | --- |
| `raw.py` | Typed raw-array filenames, checked writes, and finite-value reads. |
| `mesh.py` | Validated `Mesh` arrays and SFEM `meta.yaml` read/write. |
| `geometry.py` | Deterministic rectangle, box, annulus, annular-sector, cylindrical-sector, and spherical-shell meshes. |
| `sets.py` | Boundary extraction, SFEM local-side numbering, nodesets, sidesets, normals, measures, and orientation checks. |
| `fields.py` | Scalar/vector nodal files for generated boundary values and initial fields. |
| `mechanics.py` | First-order element quadrature, kinematics, stress transformations, energy integration, and boundary resultants. |
| `metrics.py` | Absolute, relative, weighted, maximum, and interpolated curve errors with explicit norm floors. |
| `convergence.py` | Spatial and temporal log-log convergence fits. |
| `reporting.py` | Strict construction, serialization, and validation of per-variant `verification.json`. |

## Conventions

- `nx`, `ny`, `nz`, `radial_cells`, `angular_cells`, and `axial_cells`
  denote element counts, not point counts.
- Mesh points have shape `(n_points, dimension)` and connectivity has shape
  `(n_elements, nodes_per_element)`.
- Internal geometry uses `float64` and connectivity uses `int64`. Mesh output
  defaults to checked `float32` geometry and `int32` connectivity streams.
- Local side numbers and node order match `smesh::LocalSideTable` for `TRI3`,
  `QUAD4`, `TET4`, and `HEX8`.
- Surface area vectors and normals are outward only when parent elements have
  positive orientation. Call `validate_sideset_orientation` before writing a
  generated sideset.
- Relative norms always take an explicit positive absolute floor. This avoids
  undefined normalization when the analytical field is zero or nearly zero.
- Kinematic arrays are indexed `(element, quadrature, component, derivative)`.
  TRI3/TET4 use one exact affine point; QUAD4/HEX8 use tensor-product 2-point
  Gauss quadrature.
- The spherical-shell generator creates a conforming TET4 mesh by radially
  extruding a subdivided octahedral surface. Increasing `surface_frequency`
  refines the spherical approximation.

## Minimal Use

```python
from common.geometry import box_mesh
from common.mesh import write_mesh
from common.sets import boundary_sides, validate_sideset_orientation, write_sideset

mesh = box_mesh(1.0, 1.0, 1.0, nx=4, ny=4, nz=4, element_type="HEX8")
sides = boundary_sides(mesh)
validate_sideset_orientation(mesh, sides)
write_mesh(output / "mesh", mesh)
write_sideset(output / "mesh" / "surface" / "sidesets" / "boundary", mesh, sides)
```

Run the shared infrastructure tests with:

```bash
PYTHONPATH=python venv/bin/python -m unittest discover -s verification_and_validation/tests -v
```
