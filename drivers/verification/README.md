# Generated Stokes Verification

This folder contains reusable scripts for manufactured-solution convergence
studies with the generated Stokes operator.

The M8 reference is HAL `cea-02434556`:

P.-E. Angeli, M.-A. Puscas, G. Fauchet, A. Cartalade, "FVCA8 benchmark
for the Stokes and Navier-Stokes equations with the TrioCFD code - benchmark
session", FVCA8, 2017.

Only the paper's steady Stokes studies are encoded here, as requested:

- `bercovier_engelman_2d`: paper section 3.1, domain `[0,1]^2`, viscosity
  `nu = 1`, nonhomogeneous Dirichlet boundary values from the exact velocity.
  Expected convergence is second order for velocity on triangular/rectangular
  meshes; pressure is first order on triangular meshes and second order on
  rectangular meshes.
- `taylor_green_3d`: paper section 3.2, domain `[0,1]^3`, viscosity `nu = 1`,
  nonhomogeneous Dirichlet boundary values from the exact velocity. Expected
  convergence is about second order for velocity on hexahedral meshes and
  above 1.7 on the refined tetrahedral meshes; pressure is second order on
  hexahedral meshes and first order on tetrahedral meshes.

Legacy local MMS cases remain available for comparison. Run
`PYTHONPATH=python venv/bin/python drivers/verification/stokes_mms.py` or pass
`--list-cases` to the collector/extractor to see all available cases.

## Files

- `generated_stokes_fvca8.cpp`: serial generated-`GeneratedStokes` driver for
  the paper's steady FVCA8 manufactured Stokes cases. It builds the
  Taylor-Hood generated operator, projects the paper forcing, enforces exact
  Dirichlet velocity values, constrains promoted pressure-only DOFs, and writes
  typed SFEM `Output` fields plus `summary.csv`.
- `run_generated_stokes_fvca8.py`: convenience runner for multiple generated
  driver levels plus convergence CSV/plot collection.
- `stokes_mms.py`: analytical velocity, pressure, and forcing cases.
- `run_stokes_convergence.py`: creates a convergence table from raw solution
  fields and mesh coordinates.
- `plot_convergence.py`: plots error versus mesh spacing from the generated CSV.
- `extract_stokes_fields.py`: exports numerical, exact, and error fields at
  mesh nodes for one run.
- `plot_stokes_fields.py`: plots extracted fields or errors against the
  analytical solution.

## Generated Driver

Build the driver:

```bash
cmake -S . -B build64
cmake --build build64 --target generated_stokes_fvca8 -j 4
```

Run the 2D Bercovier-Engelman steady Stokes study with the generated operator:

```bash
PYTHONPATH=python venv/bin/python drivers/verification/run_generated_stokes_fvca8.py \
  --exe build64/generated_stokes_fvca8 \
  --case bercovier_engelman_2d \
  --resolution 2 \
  --resolution 4 \
  --resolution 8 \
  --out-root runs/fvca8_bercovier \
  --plot runs/fvca8_bercovier/convergence.png
```

The compiled driver uses a dense matrix-free assembly fallback for small
systems (`SFEM_DENSE_SOLVE=1`, default up to 2048 DOFs) so convergence evidence
does not depend on an unpreconditioned saddle-point Krylov solve. Set
`SFEM_DENSE_SOLVE=0` or pass `--krylov` to the Python runner to use BiCGStab.

Run the 3D Taylor-Green steady Stokes study with the generated operator:

```bash
PYTHONPATH=python venv/bin/python drivers/verification/run_generated_stokes_fvca8.py \
  --exe build64/generated_stokes_fvca8 \
  --case taylor_green_3d \
  --resolution 2 \
  --resolution 3 \
  --resolution 4 \
  --out-root runs/fvca8_taylor_green \
  --plot runs/fvca8_taylor_green/convergence.png
```

The 3D driver defaults to `HEX27/HEX8`. Small levels use the dense fallback;
larger levels use BiCGStab automatically unless `SFEM_DENSE_SOLVE=1` is set.
Set `SFEM_MAX_IT` and `SFEM_ATOL` when collecting refined data, for example
`SFEM_MAX_IT=50000 SFEM_ATOL=1e-11`. Set `SFEM_FVCA8_USE_TETS=1` for the
tetrahedral generated path.

## Expected Data Layout

For each refinement level, provide one directory containing mesh coordinates
and one directory containing numerical solution fields. The scripts prefer
typed SFEM `Output` files (`x.float32`, `u0.float64`, `p.float64`) and retain
fallback support for legacy `.raw` files. A 2D run uses:

```text
case_root/
  n16/
    mesh/x.float32
    mesh/y.float32
    solution/u0.float64
    solution/u1.float64
    solution/p.float64
  n32/
    mesh/x.float32
    mesh/y.float32
    solution/u0.float64
    solution/u1.float64
    solution/p.float64
```

Use `--level <name>:<h>:<mesh-dir>:<solution-dir>` to describe each level.
The scripts intentionally keep mesh generation and linear solve orchestration
outside the error collector so they can be used with generated `sfem::Op`
drivers, external solvers, or archived runs.

`generated_stokes_fvca8` writes the coordinate and solution files into the same
run directory, so that directory can be passed as both `<mesh-dir>` and
`<solution-dir>`.

A 3D run additionally uses `mesh/z.float32` and `solution/u2.float64`.

## Example

```bash
PYTHONPATH=python venv/bin/python drivers/verification/run_stokes_convergence.py \
  --case bercovier_engelman_2d \
  --level n16:0.0625:runs/n16/mesh:runs/n16/solution \
  --level n32:0.03125:runs/n32/mesh:runs/n32/solution \
  --out runs/stokes_convergence.csv

PYTHONPATH=python venv/bin/python drivers/verification/plot_convergence.py \
  runs/stokes_convergence.csv \
  --out runs/stokes_convergence.png

PYTHONPATH=python venv/bin/python drivers/verification/extract_stokes_fields.py \
  --case bercovier_engelman_2d \
  --mesh-dir runs/n32/mesh \
  --solution-dir runs/n32/solution \
  --out-csv runs/n32/stokes_fields.csv

PYTHONPATH=python venv/bin/python drivers/verification/plot_stokes_fields.py \
  runs/n32/stokes_fields.csv \
  --field velocity_error \
  --out runs/n32/velocity_error.png
```
