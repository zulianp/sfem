# Verification and Validation

This directory contains self-contained numerical cases that exercise SFEM drivers against independent oracles. A case passes only when every declared comparison reproduces an analytical solution, manufactured solution, or measurement within its stated tolerance. Solver completion by itself is never a pass condition.

Run the complete suite from the repository root:

```bash
verification_and_validation/run_all.py
```

Useful options:

```bash
verification_and_validation/run_all.py --list
verification_and_validation/run_all.py --case cylindrical_pressure_vessel
verification_and_validation/run_all.py --family hyperelastic --dimension 2
verification_and_validation/run_all.py --tier fast --operator GeneratedLinearElasticity
verification_and_validation/run_all.py --variant cylindrical_pressure_vessel/default
verification_and_validation/run_all.py --build-dir build64 --output-dir /tmp/sfem-vv
verification_and_validation/run_all.py --verbose
```

The runner discovers one `case.yaml` in each immediate child directory and
validates all manifests before executing any case. New manifests use the
[version 2 schema](SCHEMA.md), which declares one or more operator, element,
and resolution variants. Version 1 remains supported for the existing pressure
vessel.

Run the runner tests with:

```bash
PYTHONPATH=python venv/bin/python -m unittest discover -s verification_and_validation/tests -v
```

Reusable mesh, field, mechanics, norm, convergence, and oracle-report helpers
are documented in [common/README.md](common/README.md).

Each case declares:

- a command that generates its mesh from source;
- checked-in YAML input templates for the driver;
- the SFEM driver and environment;
- a postprocessor that writes the common `verification.json` schema;
- oracle provenance and numerical tolerances.

The suite writes one isolated output directory and log per variant plus
`report.yaml` at the output root. Results use `PASS`, `FAIL`, `ERROR`, and
`SKIP`; skipped variants require a reason and do not count as covered. The
runner returns a non-zero status for failures or errors.

## Driver inputs

The static linear-elasticity driver accepts both its original command and the
extended YAML form:

```text
linear_elasticity <mesh> <dirichlet.yaml> <output>
linear_elasticity <mesh> <dirichlet.yaml|NONE> <neumann.yaml|NONE> <operator.yaml|NONE> <output>
```

The extended form preserves generated mesh numbering so file-backed node and
side sets remain valid. Set `SFEM_REORDER=1` only when the input sets use the
driver's reordered numbering. Operator YAML accepts top-level parameters and
per-block overrides:

```yaml
operator:
  type: GeneratedLinearElasticity
  mu: 2.0
  lmbda: 3.0
  blocks:
    - name: block_name
      mu: 4.0
      lmbda: 6.0
```

A Dirichlet scalar can be replaced by one `float64` value per node in its
node set. Paths rendered by the runner are absolute:

```yaml
value:
  path: /absolute/path/boundary_values.float64.raw
```

Dirichlet and Neumann entries accept the same multiplicative `profile`. The
base scalar or file field is multiplied by the profile value at each time:

```text
profile: {type: constant, value: 1}
profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}
profile: {type: hold, start_time: 1, before_value: 0, value: 1}
profile: {type: pulse, start_time: 1, end_time: 2, before_value: 0, value: 1, after_value: 0}
profile: {type: tabulated, path: /absolute/path/time_value.csv}
```

Tabulated files contain strictly increasing `time,value` rows and are linearly
interpolated, with endpoint values held outside the table range. The transient
drivers accept either one interleaved field or one file per component through
`SFEM_INITIAL_DISPLACEMENT`, `SFEM_INITIAL_DISPLACEMENT_COMPONENTS`,
`SFEM_INITIAL_VELOCITY`, and `SFEM_INITIAL_VELOCITY_COMPONENTS`. Component
paths are comma-separated; all state files use `float64`.

Static drivers write unconstrained `material_reaction` fields and
`quantities.yaml`. The history-aware hyperelastic driver records time,
material objective when available, total constrained reaction, and the
reaction resultant for each YAML Dirichlet condition. Three-dimensional
follower pressure requires outward-oriented `TRI3` or `QUAD4` surfaces and is
accepted only by nonlinear drivers.

## Adding a case

Create `<case_id>/case.yaml`, a deterministic mesh generator, the required YAML
templates, and a verifier. Follow [SCHEMA.md](SCHEMA.md). The verifier report
must contain a non-empty `checks` array. Every check must identify its oracle
and include `observed`, `expected`, `error`, `tolerance`, `units`, and `passed`
fields.
The top-level runner deliberately rejects empty reports so smoke tests cannot
be mistaken for validation.
