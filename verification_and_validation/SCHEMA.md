# Case Manifest Schema

Every immediate child of `verification_and_validation` that contains a
`case.yaml` is a discoverable case. The runner validates every discovered
manifest before it starts any mesh generator or driver.

## Versions

- **Version 1:** one implicit `default` variant. It remains supported for the
  existing cylindrical pressure-vessel case and keeps the historical output
  path `<output>/<case-id>`.
- **Version 2:** one or more explicit variants. Outputs use
  `<output>/<case-id>/<variant-id>` and each variant is reported separately.

New cases must use version 2.

## Version 2

```yaml
schema_version: 2
id: linear_patch_2d
name: Two-dimensional linear-elastic patch test
kind: verification
family: elastic
dimension: 2
tier: fast

source:
  description: Closed-form affine displacement, stress, energy, and reactions
  reference: oracle/README.md

oracle:
  type: analytical
  implementation: oracle.py

material:
  mu: 2.0
  lambda: 3.0

mesh:
  command:
    - "{python}"
    - "{case_dir}/generate_mesh.py"
    - "{mesh}"
    - --element
    - "{element}"
    - --nx
    - "{resolution_nx}"
    - --ny
    - "{resolution_ny}"

inputs:
  - template: dirichlet.yaml
    output: "{output}/dirichlet.yaml"

verification:
  command:
    - "{python}"
    - "{case_dir}/verify.py"
    - --case
    - "{resolved_case}"
    - --output
    - "{output}"
    - --report
    - "{output}/verification.json"
  report: "{output}/verification.json"
  tolerances:
    displacement_relative_l2: 1.0e-10
    reaction_relative: 1.0e-9

variants:
  - id: generated_quad4
    operator: GeneratedLinearElasticity
    element: QUAD4
    resolution:
      nx: 8
      ny: 8
    material_parameter_map:
      mu: mu
      lambda: lmbda
    driver:
      executable: "{build_dir}/linear_elasticity"
      arguments:
        - "{mesh}"
        - "{output}/dirichlet.yaml"
        - "{output}/solution"
      environment:
        SFEM_OPERATOR: "{operator}"
        OMP_NUM_THREADS: 1
    expected_output:
      required: []
      forbidden:
        - No progress made
        - "(max iterations)"
    tolerances:
      displacement_relative_l2: 1.0e-11
```

### Case Fields

| Field | Requirement |
| --- | --- |
| `schema_version` | Integer `2`. |
| `id` | Unique path-safe case identifier. |
| `name` | Human-readable case name. |
| `kind` | `verification` or `validation`. |
| `family` | Material family used by `--family`, such as `elastic`, `hyperelastic`, or `viscoelastic`. |
| `dimension` | Integer `2` or `3`. |
| `tier` | `fast`, `medium`, or `extended`. |
| `source` | Oracle provenance with a `description` and at least one additional reference field. |
| `oracle` | Oracle metadata with a nonempty `type`. Paths must be local to the case. |
| `mesh.command` | Nonempty command array used to generate the mesh. |
| `inputs` | Optional list of input templates and rendered output paths. |
| `verification.command` | Command that creates the oracle report. |
| `verification.report` | Path to that report. |
| `verification.tolerances` | Nonempty case-level tolerance mapping. |
| `variants` | Nonempty list of explicit variants with unique IDs. |
| `material` | Optional physical material parameters shared by variants. |

### Variant Fields

Each variant must declare:

| Field | Requirement |
| --- | --- |
| `id` | Unique path-safe identifier within the case. |
| `operator` | SFEM operator name reported as the coverage target. |
| `element` | Element name reported as the coverage target. |
| `resolution` | A scalar or nonempty mapping. Mapping entries become template variables. |
| `driver.executable` | Driver path. |
| `driver.arguments` | Command arguments; may be empty. |
| `driver.environment` | Environment mapping; may be empty but must be present. |
| `expected_output.required` | List of exact substring/count requirements. |
| `expected_output.forbidden` | List of forbidden substrings. |
| `tolerances` | Nonempty overrides merged over case-level tolerances. |
| `material_parameter_map` | Optional mapping from physical names to operator YAML keys. Generated elasticity variants map `lambda` to `lmbda`; legacy variants map it to `lambda`. |

A variant may override `family`, `dimension`, `tier`, `mesh`, `inputs`, or the
verification command/report. Metadata overrides are intended for a shared
physical case that legitimately spans those fields, not for grouping unrelated
benchmarks.

An unavailable optional variant may declare:

```yaml
skip:
  reason: CUDA support is not enabled in this build
```

It is reported as `SKIP`, never executed, and never counted as covered. Do not
use a skip to hide a missing required implementation.

## Template Variables

The runner expands these variables in command arrays, paths, environment
values, and rendered input templates:

| Variable | Value |
| --- | --- |
| `{root}` | Repository root. |
| `{suite_dir}` | `verification_and_validation` directory. |
| `{case_dir}` | Source case directory. |
| `{case}` | Source `case.yaml`. |
| `{resolved_case}` | Per-run manifest with selected variant and merged tolerances. |
| `{output}` | Isolated variant output directory. |
| `{mesh}` | Conventional generated mesh directory under `{output}`. |
| `{build_dir}` | Selected SFEM build directory. |
| `{python}` | Repository virtual-environment Python. |
| `{variant}`, `{variant_id}` | Variant ID. |
| `{operator}` | Variant operator. |
| `{element}` | Variant element. |
| `{material_<name>}` | Physical material value declared by the case or variant. |
| `{material_key_<name>}` | Operator key declared by `material_parameter_map`, defaulting to the physical name. |
| `{resolution}` | Scalar resolution or compact JSON for a mapping. |
| `{resolution_<key>}` | One variable for each resolution mapping entry. |

The resolved manifest is emitted as YAML with scalar types preserved. Verifiers
should read `{resolved_case}` so variant tolerance overrides are visible.

## Oracle Report

The verification command must write a JSON object containing a nonempty
`checks` array. Every check requires:

```json
{
  "name": "displacement_relative_l2",
  "oracle": {"type": "analytical", "reference": "oracle.py"},
  "observed": 2.5e-12,
  "expected": 0.0,
  "error": 2.5e-12,
  "tolerance": 1.0e-10,
  "passed": true
}
```

Optional `diagnostics` and `artifacts` mappings are copied into the aggregate
`report.yaml`.
An empty report or a smoke-only check is rejected. Check names and tolerance
values must match the resolved manifest one-for-one; a verifier cannot add,
remove, or weaken a declared tolerance.

## Status and Coverage

- `PASS`: all physical checks for the variant passed.
- `FAIL`: the driver and verifier completed, but at least one physical check
  exceeded tolerance.
- `ERROR`: setup, execution, convergence-output checks, or report validation
  failed.
- `SKIP`: the manifest declared a machine-readable skip reason.

`SKIP` is non-fatal for the process but never contributes to coverage. A case
is fully covered only when every selected variant is `PASS`. The runner exits
nonzero when any selected case is `FAIL` or `ERROR`.

## Version 1 Compatibility

Version 1 retains the original top-level `mesh`, `driver`, and `verification`
layout. The runner converts it internally to one `default` variant. Optional
`family`, `dimension`, `tier`, `operator`, `element`, `resolution`, and
`oracle.type` fields improve filtering and reporting without changing
execution.

Version 1 manifests still require a nonempty tolerance mapping and oracle
provenance. They should not be used for new cases.
