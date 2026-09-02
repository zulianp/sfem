# Matrix-Format Generation

Matrix-format requests are optional generator inputs. They do not belong in
symbolic material definitions; the same forms can be generated with or without
assembled matrix artifacts.

## CLI Examples

Generate CRS assembly artifacts:

```bash
PYTHONPATH=python venv/bin/python -m codegen.framework.generators.neohookean_ogden \
  --element HEX8 \
  --matrix-format crs
```

Generate BSR, DIA, COO, and patch variants:

```bash
PYTHONPATH=python venv/bin/python -m codegen.framework.generators.neohookean_ogden \
  --element HEX8 \
  --matrix-format bsr \
  --matrix-format dia \
  --matrix-format coo \
  --matrix-format patch
```

Generate every maintained matrix format:

```bash
PYTHONPATH=python venv/bin/python -m codegen.framework.generators.neohookean_ogden \
  --element HEX8 \
  --matrix-format all
```

Request standard and packed-layout metadata:

```bash
PYTHONPATH=python venv/bin/python -m codegen.framework.generators.neohookean_ogden \
  --element HEX8 \
  --matrix-format all \
  --matrix-layout all \
  --packed-pass all \
  --patch-node-index-filter \
  --dump-plan
```

`--matrix-format`, `--matrix-layout`, and `--packed-pass` accept repeated values
or comma-separated lists. `all` expands to the maintained choices for that
option. Packed layouts require `one_pass`, `two_pass`, or `all`; `none` is only
valid for standard layout and is not accepted as a packed-pass request.

## Python API Examples

Diagnostics-only metadata can be requested for forms that do not yet emit real
assembly kernels:

```python
from sfem import gen
from codegen.framework.materials.laplace import material

result = gen.generate(
    material,
    "/tmp/sfem_laplace_matrix_formats",
    elements=("TRI3",),
    matrix_formats=("crs", "bsr", "dia", "coo", "patch"),
    matrix_mesh_layouts=("standard", "packed"),
    matrix_packed_passes=("one_pass", "two_pass"),
    dump_plan=True,
)
```

Energy-style generated operators with a Hessian action form can request real
assembly entry points. Generated NeoHookean is the default runtime regression
material:

```python
from sfem import gen
from codegen.framework.materials.neohookean_ogden import material

result = gen.generate(
    material,
    "/tmp/sfem_neohookean_matrix_formats",
    elements=("HEX8",),
    matrix_formats="crs,bsr,dia,coo,patch",
    compile=True,
)
```

Generated wrappers expose CRS and BSR assembly through the SFEM frontend
factory path. They also expose concrete `hessian_dia`, `hessian_coo`, and
`hessian_patch` methods that route to the generated C ABI entry points when
those standard-layout formats are available for the generated form. Requests
for formats that are not integrated into the selected frontend operator path
fail with a runtime error instead of falling back to another matrix format.
The generated C ABI diagnostics include the assembly kind, index policy, value
layout, accumulation policy, structural compatibility, reduction policy, block
size, and block entry counts used by the selected matrix-format layout plan.

## Benchmark Reports

Plan dumps can be converted to per-format CSV diagnostics. Provide measured
elapsed time and repeat count when a benchmark harness has already timed the
assembly call:

```bash
PYTHONPATH=python venv/bin/python -m codegen.framework.scripts.matrix_format_benchmark_report \
  /tmp/sfem_neohookean_matrix_formats/neohookean_ogden_plan.json \
  --nelements 100000 \
  --elapsed-seconds 0.25 \
  --repeat 10
```

The report includes the selected assembly layout fields plus total FLOPs,
total bytes, arithmetic intensity, seconds-per-call, bandwidth, and achieved GFLOP/s
columns.
