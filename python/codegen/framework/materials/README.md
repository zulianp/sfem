# Material Examples

These examples contain only symbolic material definitions and weak forms. The
`sfem.gen` API owns element specialization, kernel construction, generated-file
management, diagnostics, compilation, and command-line handling.

Run an example from the repository root:

```bash
PYTHONPATH=python python -m codegen.framework.materials.neohookean_ogden \
    --out-dir /tmp/neohookean --element HEX8 --compile

PYTHONPATH=python python -m codegen.framework.materials.two_phase_flow \
    --out-dir /tmp/two_phase_flow --element HEX8 --compile
```
