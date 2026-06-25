# Material Examples

These examples contain only symbolic material definitions and weak forms. The
`sfem.gen` API owns element specialization, kernel construction, generated-file
management, diagnostics, compilation, command-line handling, and the generated
`sfem::Op` wrapper.

Each generation produces:

- dimension/family-local kernels;
- element-specialized mesh kernels;
- `kernel_math.hpp` and `kernel_diagnostics.hpp`;
- `op/sfem_<Operator>.hpp` and `op/sfem_<Operator>.cpp`.

The generated operator dispatches by `smesh::ElemType` and calls the
isoparametric mesh kernels directly. Hyperelastic operators expose `value`,
`gradient`, and matrix-free `apply`; coupled residual operators expose
`gradient` and Jacobian-action `apply`, with the previous state supplied through
`set_field("previous", ...)` or `update(previous, current)`.

Run an example from the repository root:

```bash
PYTHONPATH=python python -m codegen.framework.materials.neohookean_ogden \
    --out-dir /tmp/neohookean --element HEX8 --compile

PYTHONPATH=python python -m codegen.framework.materials.two_phase_flow \
    --out-dir /tmp/two_phase_flow --element HEX8 --compile
```
