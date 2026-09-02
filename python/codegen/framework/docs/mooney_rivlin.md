# Incompressible Mooney-Rivlin Codegen Example

This example shows the high-level SFEM code-generation path for an
incompressible Mooney-Rivlin material. The material definition lives in
`python/codegen/framework/materials/mooney_rivlin.py` and uses the public
`sfem.gen` API to define an energy form. The code generator then emits
`objective`, `gradient`, and `apply` kernels plus the generated `sfem::Op`
wrapper.

The material energy is the standard 3D Mooney-Rivlin form

```text
W(F) = mu * (I1 - 3 + I2 - 3) + 0.5 * lambda * (J - 1)^2
```

In 3D, `I1` and `I2` are the usual invariants of `C = F^T F`. In 2D, the
deformation is treated as plane strain by embedding into 3D with `F_zz = 1`:

```text
I1_3d = I1_2d + 1
I2_3d = I2_2d + I1_2d
J = det(F_2d) = det(F_3d)
```

This is a penalty-based nearly-incompressible displacement formulation. The
current weak-form kernel ABI exposes two scalar material parameters named `mu`
and `lmbda`; `lmbda` is the bulk penalty coefficient for the volumetric term.

## Files

- `../materials/mooney_rivlin.py`: high-level Python material example.
- `mooney_rivlin.sh`: runs code generation and compiles generated sources.
- Generated files are written to `generated/mooney_rivlin` by default.

## Run

From the repository root:

```bash
source $CODE_DIR/merge_git_repos/sfem/venv/bin/activate
python/codegen/framework/docs/mooney_rivlin.sh
```

The default element is `HEX8` with vector size `16`. To choose another supported
element:

```bash
python/codegen/framework/docs/mooney_rivlin.sh /tmp/mr_tet4 TET4 16
```

Expected outputs include:

- `kernel_diagnostics.hpp`
- `d<dim>/<element>/mooney_rivlin_<element>_operator.cpp`
- `d<dim>/mooney_rivlin_d<dim>_<family>_local.hpp`
- `op/sfem_GeneratedMooneyRivlin.hpp`
- compiled object files when `--compile` is used

The exported C ABI functions follow the standard generated naming pattern, for
example for `HEX8`:

```text
mooney_rivlin_hex8_hex8_objective_isoparametric_mesh_soa
mooney_rivlin_hex8_hex8_gradient_isoparametric_mesh_soa
mooney_rivlin_hex8_hex8_apply_isoparametric_mesh_soa
```
