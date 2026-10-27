# Playground examples

The cases below live in `workflows/`. Most come with a `run.sh` that builds `playground` if
needed, runs the case into an output folder, and writes an XDMF file for ParaView:

```sh
workflows/elastodynamics/run.sh [output-dir] [key=value ...]
```

Every `key=value` after the output folder is passed on as `--set key=value`. The scripts use the
build in `build64/` unless `SFEM_BUILD_DIR` names another, and skip building with
`SFEM_SKIP_BUILD=1`.

## Dynamics and contact

| Case | What it shows |
|---|---|
| `workflows/elastodynamics/case.yaml` | A linear elastic block falling under gravity onto a hemisphere: semi-structured refinement (`mesh.refine: 4`), the `ssmgc` obstacle, Newmark, every output field and the contact probe. The header notes host and GH200 timings. |
| `workflows/mooney_rivlin_gravity/contact.yaml` | A viscoelastic block dropped onto a hemisphere with the `node_sdf` obstacle and the contact-stabilised Newmark scheme; `gap` and `contact_force` output. The folder's README explains the contact method. |
| `workflows/mooney_rivlin_gravity/cantilever.yaml` | A Prony Mooney-Rivlin cantilever under gravity (`run.sh cantilever`). |
| `workflows/kelvin_voigt/case.yaml` | Linear Kelvin-Voigt viscoelasticity, Newmark with the consistent mass, a traction, a control-point probe. |
| `workflows/mooney_rivlin_kelvin_voigt/pulse.yaml` | Mooney-Rivlin Kelvin-Voigt under a half-sine traction pulse (`loads.profile`), with `material_reaction` output. |

## Viscoelastic torsion (Prony series)

`workflows/prony_torsion/cases/` holds torsion tests of a Prony Mooney-Rivlin bar, run with
`workflows/prony_torsion/run_case.sh <case> [workdir]`, which also builds the mesh and checks
the result (see its README):

| Case | What it shows |
|---|---|
| `torsion_hold.yaml`, `torsion_release.yaml` | A rotation ramped, held, then released (`constraints[].release`); `history_storage: int8_scaled`; the torque probe. |
| `creep_torsion.yaml` | A torque-controlled rotation (`control.torque`): creep. |
| `cyclic_torsion.yaml` | A `sine` profile: hysteresis. |
| `axis_z_torsion.yaml` | Rotation about `z`. |
| `bdf2_torsion_release.yaml`, `newmark_torsion_release.yaml` | The same in `dynamic`, with BDF2 and with damped Newmark (gamma = 0.6, beta = 0.64). |
| `wlf_reference.yaml`, `wlf_warm.yaml` | Time-temperature shift (`wlf`, `temperature`). |

## Hyperelasticity on meshes from files

| Case | What it shows |
|---|---|
| `workflows/hyperelasticity/hyperelasticity.yaml` | The hand-written NeoHookean-Ogden on TET10 with a matrix-free solver, a rotation over 24 steps. |
| `workflows/hyperelasticity/hyperelasticity_hex8.yaml` | The packed HEX8 variant (`NeoHookeanOgdenPacked`). |
| `workflows/hyperelasticity/active_strain_hex8.yaml` | An active-strain inclusion (`fields`) with a time profile. |
| `workflows/hyperelasticity/hyperelasticity_torus.yaml`, `hyperelasticity_blade.yaml` | Steady solves on a torus and a blade. |
| `workflows/bone/bone.yaml`, `workflows/cables/cables.yaml` | Steady NeoHookean-Ogden on meshes with file sidesets. |

These cases expect their meshes to be prepared by the scripts beside them.

## A large transient contact case

`workflows/ldp_containment/transient/` holds a two-stage run on an industrial mesh (see its
README): `sim1.yaml` presses a part into place quasi-statically against an obstacle given as an
SDF grid (`mesh.blocks`, disc sidesets, a 15-term Prony EPDM, `obstacle.shift`, `cutoff`,
`release`); `sim2.yaml` starts from sim1's displacement (`initial_conditions`) and runs BDF2
dynamics under a tabulated pressure, with single-precision tangents (`storage_bytes: 4`) and
named probes. `run_transient.sh <case>` runs both stages; the meshes are not in the repository.

## More cases

- The Playground tests, `src/frontend/tests/sfem_PlaygroundCasesTest.cpp` and
  `sfem_PlaygroundContactTest.cpp`, hold small inline cases for most features.
- `verification_and_validation/` builds playground cases from its manifests and checks them
  against analytic solutions; see its README.
