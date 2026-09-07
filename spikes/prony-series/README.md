# Prony-series viscoelastic torsion, with release

A self-contained spike on top of an installed SFEM. It builds one executable,
`prony_visco_torsion`, which drives SFEM's Prony-series viscoelastic Mooney-Rivlin material
(`sfem::MooneyRivlinVisco`) through a scenario described entirely in a YAML file: a beam is
twisted about its axis, the twist is held while the reaction torque relaxes through the
relaxation spectrum, and then the grip is **released** so the beam recovers.

The driver is the viscoelastic counterpart of `drivers/mech/hyperelasticity_bdf2.exe.cpp` and
keeps its shape -- predictor, Newton with an assembled tangent, per-step export, control-point
CSV. What is new here is the scenario description and the releasable torsion condition.

## Building

```bash
cmake -S . -B build -DSFEM_DIR=<sfem-install-prefix>/lib/cmake
cmake --build build -j
```

A single `-DSFEM_DIR` is enough: the spike picks `ryml`, `matrixio`, `smesh`, `SCCD` and `ssdf`
out of the same prefix, exactly as `spikes/cvfem` does.

## Running it

```bash
cmake -S . -B build -DSFEM_DIR=<sfem-install-prefix>/lib/cmake
cmake --build build -j
```

Then any case, end to end — mesh, solve, ParaView export, analysis:

```bash
./run_case.sh                     # list the cases
./run_case.sh creep_torsion       # watch the structure creep under a held torque
./run_case.sh cyclic_torsion      # the hysteresis loop and the loss angle
./run_case.sh torsion_release     # twist, hold, release, recover
./run_case.sh <case> <workdir>    # somewhere other than ./output
```

The mesh is built once per workdir and reused, so running several cases into the same one only
pays for it the first time. Each case picks up the analysis that suits it: the relaxation checks
and the Prony recovery for a held twist, the creep checks under a held torque, the hysteresis
fit for a cyclic drive.

The original scenario also has its own script, which additionally runs the never-released
control and compares the two:

```bash
./run_torsion_release.sh
PRONY_RUN_CONTROL=1 ./run_torsion_release.sh
```

And the characterisation sweeps:

```bash
venv/bin/python tools/sweep_severity.py --sweep all
venv/bin/python tools/sweep_severity.py --sweep frequency   # the simulated DMA
```

The script writes a HEX8 beam, copies the case files next to it, runs the driver from that
directory and checks the resulting history. Everything lands under `output/`:

```text
output/mesh/                       the beam, with left/right sidesets
output/torsion_release.yaml        the case, as it ran
output/results_release/history.csv time, angle, released, torque, control-point displacement
output/results_release/out/        transient displacement fields
output/results_release/output.xdmf ParaView-ready transient pair
output/logs/torsion_release.log    the complete unfiltered driver log
```

To run the driver by hand, always go through the wrapper rather than composing a pipeline:

```bash
PRONY_EXE=build/prony_visco_torsion tools/pronyrun.sh mytag cases/torsion_release.yaml
```

It tees a complete raw log and line-buffers the filtered view, so a long run is observable
while it runs instead of only when it ends.

## The case file

One document describes the whole experiment. `cases/torsion_release.yaml` is the scenario and
`cases/torsion_hold.yaml` the control; the keys are:

| block | meaning |
| --- | --- |
| `mesh` | SFEM mesh folder. **HEX8 only** -- the viscoelastic kernels are dispatched for `smesh::HEX8` and nothing else (`operators/mooney_rivlin_visco.cpp`). |
| `material` | `C10`, `C01`, `K`, a `prony` sequence of `{g, tau}` terms, and optional `wlf` time-temperature superposition. |
| `time` | `dt`, `t_end`. `dt` is fixed for the whole run: the Prony coefficients `alpha_i = exp(-dt/tau_i)` are computed from it once. |
| `dynamics` | `quasi_static` (default), `bdf2` or `newmark`, with `density` and — for Newmark — `beta` and `gamma`. See **Integrators** below. |
| `torsion` | `sideset`, `axis`, `rotation_center`, a `release` block, a `profile` (`ramp` with `ramp_time`, or `cyclic` with `period`), and a `control`: `angle` (default, prescribing `angle`) or `torque` (prescribing `torque` and solving for the twist). |
| `solver` | Newton and linear-solver settings. |
| `output` | where fields and the history CSV go, plus a `control_point`. |
| `dirichlet_conditions` | SFEM's own dirichlet schema, inline. `dirichlet_file:` names a separate file instead. |

Paths inside the document are resolved relative to the document's own directory, **except**
those inside `dirichlet_conditions`, which SFEM resolves against the working directory. Run the
driver from the directory that holds the case and the two agree; `run_torsion_release.sh` does.

### The release

```yaml
torsion:
  sideset: mesh/surface/sidesets/right
  axis: x
  rotation_center: [0.0, 0.1, 0.1]
  angle: 0.6
  ramp_time: 1.0
  release:
    mode: free       # remove the constraint; the face becomes traction free
    time: 15.0
```

`mode: free` removes the torsion constraint from the `sfem::Function` at `release_time`. The
face stops being prescribed at all, so it is traction free from that step on, and the twisted
configuration reached at release becomes the initial state of the recovery. The clamped end
still removes the rigid body modes. `mode: hold` never releases, which is the control case for
pure stress relaxation.

Under the hood this is `sfem::Rotate<Axis0, Axis1>` with its step counter turned into a direct
angle setter, so the prescribed twist follows physical time rather than an integer step index
(`prony_torsion_bc.hpp`).

## What the history says

The torque reported per step is the moment about the rotation axis that the grip exerts on the
body, summed over the twisted sideset with the lever arm taken in the deformed configuration.
It has to be read off the **operator** gradients, not off `Function::gradient`, which finishes
by calling `constraints_gradient` and overwrites every constrained entry with the constraint
violation -- zero once the constraint is satisfied, which is exactly where the reaction lives.

`tools/validate_torsion_release.py` checks the three phases against each other rather than
against reference numbers, so it stays meaningful when the material parameters change:

* while the twist is held, the torque must decay, and monotonically;
* after release, the torque must fall to the residual floor of the nonlinear solve;
* the control point must keep recovering after release rather than being finished at the
  first step, which is what a purely elastic body would do.

With `--control` it also checks that the released run and the held control agree before the
release, so the difference after it can be attributed to the release and nothing else.

## Recovering the Prony parameters from the run

The torque history is not just a plot: the relaxation function can be read back out of it and
compared with the one the case asked for. That is what makes this a measurement rather than a
demonstration, and it is what `--case` turns on in the validator.

Under quasi-linear viscoelasticity the stress is `sigma(t) = int_0^t G(t-s) d sigma_el(s)`, so
with the `tau_i` known from the case file the hold-phase torque is *linear* in the unknowns and
an ordinary least-squares fit in `1 + n_terms` coefficients recovers them. No optimiser, no
initial guess, nothing to tune.

The one trap is worth naming, because getting it wrong looks like a material error rather than
an analysis error. **The decay after a ramp is not the relaxation function translated to the end
of the ramp.** A term whose `tau` is comparable to the ramp has already done much of its
relaxing before the hold begins, so its apparent weight is suppressed and the slower terms
absorb the difference. The exact correction for a linear ramp is a factor
`R_i = (tau_i / t_ramp) (exp(t_ramp / tau_i) - 1)` on each coefficient, and the exponentials must
be measured from `t = 0` rather than from the end of the ramp. In this spike's default case
`tau_fast` and `ramp_time` are both `1.0`, and ignoring this reads `g[0]` as `0.30` instead of
`0.40` — a 25% error that looks exactly like a broken material.

With the correction, the default scenario returns:

| | asked | recovered |
| --- | --- | --- |
| `g_inf` | 0.3000 | 0.3015 |
| `g[0]` (tau=1) | 0.4000 | 0.3989 |
| `g[1]` (tau=10) | 0.3000 | 0.2996 |

at a relative fit rms of `9.9e-6`.

## Watching the structure adapt: creep

A prescribed twist is the wrong experiment for this. Under displacement control the structure
does not visibly adapt at all — measured on this beam over the hold, **the reaction torque
decays 63% while the interior displacement drifts 0.33%**. That is not a parameter choice and no
amount of tuning `tau` changes it: at fixed boundary displacement, relaxation scales the
deviatoric stress nearly uniformly, so the displacement field satisfying equilibrium is
essentially unchanged. The relaxation is in the stress, not in the shape.

`torsion.control: torque` prescribes the moment instead and solves for the twist, and then the
structure genuinely keeps turning. The driver wraps the nonlinear solve in a scalar secant
iteration on the angle, warm-started from the previous step, so it costs one or two extra solves
per step. `cases/creep_torsion.yaml`:

| t / s | twist / rad | × θ₀ | tip \|u_yz\| | torque |
| --- | --- | --- | --- | --- |
| 0.2 | 0.4979 | 1.000 | 0.0697 | 8.0000e-05 |
| 1 | 0.6265 | 1.258 | 0.0872 | 8.0000e-05 |
| 3 | 0.8073 | 1.621 | 0.1111 | 8.0000e-05 |
| 10 | 1.0584 | 2.126 | 0.1428 | 8.0000e-05 |
| 30 | 1.3777 | 2.767 | 0.1798 | 8.0000e-05 |
| 60 | 1.5263 | 3.065 | 0.1955 | 8.0000e-05 |
| 120 | 1.5692 | 3.152 | 0.1998 | 8.0000e-05 |

The twist triples and the tip sweeps from 0.070 to 0.200 under a torque held constant to eight
figures. That is the slow adaptation, and it is visible in the exported fields, not just in the
history.

**The creep factor is `1/g_inf`**, and checking that turned up the same trap twice over. Run out
to `t = 250` with a ramp short against `tau_fast`, the factor reaches **3.250** against the
theoretical **3.3333**; the residual 2.5% is the ramp bias that remains at a 0.1 s ramp. Two
wrong explanations were tested and discarded on the way:

* *Geometric stiffening.* Refuted by amplitude: at torques of 8e-5, 8e-6 and 8e-7 — the last a
  twist of 0.006 rad, entirely linear — the factor sat at 2.768, 2.792, 2.785. It did not
  approach the limit as the amplitude fell, so it was not nonlinearity.
* *The unrelaxed volumetric stress*, which does bias the relaxation fit and the loss angle.
  Refuted by sweeping `K` over three decades, `K/C10` from 2 to 1667: the factor moved only
  2.801 to 2.774.

What it actually was: **θ₀ was being read at the end of a 1 s ramp while `tau_fast` is also 1 s**,
so the reference twist had already crept. A 0.05 s ramp reads 3.19 where a 1 s ramp reads 2.72.
It is the identical mistake to reading a relaxation curve as though its ramp were a step, and the
shipped case ramps the torque over a single step because of it.

## Seeing the viscoelastic lag

The relaxation scenario shows viscoelasticity as a torque that decays at fixed twist. The lag
during *loading* is there too, but it is easy to miss, for three reasons: the default case takes
only four samples through its ramp, nothing in the output surfaces it, and at 0.6 rad it is
confounded with geometric nonlinearity, which softens an elastic body too.

Against a matched elastic run — same mesh, same twist, `prony` omitted — it is unambiguous. The
ratio of the two torques falls monotonically through the ramp as the earlier twist relaxes while
the twist keeps growing:

| twist / rad | 0.28 | 0.56 | 0.84 | 1.40 | 1.96 | 2.52 | 2.80 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T_visco / T_elastic | 0.960 | 0.924 | 0.893 | 0.840 | 0.797 | 0.764 | 0.749 |

But "lag" properly means a *phase* lag, and for that the twist has to cycle. `torsion.profile:
cyclic` drives `angle(t) = angle * sin(2 pi t / period)`, and
`tools/analyse_hysteresis.py` reads the answer off the last cycle: the torque-angle curve opens
into a hysteresis loop instead of retracing itself, the torque leads the twist, and the loop area
is the energy dissipated per cycle.

```bash
venv/bin/python tools/analyse_hysteresis.py <workdir>/results_cyclic/history.csv \
    --case cases/cyclic_torsion.yaml
```

This is checkable, not merely viewable. Fitting the fundamental `T = a0 + a1 cos(wt) + b1 sin(wt)`
against a drive `theta = A sin(wt)` gives `delta = atan2(a1, b1)`, and the Prony series predicts
it in closed form:

```text
G'(w)  = g_inf + sum_i g_i (w tau_i)^2 / (1 + (w tau_i)^2)
G''(w) =         sum_i g_i (w tau_i)   / (1 + (w tau_i)^2)
tan delta = G'' / G'
```

`--sweep frequency` runs that across the spectrum — a simulated DMA, the same complex-modulus
model `python/sfem/regression/regression_prony.py` fits to measured data:

| period / s | omega | delta measured | delta predicted | ratio | tan delta | loop area | fit rms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | 0.105 | 22.413° | 22.523° | 0.995 | 0.4124 | 2.26e-6 | 4.8e-5 |
| 20 | 0.314 | 18.212° | 18.291° | 0.996 | 0.3290 | 2.37e-6 | 5.9e-4 |
| 12 | 0.524 | 17.949° | 18.013° | 0.997 | 0.3239 | 2.59e-6 | 1.3e-3 |
| 4 | 1.571 | 12.731° | 12.770° | 0.997 | 0.2259 | 2.36e-6 | 8.6e-4 |
| 2 | 3.142 | 7.375° | 7.405° | 0.996 | 0.1294 | 1.47e-6 | 8.3e-4 |

The lag tracks theory to within half a percent over a thirtyfold range of frequency, and it
vanishes at both ends of the spectrum — at `w tau << 1` the material relaxes fully within a
cycle, at `w tau >> 1` it has no time to relax at all. The peak in between is the signature of
the relaxation times themselves.

The measured angle sits consistently *below* the prediction, by 0.3-0.5%, and that is expected
rather than error: the series relaxes only the deviatoric stress, so the unrelaxed volumetric
part adds to storage without adding to loss. It is the same mechanism as the bias in the
relaxation fit, and it moves the loss angle the same way. The reported fit residual is the guard
on the whole measurement — a large residual means the loop is not elliptical (geometric
nonlinearity at large amplitude) and the phase is not meaningful, which is why the shipped case
uses a 0.15 rad amplitude rather than the relaxation case's 0.6.

## Severity sweeps

`tools/sweep_severity.py` characterises the material by sweeping the severity of each thing the
model approximates, with a control at zero severity, rather than by quoting one run. Three axes:

```bash
venv/bin/python tools/sweep_severity.py --sweep all
```

**Numerical severity — `dt` at fixed `tau`.** The recursion is `alpha_i = exp(-dt/tau_i)`, so
`dt/tau_fast` is how coarsely the fastest mode is resolved. The recovered weights are identical
to four figures at `dt = 0.2`, `0.1` and `0.05` (`0.3017 / 0.3987 / 0.2996`) and only
`dt = 0.4` departs (`0.2977 / 0.4059 / 0.2963`). Cost is linear in the step count — 22 s, 37 s,
71 s, 142 s at 900 dof and 176 elements on one thread — so the default `dt = 0.2` is where this
axis stops buying anything.

**Physical severity — the twist angle.** Swept over an eightfold range, `0.15` to `1.2` rad, the
recovered weights move by less than `0.001` (`0.3014 -> 0.3021` on `g_inf`) and the fit rms grows
only from `4.1e-5` to `5.6e-5`. The material is quasi-linear across the whole range this scenario
uses, which is worth knowing and was not previously established.

**Model severity — the bulk modulus.** Neither of the above removes a systematic bias of a few
parts in a thousand, and this axis explains it: the Prony series multiplies only the
*deviatoric* stress, so the torque is `g(t) T_dev + T_vol` with an unrelaxed volumetric part that
the fit has no choice but to absorb into `g_inf`. Sweeping `K` shows exactly that signature —
`g_inf` rising and the `g_i` falling as the material stiffens volumetrically:

| K | g_inf | g[0] | g[1] | fit rms | max error |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.3007 | 0.3995 | 0.2999 | 2.1e-5 | 7e-4 |
| 50 | 0.3017 | 0.3987 | 0.2996 | 4.3e-5 | 1.7e-3 |
| 500 | 0.3055 | 0.3958 | 0.2987 | 1.3e-4 | 5.5e-3 |
| *asked* | *0.3000* | *0.4000* | *0.3000* | | |

So the recovery is exact in the incompressible-stress sense and the residual is a known,
quantified property of reading a deviatoric relaxation function off a total torque — not an
error in the integration or in the fit.

**Zero-severity control.** With `prony` omitted the solid is elastic and the torque through the
hold must be exactly constant. It is, to `0.000%`. Every number above is gated on that.

## Other cases

Beyond `torsion_release.yaml` and its `torsion_hold.yaml` control:

| case | what it exercises |
| --- | --- |
| `wlf_reference.yaml`, `wlf_warm.yaml` | Time-temperature superposition. At `a_T = 1` the peak torque is `8.43e-5` and relaxes 59% over the hold; at `a_T = 23.9` the peak is already down to `4.35e-5` — most of the fast relaxation happened during the ramp — with only 30% left to shed. The temperatures sit close to `T_ref` deliberately: these EPDM constants are steep enough that by `-45 C` the shift is 526, every term satisfies `dt/tau_eff > 30`, and the material filters all of them out as fully relaxed, making two warm runs literally the same problem. |
| `bdf2_torsion_release.yaml` | `dynamics: {type: bdf2}`, i.e. `sfem::BDF2InertiaPotential`. |
| `newmark_torsion_release.yaml` | `dynamics: {type: newmark}`, i.e. `sfem::NewmarkInertiaPotential` — the integrator the Mooney-Rivlin tests in `frontend/tests` use, so results are directly comparable with theirs. See below. |
| `creep_torsion.yaml` | Constant torque, twist solved for. The case to use to watch the structure itself keep turning. See **Watching the structure adapt**. |
| `cyclic_torsion.yaml` | Sinusoidal twist at a 12 s period, for the hysteresis loop and the loss angle. See **Seeing the viscoelastic lag**. |
| `axis_z_torsion.yaml` | The runtime axis selection between the three `sfem::Rotate<Axis0, Axis1>` instantiations. A code-path check, not a physically interesting scenario. |

## Integrators

Three, selected by `dynamics.type`:

| | what it is | when |
| --- | --- | --- |
| `quasi_static` | no inertia at all | the default, and the right regime for a relaxation experiment — the interesting time scales are the `tau_i`, not an elastic transit time |
| `bdf2` | `sfem::BDF2InertiaPotential`, as in `drivers/mech/hyperelasticity_bdf2` | second order and L-stable, so it damps high frequencies numerically — convenient across the step change at release, misleading if the ring-down is what you came for |
| `newmark` | `sfem::NewmarkInertiaPotential`, as in `frontend/tests/sfem_MooneyRivlin*Test.cpp` | when the ring-down matters, or to compare directly against those tests |

The two inertia operators are *the same code* apart from a `density == 0` shortcut in BDF2's
`initialize`. Neither carries a scheme: both are a lumped-mass penalty
`1/2 alpha (u - u_hat)^T M (u - u_hat)`, and which integrator you have comes entirely from the
`alpha` and `u_hat` the driver feeds them. The Newmark arithmetic here is lifted verbatim from
`drivers/mech/mooney_rivlin_kelvin_voigt_newmark`, which agrees line for line with the
hand-rolled scheme in the gravity test once `a_{n+1} = (u - u_hat)/(beta dt^2)` is substituted
into its velocity update.

**Choosing beta and gamma.** Newmark is unconditionally stable only for
`beta >= gamma/2 >= 1/4`. `(1/4, 1/2)` is the trapezoidal rule — second order, exactly
non-dissipative — and is what the frontend tests use. It is *not* what
`newmark_torsion_release.yaml` uses, for a reason worth recording: a linear twist ramp starts
with a velocity discontinuity, which is an impulse, and a scheme with no damping rings on it
forever at the `2*dt` period. Measured on this beam the reaction torque then alternates sign
every step at an amplitude two orders above the signal, and it does so identically at `K = 5` and
`K = 50` — that independence from the bulk modulus is what identifies it as the integrator
rather than a stiff physical mode.

The instinct on seeing that is to raise `gamma` for damping, and **raising `gamma` alone is
exactly what leaves the stable region**: `(0.25, 0.6)` and `(0.25, 0.7)` both diverge on the
first step, which reads like a solver failure rather than the parameter error it is. The case
parser therefore refuses any pair with `beta < gamma/2` and names the `beta` to use. The shipped
case takes `gamma = 0.6` with `beta = (1+gamma)^2/4 = 0.64`, the standard damped pairing, which
removes the ringing entirely:

| beta | gamma | stable? | result |
| --- | --- | --- | --- |
| 0.25 | 0.5 | marginal | runs, but rings at `2*dt` off the impulsive start |
| 0.25 | 0.6, 0.7 | no | diverges on the first step — refused by the parser |
| 0.64 | 0.6 | yes | clean |
| 0.7225 | 0.7 | yes | cleanest |

**What a dynamic run costs you.** Two things the quasi-static case gives and a dynamic one does
not. The reaction torque now carries the inertial term, so during the hold it is neither
monotone nor proportional to `G(t)` — the validator detects a dynamic case and skips both the
relaxation checks and the Prony recovery rather than reporting nonsense (run the fit on dynamic
data anyway and its relative rms goes from `9.9e-6` to `0.87`, which is the fit saying so). And
the tangent gains `alpha*M` on its diagonal, which regularises it — precisely how the three
defects below stayed hidden in SFEM's own dynamic tests. **Never gate the operator on a dynamic
case**; that is what `prony_tangent_check` is for.

What you get in exchange is the ring-down: after release the control point swings from
`uy = -0.073` through zero to `+0.014`, four direction reversals, where the quasi-static run
recovers monotonically.

## What this spike found

### Three defects in the viscoelastic operator, found by this spike and fixed

`prony_tangent_check` sweeps the step size in a central-difference test of
`MooneyRivlinVisco::hessian_bsr` against `MooneyRivlinVisco::gradient`. A correct tangent traces
a V -- error falling as `eps^2`, then rising with round-off; a wrong one sits on a floor no step
size gets below. It found three real defects, all now fixed in SFEM proper.

The sharpest evidence was a severity sweep on the amount of accumulated Prony history, since
`S_hist` is identically zero before any history exists:

| history steps | relative error |
| --- | --- |
| 0 | 2.2e-11 (exact) |
| 1 | 7.8e-5 |
| 5 | 4.0e-4 |
| 20 | 3.4e-3 |

Zero at zero severity and monotone in it: the fault was entirely in the tangent's history term,
which is a pure geometric stiffness `I (x) S_hist`.

1. **Element-matrix ordering** (`python/codegen/sr_visco_hyper_unique_Hi.py`,
   `__compute_geom_stiff_single`). The geometric term was scattered node-major,
   `row = test_node*dim + d`, into a matrix that the rest of the codebase — `FE.SoA` in
   `grad_tensorize`, the algorithmic hessian beside it, and `hex8_local_to_global_bsr3` which
   reads it back — treats as component-major. A permutation of the correct entries: symmetric,
   right trace, wrong tangent, which is why no scalar fit could absorb it. Fixing it took the
   cube from `4.0e-4` to `1.9e-11`.

2. **Transposed inverse Jacobian**, same function. It contracted `Jinv[d,k]` where
   `FE.physical_grad` and the algorithmic metric tensor both contract the first index. This is
   invisible on an axis-aligned mesh, where `J` is diagonal and `J^-1` is its own transpose —
   which is why the gate had to grow a distorted-mesh variant before the defect could be seen at
   all. With ordering fixed and this one not, the cube read `1.9e-11` and the distorted mesh
   `3.6e-5`; fixing it took the distorted mesh to `1.8e-11` and left the cube untouched.

3. **Diagonal extraction** (`operators/hex8/hex8_mooney_rivlin_visco_flexible.cpp`).
   `hessian_diag` read node-major positions out of the same component-major element matrix, so
   it returned the right set of diagonal values assigned to the wrong degrees of freedom — 8.6%
   wrong in norm, and wrong with **no Prony terms at all**. Nothing covered it: the two frontend
   tests that use this material extract the diagonal from the assembled BSR by hand instead of
   calling `hessian_diag`.

A fourth suspicion — a wrong power of the determinant, from the geometric kernel using
`symbol_jacobian_inverse_as_adjugate()` together with `det` in `dV` — was checked and dismissed.
That accessor already divides by the determinant despite its name, and the algorithmic path uses
the identical convention.

**Why the existing tests never caught any of this.** `sfem_MooneyRivlinGravityTest` and
`sfem_MooneyRivlinViscoTest` are dynamic Newmark runs that add `c0 M` to the tangent diagonal
with `c0 = 1/(beta dt^2)` reaching several hundred; that term dominates and regularises a `4e-4`
material error away. The other five `MR*Validation` programs return zero unconditionally — they
print tables and always pass. The permanent gate added at
`frontend/tests/sfem_MRViscoTangentFDTest.cpp` deliberately uses the operator alone: no mass
term, no constraints, no `Function`.

**What it bought.** Measured on this spike's scenario at 1200 dof, 240 HEX8 elements, 200 steps,
one thread:

| | before | after |
| --- | --- | --- |
| release step, Newton iterations | 180 | 65 |
| release step, linear iterations | 178,211 | 37,943 |
| first recovery step, Newton | 96 | 21 |
| total linear iterations | 2,103,506 | 320,357 |
| wall clock | 52 s | 11 s |

The physics is unchanged, as it must be — a wrong tangent does not move the root of the
residual, only the cost of finding it. The relaxation and recovery figures are identical and the
post-release torque floor improved from `3.0e-10` to `3.7e-12`, which is the better-converged
solve showing through.

### At this problem size, OpenMP costs more than it saves

Every CG iteration goes through the constraint handling, whose loops run over a few hundred
boundary nodes. Starting and joining an OpenMP team for a loop that short costs far more than
the loop itself. Measured on a 2160-dof beam (720 nodes, 475 HEX8 elements), 12 steps of the ramp phase,
72,943 CG iterations, Apple M-series laptop:

| threads | wall | user | sys |
| --- | --- | --- | --- |
| 1 | 3.07 s | 3.00 s | 0.02 s |
| 4 | 9.66 s | 5.76 s | 6.83 s |
| 8 | 27.46 s | 13.34 s | 37.30 s |

The whole difference is system time, which is thread barriers. `run_torsion_release.sh`
therefore sets `OMP_NUM_THREADS=1` by default; raise it for a mesh large enough to pay for the
teams. A trace of the eight-thread run of that same 2160-dof case puts 17.9 s of the 41.8 s total in
`Function::copy_constrained_dofs` against 4.8 s in the BSR mat-vec it exists to correct, while
the whole element assembly -- `MooneyRivlinVisco::hessian_bsr` -- accounts for 0.1 s.

## Layout

```text
prony_visco_torsion.cpp   the driver
prony_case.hpp/.cpp       the YAML case description and its parser
prony_torsion_bc.hpp      time-scheduled, releasable torsion built on sfem::Rotate
prony_tangent_check.cpp   central-difference gate on the operator's tangent and its diagonal
cases/                    torsion_release.yaml and the torsion_hold.yaml control
tools/pronyrun.sh         run wrapper that keeps a long run's diagnostics observable
tools/validate_torsion_release.py   phase, creep and dynamic checks plus the Prony recovery fit
tools/sweep_severity.py   the dt / angle / bulk / frequency sweeps and the elastic control
tools/analyse_hysteresis.py  loss angle and dissipated energy from a cyclic run
run_case.sh               run any case in cases/ end to end
run_torsion_release.sh    the original scenario, with its never-released control
```
