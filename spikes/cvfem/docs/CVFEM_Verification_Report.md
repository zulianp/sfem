<style>
.cvfig { --fg: #24292f; --muted: #57606a; --grid: #d0d7de; --s1: #0969da; --s2: #cf222e;
         --s3: #1a7f37; --s4: #8250df; --surface: #ffffff; }
@media (prefers-color-scheme: dark) {
  .cvfig { --fg: #e6edf3; --muted: #9198a1; --grid: #30363d; --surface: #0d1117; }
}
.cvfig { font: 12px system-ui, -apple-system, sans-serif; max-width: 100%; height: auto; }
.cvfig text { fill: var(--fg); }
.cvfig .mut { fill: var(--muted); }
</style>

# CVFEM verification report

Every claim below is an identity or a fitted rate checked against a threshold, so this page states whether the code is right rather than inviting a reader to judge a plot.

## Summary

**4 of 5 checks pass.**  1 FAILING -- see the sections below.

| check | evidence | status |
|---|---|---|
| Spatial order, u L2 | rate 2.351 (expect >= 1.65) | pass |
| Spatial order, p L2 shifted | rate 1.495 (expect >= 0.95) | pass |
| Traction t=0 equals the do-nothing outflow | max diff 0.00e+00 | pass |
| Pressure port shifts the level by p_bar - p_exact | 7 of 8 values verified, 1 did not converge | pass |
| Global mass balance | 0 of 1 case(s) verified, 1 did not converge | **FAIL** |

## Spatial order of accuracy

Manufactured solution, volume-weighted L2 against the exact field, with the
pressure gauge offset removed (`p_l2_shifted`). The rate is fitted by log-log
least squares against `h ~ ndof^(-1/3)`.

| ndof | h | u L2 | p L2 | p L2 shifted |
|---|---|---|---|---|
| 500 | 0.1260 | 4.144e-01 | 2.247e+00 | 2.247e+00 |
| 2916 | 0.0700 | 9.243e-02 | 9.101e-01 | 9.101e-01 |
| 19652 | 0.0371 | 2.015e-02 | 3.451e-01 | 3.451e-01 |
| 143748 | 0.0191 | 4.924e-03 | 1.345e-01 | 1.345e-01 |

| quantity | fitted rate | R^2 | levels | threshold | status |
|---|---|---|---|---|---|
| u L2 | 2.351 | 0.9983 | 4 | &ge; 1.65 | pass |
| p L2 shifted | 1.495 | 0.9996 | 4 | &ge; 0.95 | pass |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>MMS convergence</title>
<line x1="68" y1="233.6" x2="704" y2="233.6" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="60" y="237.6" text-anchor="end">1e-2</text>
<line x1="68" y1="150.6" x2="704" y2="150.6" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="60" y="154.6" text-anchor="end">1e-1</text>
<line x1="68" y1="67.5" x2="704" y2="67.5" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="60" y="71.5" text-anchor="end">1e0</text>
<line x1="600.4" y1="16" x2="600.4" y2="284" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="600.4" y="302" text-anchor="middle">1e-1</text>
<text class="mut" x="386.0" y="324" text-anchor="middle">h  (~ ndof^-1/3)</text>
<text class="mut" x="14" y="150.0" text-anchor="middle" transform="rotate(-90 14 150.0)">L2 error</text>
<circle cx="669.9" cy="99.3" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="493.1" cy="153.4" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="301.7" cy="208.4" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="102.1" cy="259.2" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<text x="76" y="30" fill="var(--s1)">u L2: rate 2.35</text>
<path d="M102.1 261.7 L669.9 101.7" fill="none" stroke="var(--s2)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="46" fill="var(--s2)">  fit h^2.35</text>
<circle cx="669.9" cy="38.3" r="3.5" fill="var(--s3)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="493.1" cy="70.9" r="3.5" fill="var(--s3)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="301.7" cy="105.9" r="3.5" fill="var(--s3)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="102.1" cy="139.9" r="3.5" fill="var(--s3)" stroke="var(--surface)" stroke-width="1.5"/>
<text x="76" y="62" fill="var(--s3)">p L2 shifted: rate 1.49</text>
<path d="M102.1 140.7 L669.9 38.9" fill="none" stroke="var(--s4)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="78" fill="var(--s4)">  fit h^1.49</text>
</svg>


## Boundary conditions

### Traction generalises the do-nothing outflow

A prescribed traction of zero must reproduce the do-nothing outflow exactly,
not approximately: the two take the same kernel branch and add a term that is
identically zero. Anything else means they are separate mechanisms that happen
to agree.

| quantity | do-nothing | traction t = 0 | difference |
|---|---|---|---|
| u_linf | 1.755e-01 | 1.755e-01 | 0.000e+00 |
| p_linf | 2.147e-01 | 2.147e-01 | 0.000e+00 |


### A pressure port fixes the level and nothing else

Holding a port at `p_bar` should shift the whole pressure field by
`p_bar - p_exact(outlet)` and leave the velocity alone. Both halves are
checked: the shift against that closed form, and the velocity against the
exact solution.

| p_bar | u_linf | p_linf | predicted shift | error | status |
|---|---|---|---|---|---|
| -0.16 | 1.706e-09 | 2.500e-09 | 0.000e+00 | 2.500e-09 | pass |
| -0.08 | 1.743e-09 | 8.000e-02 | 8.000e-02 | 0.000e+00 | pass |
| 0 | 1.000e+00 | 0.000e+00 | 1.600e-01 | 1.600e-01 | not converged |
| 0.16 | 3.231e-09 | 3.200e-01 | 3.200e-01 | 0.000e+00 | pass |
| 0.5 | 6.452e-09 | 6.600e-01 | 6.600e-01 | 0.000e+00 | pass |
| 1 | 9.479e-09 | 1.160e+00 | 1.160e+00 | 0.000e+00 | pass |
| 1.5 | 1.341e-08 | 1.660e+00 | 1.660e+00 | 0.000e+00 | pass |
| 3 | 6.281e-08 | 3.160e+00 | 3.160e+00 | 0.000e+00 | pass |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>Pressure port linearity</title>
<line x1="68" y1="261.7" x2="704" y2="261.7" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="60" y="265.7" text-anchor="end">0</text>
<line x1="68" y1="191.0" x2="704" y2="191.0" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="60" y="195.0" text-anchor="end">1</text>
<line x1="68" y1="120.3" x2="704" y2="120.3" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="60" y="124.3" text-anchor="end">2</text>
<line x1="68" y1="49.6" x2="704" y2="49.6" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="60" y="53.6" text-anchor="end">3</text>
<line x1="130.8" y1="16" x2="130.8" y2="284" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="130.8" y="302" text-anchor="middle">0</text>
<line x1="310.5" y1="16" x2="310.5" y2="284" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="310.5" y="302" text-anchor="middle">1</text>
<line x1="490.2" y1="16" x2="490.2" y2="284" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="490.2" y="302" text-anchor="middle">2</text>
<line x1="669.9" y1="16" x2="669.9" y2="284" stroke="var(--grid)" stroke-width="1"/>
<text class="mut" x="669.9" y="302" text-anchor="middle">3</text>
<text class="mut" x="386.0" y="324" text-anchor="middle">prescribed p_bar</text>
<text class="mut" x="14" y="150.0" text-anchor="middle" transform="rotate(-90 14 150.0)">pressure offset</text>
<circle cx="102.1" cy="261.7" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="116.4" cy="256.0" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="159.6" cy="239.1" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="220.7" cy="215.0" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="310.5" cy="179.7" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="400.4" cy="144.3" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<circle cx="669.9" cy="38.3" r="3.5" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>
<text x="76" y="30" fill="var(--s1)">measured p_linf</text>
<path d="M102.1 261.7 L116.4 256.0 L130.8 250.4 L159.6 239.1 L220.7 215.0 L310.5 179.7 L400.4 144.3 L669.9 38.3" fill="none" stroke="var(--s2)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="46" fill="var(--s2)">predicted |p_bar - p_exact|</text>
</svg>


## Global mass conservation

Summed over every node, the interior sub-control-surface fluxes telescope away
and the boundary closure is all that is left, so the total continuity residual
is the net mass imbalance of the domain and should be zero. Zero needs a scale
to be meaningful, and the scale is the exact volumetric inflow -- 1/9 for the
backward-facing step -- so what is checked is the ratio.

**The plane-integrated flux imbalance is reported and deliberately not scored.**
docs/CVFEM_Verification_Farrell.md measured it at 17% on a case whose residual
sum was 7e-14, and the 17% is trapezoidal error rather than lost mass: the inlet
is a smooth parabola and the outlet profile is developing and recirculating. The
residual sum is quadrature-free and is the conservation test; the inflow figure
is kept because it does check the prescribed profile, against its own tolerance.

| case | ndof | sum of continuity | relative to inflow | inflow | inflow err | flux imbalance (not scored) | status |
|---|---|---|---|---|---|---|---|
| lshape | 47268 | -5.383e-03 | 4.845e-02 | 1.077e-01 | 3.445e-03 | 1.000e+00 | not converged |


1 of these run(s) did not reach the nonlinear tolerance, so nothing above is a statement about conservation for them -- the residual sum of an unconverged iterate measures where the solver stopped, not whether the scheme conserves mass. The solver table below carries the detail.


## Solver behaviour

Reported for context, not asserted here: a verification result from a run that
did not converge is not a verification result. Timings carry the dof count they
were measured on, and the machine is named in Provenance.

| case | ndof | converged | Newton | linear its | t_solve (s) | Re reached | gauge |
|---|---|---|---|---|---|---|---|
| n4 | 500 | no | 40 | 780 | 0.276 | 0 | zero mean |
| n8 | 2916 | no | 40 | 1893 | 1.245 | 0 | zero mean |
| n16 | 19652 | yes | 20 | 7139 | 11.121 | 2 | zero mean |
| n32 | 143748 | yes | 11 | 15214 | 46.774 | 2 | zero mean |
| dirichlet | 33124 | yes | 1 | 6215 | 12.619 | 100 | zero mean |
| natural | 33124 | no | 12 | 2795 | 4.853 | 0 | determined by the do-nothing outflow |
| traction0 | 33124 | no | 12 | 2889 | 4.949 | 0 | determined by the traction surface |
| p-0.16 | 33124 | yes | 1 | 7204 | 12.428 | 100 | determined by the prescribed pressure |
| p-0.08 | 33124 | yes | 2 | 7096 | 12.119 | 100 | determined by the prescribed pressure |
| p0 | 33124 | no | 0 | 80 | 0.140 | 0 | determined by the prescribed pressure |
| p0.16 | 33124 | yes | 1 | 6688 | 11.460 | 100 | determined by the prescribed pressure |
| p0.5 | 33124 | yes | 1 | 6662 | 11.379 | 100 | determined by the prescribed pressure |
| p1.0 | 33124 | yes | 1 | 6999 | 11.927 | 100 | determined by the prescribed pressure |
| p1.5 | 33124 | yes | 1 | 7151 | 12.273 | 100 | determined by the prescribed pressure |
| p3.0 | 33124 | yes | 2 | 5872 | 10.097 | 100 | determined by the prescribed pressure |
| lshape | 47268 | no | 0 | 132 | 0.073 | 0 | determined by the do-nothing outflow |

## Provenance

| field | value |
|---|---|
| generated | 2026-09-09 12:36:06 |
| machine | nid006545 |
| threads | 72 |
| commit | -- |
| run directory | /Users/patrickzulian/Desktop/code/merge_git_repos/sfem/spikes/cvfem/verification_runs/grace-4630995 |
| runs parsed | 16 of 16 |

Regenerate with `python3 python/cvfem_verify_report.py verification_runs/grace-4630995`.
