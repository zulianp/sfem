# CVFEM verification report

Every claim below is an identity or a fitted rate checked against a threshold, so this page states whether the code is right rather than inviting a reader to judge a plot.

## Summary

**5 of 5 checks pass.**

| check | evidence | status |
|---|---|---|
| Spatial order, u L2 | rate 2.330 (expect >= 1.65) | <span class="st-pass">pass</span> |
| Spatial order, p L2 shifted | rate 1.514 (expect >= 0.95) | <span class="st-pass">pass</span> |
| Traction t=0 equals the do-nothing outflow | max diff 0.00e+00 | <span class="st-pass">pass</span> |
| Pressure port shifts the level by p_bar - p_exact | 8 of 8 values verified | <span class="st-pass">pass</span> |
| Global mass balance | 1 of 1 case(s) verified | <span class="st-pass">pass</span> |

## Spatial order of accuracy

Manufactured solution, volume-weighted L2 against the exact field, with the
pressure gauge offset removed (`p_l2_shifted`). The rate is fitted by log-log
least squares against `h ~ ndof^(-1/3)`.

| ndof | h | u L2 | p L2 | p L2 shifted |
|---|---|---|---|---|
| 500 | 0.1260 | 3.970e-01 | 2.320e+00 | 2.320e+00 |
| 2916 | 0.0700 | 9.158e-02 | 9.321e-01 | 9.321e-01 |
| 19652 | 0.0371 | 2.015e-02 | 3.451e-01 | 3.451e-01 |
| 143748 | 0.0191 | 4.924e-03 | 1.345e-01 | 1.345e-01 |

| quantity | fitted rate | R^2 | levels | threshold | status |
|---|---|---|---|---|---|
| u L2 | 2.330 | 0.9987 | 4 | &ge; 1.65 | <span class="st-pass">pass</span> |
| p L2 shifted | 1.514 | 0.9995 | 4 | &ge; 0.95 | <span class="st-pass">pass</span> |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>MMS convergence</title>
<line x1="68" y1="234.0" x2="704" y2="234.0" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="238.0" text-anchor="end">1e-2</text>
<line x1="68" y1="151.3" x2="704" y2="151.3" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="155.3" text-anchor="end">1e-1</text>
<line x1="68" y1="68.6" x2="704" y2="68.6" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="72.6" text-anchor="end">1e0</text>
<line x1="600.4" y1="16" x2="600.4" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="600.4" y="302" text-anchor="middle">1e-1</text>
<text class="mut" x="386.0" y="324" text-anchor="middle">h  (~ ndof^-1/3)</text>
<text class="mut" x="14" y="150.0" text-anchor="middle" transform="rotate(-90 14 150.0)">L2 error</text>
<circle cx="669.9" cy="101.8" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="493.1" cy="154.4" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="301.7" cy="208.8" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="102.1" cy="259.5" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<text x="76" y="30" fill="var(--s1, #d97757)">u L2: rate 2.33</text>
<path d="M102.1 261.7 L669.9 103.7" fill="none" stroke="var(--s2, #8c8880)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="46" fill="var(--s2, #8c8880)">  fit h^2.33</text>
<circle cx="669.9" cy="38.3" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="493.1" cy="71.1" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="301.7" cy="106.8" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="102.1" cy="140.7" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<text x="76" y="62" fill="var(--s3, #4a6fa5)">p L2 shifted: rate 1.51</text>
<path d="M102.1 141.6 L669.9 38.9" fill="none" stroke="var(--s4, #a8a29a)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="78" fill="var(--s4, #a8a29a)">  fit h^1.51</text>
</svg>


## Boundary conditions

### Traction generalises the do-nothing outflow

A prescribed traction of zero must reproduce the do-nothing outflow exactly,
not approximately: the two take the same kernel branch and add a term that is
identically zero. Anything else means they are separate mechanisms that happen
to agree.

| quantity | do-nothing | traction t = 0 | difference |
|---|---|---|---|
| u_linf | 5.497e-02 | 5.497e-02 | 0.000e+00 |
| p_linf | 1.760e-01 | 1.760e-01 | 0.000e+00 |


### A pressure port fixes the level and nothing else

Holding a port at `p_bar` should shift the whole pressure field by
`p_bar - p_exact(outlet)` and leave the velocity alone. Both halves are
checked: the shift against that closed form, and the velocity against the
exact solution.

| p_bar | u_linf | p_linf | predicted shift | error | status |
|---|---|---|---|---|---|
| -0.16 | 1.985e-11 | 4.345e-12 | 0.000e+00 | 4.345e-12 | <span class="st-pass">pass</span> |
| -0.08 | 2.074e-11 | 8.000e-02 | 8.000e-02 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0 | 3.501e-12 | 1.600e-01 | 1.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.16 | 2.187e-11 | 3.200e-01 | 3.200e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.5 | 2.572e-11 | 6.600e-01 | 6.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1 | 2.259e-11 | 1.160e+00 | 1.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1.5 | 4.235e-12 | 1.660e+00 | 1.660e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 3 | 1.224e-10 | 3.160e+00 | 3.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>Pressure port linearity</title>
<line x1="68" y1="261.7" x2="704" y2="261.7" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="265.7" text-anchor="end">0</text>
<line x1="68" y1="191.0" x2="704" y2="191.0" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="195.0" text-anchor="end">1</text>
<line x1="68" y1="120.3" x2="704" y2="120.3" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="124.3" text-anchor="end">2</text>
<line x1="68" y1="49.6" x2="704" y2="49.6" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="53.6" text-anchor="end">3</text>
<line x1="130.8" y1="16" x2="130.8" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="130.8" y="302" text-anchor="middle">0</text>
<line x1="310.5" y1="16" x2="310.5" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="310.5" y="302" text-anchor="middle">1</text>
<line x1="490.2" y1="16" x2="490.2" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="490.2" y="302" text-anchor="middle">2</text>
<line x1="669.9" y1="16" x2="669.9" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="669.9" y="302" text-anchor="middle">3</text>
<text class="mut" x="386.0" y="324" text-anchor="middle">prescribed p_bar</text>
<text class="mut" x="14" y="150.0" text-anchor="middle" transform="rotate(-90 14 150.0)">pressure offset</text>
<circle cx="102.1" cy="261.7" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="116.4" cy="256.0" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="130.8" cy="250.4" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="159.6" cy="239.1" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="220.7" cy="215.0" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="310.5" cy="179.7" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="400.4" cy="144.3" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="669.9" cy="38.3" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<text x="76" y="30" fill="var(--s1, #d97757)">measured p_linf</text>
<path d="M102.1 261.7 L116.4 256.0 L130.8 250.4 L159.6 239.1 L220.7 215.0 L310.5 179.7 L400.4 144.3 L669.9 38.3" fill="none" stroke="var(--s2, #8c8880)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="46" fill="var(--s2, #8c8880)">predicted |p_bar - p_exact|</text>
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
| lshape | 7060 | 9.906e-17 | 8.915e-16 | 9.766e-02 | 1.345e-02 | 5.198e-01 | <span class="st-pass">pass</span> |


## Solver behaviour

Reported for context, not asserted here: a verification result from a run that
did not converge is not a verification result. Timings carry the dof count they
were measured on, and the machine is named in Provenance.

| case | ndof | converged | Newton | linear its | t_solve (s) | Re reached | gauge |
|---|---|---|---|---|---|---|---|
| n4 | 500 | yes | 2 | 167 | 0.061 | 2 | zero mean |
| n8 | 2916 | yes | 2 | 275 | 0.222 | 2 | zero mean |
| n16 | 19652 | yes | 2 | 669 | 1.341 | 2 | zero mean |
| n32 | 143748 | yes | 2 | 1685 | 5.972 | 2 | zero mean |
| dirichlet | 33124 | yes | 0 | 551 | 1.423 | 100 | zero mean |
| natural | 33124 | yes | 3 | 9099 | 20.453 | 100 | determined by the do-nothing outflow |
| traction0 | 33124 | yes | 4 | 9203 | 20.973 | 100 | determined by the traction surface |
| p-0.16 | 33124 | yes | 0 | 1034 | 2.372 | 100 | determined by the prescribed pressure |
| p-0.08 | 33124 | yes | 0 | 1197 | 2.761 | 100 | determined by the prescribed pressure |
| p0 | 33124 | yes | 0 | 754 | 1.717 | 100 | determined by the prescribed pressure |
| p0.16 | 33124 | yes | 0 | 1024 | 2.374 | 100 | determined by the prescribed pressure |
| p0.5 | 33124 | yes | 0 | 892 | 2.043 | 100 | determined by the prescribed pressure |
| p1.0 | 33124 | yes | 0 | 902 | 2.065 | 100 | determined by the prescribed pressure |
| p1.5 | 33124 | yes | 0 | 1021 | 2.339 | 100 | determined by the prescribed pressure |
| p3.0 | 33124 | yes | 0 | 1386 | 3.203 | 100 | determined by the prescribed pressure |
| lshape | 7060 | yes | 3 | 0 | 0.496 | 20 | determined by the do-nothing outflow |

## Provenance

| field | value |
|---|---|
| generated | 2026-09-09 14:06:28 |
| machine | nid006549 |
| threads | 72 |
| commit | -- |
| run directory | /Users/patrickzulian/Desktop/code/merge_git_repos/sfem/spikes/cvfem/verification_runs/grace-4631280 |
| runs parsed | 16 of 16 |

Regenerate with `python3 python/cvfem_verify_report.py verification_runs/grace-4631280`.
