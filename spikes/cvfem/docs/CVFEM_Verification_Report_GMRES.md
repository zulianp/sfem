# CVFEM verification report

Every claim below is an identity or a fitted rate checked against a threshold, so this page states whether the code is right rather than inviting a reader to judge a plot.

## Summary

**6 of 6 checks pass.**

| check | evidence | status |
|---|---|---|
| Spatial order, u L2 | rate 2.330 (expect >= 1.65) | <span class="st-pass">pass</span> |
| Spatial order, p L2 shifted | rate 1.514 (expect >= 0.95) | <span class="st-pass">pass</span> |
| Traction t=0 equals the do-nothing outflow | max diff 0.00e+00 | <span class="st-pass">pass</span> |
| Pressure port shifts the level by p_bar - p_exact | 8 of 8 values verified | <span class="st-pass">pass</span> |
| Pump: the port carries what the diaphragm sweeps | 6 of 6 instant(s) verified | <span class="st-pass">pass</span> |
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
| -0.16 | 2.780e-12 | 3.564e-13 | 0.000e+00 | 3.564e-13 | <span class="st-pass">pass</span> |
| -0.08 | 6.256e-12 | 8.000e-02 | 8.000e-02 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0 | 2.558e-10 | 1.600e-01 | 1.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.16 | 4.818e-12 | 3.200e-01 | 3.200e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.5 | 7.778e-11 | 6.600e-01 | 6.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1 | 8.129e-12 | 1.160e+00 | 1.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1.5 | 1.808e-10 | 1.660e+00 | 1.660e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 3 | 1.992e-11 | 3.160e+00 | 3.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |

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


## Diaphragm pump

A closed chamber with a diaphragm that moves and one port that lets fluid in or
out. The diaphragm is transpiration on a fixed mesh: the wall does not move, its
normal velocity is prescribed through it. That is exact for the mass it carries
and silent about the geometric nonlinearity a real diaphragm has, which an ALE
formulation would capture and this deliberately does not.

It is checked on an identity, not against a solution, because it has none. The
chamber is fixed and the flow incompressible, so the flux through its closed
boundary is zero; the walls carry none and the diaphragm's velocity is
prescribed. So the port must carry exactly what the diaphragm sweeps,
`rho V Lx Lz`, and if transpiration is moving the wrong mass the identity fails
by exactly that much. Both fluxes are integrated on the operator's own boundary
sub-control surfaces, so this measures the discretisation rather than a second
quadrature's opinion of it.

There is no valve, so the pump does not rectify: over a full cycle it moves
fluid back and forth and nets nothing. That is the scope, not an oversight.

| run | t | v diaphragm | swept | port flux | abs(port - swept) | abs(port + diaphragm) | status |
|---|---|---|---|---|---|---|---|
| steady | -- | -- | +1.000000000 | +1.000000000 | 1.832e-14 | 1.832e-14 | <span class="st-pass">pass</span> |
| t1 | 0.125 | +0.7071 | +0.707106781 | +0.707106781 | 2.265e-14 | 2.265e-14 | <span class="st-pass">pass</span> |
| t2 | 0.250 | +1.0000 | +1.000000000 | +1.000000000 | 2.676e-14 | 2.676e-14 | <span class="st-pass">pass</span> |
| t4 | 0.500 | +0.0000 | +0.000000000 | +0.000000000 | 6.619e-15 | 6.619e-15 | <span class="st-pass">pass</span> |
| t6 | 0.750 | -1.0000 | -1.000000000 | -1.000000000 | 1.932e-14 | 1.932e-14 | <span class="st-pass">pass</span> |
| t8 | 1.000 | -0.0000 | -0.000000000 | +0.000000000 | 8.328e-15 | 8.328e-15 | <span class="st-pass">pass</span> |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>Pump: swept volume against port flux</title>
<line x1="68" y1="261.7" x2="704" y2="261.7" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="265.7" text-anchor="end">-1</text>
<line x1="68" y1="205.8" x2="704" y2="205.8" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="209.8" text-anchor="end">-0.5</text>
<line x1="68" y1="150.0" x2="704" y2="150.0" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="154.0" text-anchor="end">0</text>
<line x1="68" y1="94.2" x2="704" y2="94.2" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="98.2" text-anchor="end">0.5</text>
<line x1="68" y1="38.3" x2="704" y2="38.3" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="42.3" text-anchor="end">1</text>
<line x1="102.1" y1="16" x2="102.1" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="102.1" y="302" text-anchor="middle">-1</text>
<line x1="244.0" y1="16" x2="244.0" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="244.0" y="302" text-anchor="middle">-0.5</text>
<line x1="386.0" y1="16" x2="386.0" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="386.0" y="302" text-anchor="middle">0</text>
<line x1="528.0" y1="16" x2="528.0" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="528.0" y="302" text-anchor="middle">0.5</text>
<line x1="669.9" y1="16" x2="669.9" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="669.9" y="302" text-anchor="middle">1</text>
<text class="mut" x="386.0" y="324" text-anchor="middle">prescribed diaphragm velocity</text>
<text class="mut" x="14" y="150.0" text-anchor="middle" transform="rotate(-90 14 150.0)">flux through the port</text>
<circle cx="586.8" cy="71.0" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="669.9" cy="38.3" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="386.0" cy="150.0" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="102.1" cy="261.7" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="386.0" cy="150.0" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<text x="76" y="30" fill="var(--s1, #d97757)">port flux</text>
<path d="M102.1 261.7 L386.0 150.0 L386.0 150.0 L586.8 71.0 L669.9 38.3" fill="none" stroke="var(--s2, #8c8880)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="46" fill="var(--s2, #8c8880)">rho V Lx Lz</text>
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
| lshape | 7060 | 6.342e-15 | 5.707e-14 | 9.766e-02 | 1.345e-02 | 4.994e-01 | <span class="st-pass">pass</span> |


## Solver behaviour

Reported for context, not asserted here: a verification result from a run that
did not converge is not a verification result. Timings carry the dof count they
were measured on, and the machine is named in Provenance.

| case | ndof | converged | Newton | linear its | t_solve (s) | Re reached | gauge |
|---|---|---|---|---|---|---|---|
| n4 | 500 | yes | 2 | 168 | 0.067 | 2 | zero mean |
| n8 | 2916 | yes | 2 | 276 | 0.167 | 2 | zero mean |
| n16 | 19652 | yes | 2 | 688 | 1.027 | 2 | zero mean |
| n32 | 143748 | yes | 2 | 1771 | 4.374 | 2 | zero mean |
| dirichlet | 33124 | yes | 0 | 554 | 0.937 | 100 | zero mean |
| natural | 33124 | yes | 4 | 9391 | 12.750 | 100 | determined by the do-nothing outflow |
| traction0 | 33124 | yes | 4 | 9391 | 12.833 | 100 | determined by the traction surface |
| p-0.16 | 33124 | yes | 0 | 881 | 1.220 | 100 | determined by the prescribed pressure |
| p-0.08 | 33124 | yes | 0 | 863 | 1.202 | 100 | determined by the prescribed pressure |
| p0 | 33124 | yes | 0 | 750 | 1.040 | 100 | determined by the prescribed pressure |
| p0.16 | 33124 | yes | 0 | 914 | 1.281 | 100 | determined by the prescribed pressure |
| p0.5 | 33124 | yes | 0 | 907 | 1.234 | 100 | determined by the prescribed pressure |
| p1.0 | 33124 | yes | 0 | 699 | 0.965 | 100 | determined by the prescribed pressure |
| p1.5 | 33124 | yes | 0 | 3157 | 4.358 | 100 | determined by the prescribed pressure |
| p3.0 | 33124 | yes | 0 | 1358 | 1.840 | 100 | determined by the prescribed pressure |
| lshape | 7060 | yes | 3 | 2685 | 2.064 | 20 | determined by the do-nothing outflow |
| steady | 2916 | yes | 3 | 1329 | 0.418 | 20 | determined by the prescribed pressure |
| t1 | 2916 | yes | 3 | 1155 | 0.347 | 20 | determined by the prescribed pressure |
| t2 | 2916 | yes | 3 | 2365 | 0.749 | 20 | determined by the prescribed pressure |
| t4 | 2916 | yes | 3 | 4293 | 1.308 | 20 | determined by the prescribed pressure |
| t6 | 2916 | yes | 3 | 7340 | 2.292 | 20 | determined by the prescribed pressure |
| t8 | 2916 | yes | 3 | 10217 | 3.303 | 20 | determined by the prescribed pressure |

## Provenance

| field | value |
|---|---|
| generated | 2026-09-11 10:32:23 |
| machine | nid006558 |
| threads | 72 |
| commit | -- |
| linear solver | fgmres, restart 480 |
| element refine level | 1 |
| run directory | /Users/patrickzulian/Desktop/code/merge_git_repos/sfem/spikes/cvfem/verification_runs/grace-fgmres-det-4644413 |
| runs parsed | 22 of 22 |

Regenerate with `python3 python/cvfem_verify_report.py verification_runs/grace-fgmres-det-4644413`.
