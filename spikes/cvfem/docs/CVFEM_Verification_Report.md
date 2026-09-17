# CVFEM verification report

Every claim below is an identity or a fitted rate checked against a threshold, so this page states whether the code is right rather than inviting a reader to judge a plot.

## Summary

**13 of 13 checks pass.**

| check | evidence | status |
|---|---|---|
| Spatial order, u L2 (Re 1) | rate 2.308 (expect >= 1.65) | <span class="st-pass">pass</span> |
| Spatial order, p L2 shifted (Re 1) | rate 1.458 (expect >= 0.95) | <span class="st-pass">pass</span> |
| Spatial order, u L2 (Re 100) | rate 0.727 (expect >= 0.50) | <span class="st-pass">pass</span> |
| Spatial order, p L2 shifted (Re 100) | rate 1.127 (expect >= 0.50) | <span class="st-pass">pass</span> |
| Traction t=0 equals the do-nothing outflow | max diff 0.00e+00 | <span class="st-pass">pass</span> |
| Pressure port shifts the level by p_bar - p_exact (direct) | 8 of 8 values verified | <span class="st-pass">pass</span> |
| Pressure port shifts the level by p_bar - p_exact (mg) | 8 of 8 values verified | <span class="st-pass">pass</span> |
| Pressure port shifts the level by p_bar - p_exact (vanka) | 8 of 8 values verified | <span class="st-pass">pass</span> |
| Budget closes and eps_num vanishes at second order (mg) | rate 2.022, closure 2.84e-03 at the finest | <span class="st-pass">pass</span> |
| Budget is the same on both solver stacks | 0.000e+00 relative to eps_visc at 75140 dof | <span class="st-pass">pass</span> |
| Pump: the port carries what the diaphragm sweeps | 6 of 6 instant(s) verified | <span class="st-pass">pass</span> |
| Global mass balance | 2 of 2 case(s) verified | <span class="st-pass">pass</span> |
| Every run converged | 47 of 47 converged | <span class="st-pass">pass</span> |

## Spatial order of accuracy

Manufactured solution, volume-weighted L2 against the exact field, with the
pressure gauge offset removed (`p_l2_shifted`). The rate is fitted by log-log
least squares against `h ~ ndof^(-1/3)`.


### Re = 1 -- upwinding inactive, a consistency statement

| ndof | h | u L2 | local rate | p L2 | p L2 shifted |
|---|---|---|---|---|---|
| 500 | 0.1260 | 3.979e-01 | -- | 2.676e+00 | 2.676e+00 |
| 2916 | 0.0700 | 9.090e-02 | 2.51 | 1.159e+00 | 1.159e+00 |
| 19652 | 0.0371 | 2.106e-02 | 2.30 | 4.598e-01 | 4.598e-01 |
| 143748 | 0.0191 | 5.071e-03 | 2.15 | 1.708e-01 | 1.708e-01 |

| quantity | fitted rate | R^2 | levels | threshold | status |
|---|---|---|---|---|---|
| u L2 | 2.308 | 0.9988 | 4 | &ge; 1.65 | <span class="st-pass">pass</span> |
| p L2 shifted | 1.458 | 0.9999 | 4 | &ge; 0.95 | <span class="st-pass">pass</span> |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>MMS convergence (Re 1)</title>
<line x1="68" y1="235.8" x2="704" y2="235.8" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="239.8" text-anchor="end">1e-2</text>
<line x1="68" y1="154.6" x2="704" y2="154.6" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="158.6" text-anchor="end">1e-1</text>
<line x1="68" y1="73.4" x2="704" y2="73.4" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="77.4" text-anchor="end">1e0</text>
<line x1="600.4" y1="16" x2="600.4" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="600.4" y="302" text-anchor="middle">1e-1</text>
<text class="mut" x="386.0" y="324" text-anchor="middle">h  (~ ndof^-1/3)</text>
<text class="mut" x="14" y="150.0" text-anchor="middle" transform="rotate(-90 14 150.0)">L2 error</text>
<circle cx="669.9" cy="105.9" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="493.1" cy="158.0" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="301.7" cy="209.6" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="102.1" cy="259.8" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<text x="76" y="30" fill="var(--s1, #d97757)">u L2: rate 2.31</text>
<path d="M102.1 261.7 L669.9 108.0" fill="none" stroke="var(--s2, #8c8880)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="46" fill="var(--s2, #8c8880)">  fit h^2.31</text>
<circle cx="669.9" cy="38.7" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="493.1" cy="68.2" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="301.7" cy="100.8" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="102.1" cy="135.8" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<text x="76" y="62" fill="var(--s3, #4a6fa5)">p L2 shifted: rate 1.46</text>
<path d="M102.1 135.4 L669.9 38.3" fill="none" stroke="var(--s4, #a8a29a)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="78" fill="var(--s4, #a8a29a)">  fit h^1.46</text>
</svg>


### Re = 100 -- upwinding active, the accuracy statement

The scheme is first-order donor-cell upwind with no limiter, so a
rate near 1 here is the expected result, not a defect -- and it is
the measurement that decides whether a higher-order convection
scheme is worth its cost.

Read the local rates, not just the fit. They do not settle, so the
fitted value is a bound on the order rather than a measurement of
it: what is established is that the ladder converges and that the
order is nowhere near the 2.3 the Re = 1 arm reports. The gate is
set well below the fitted rate for that reason.

| ndof | h | u L2 | local rate | p L2 | p L2 shifted |
|---|---|---|---|---|---|
| 500 | 0.1260 | 3.355e-01 | -- | 2.462e-01 | 2.458e-01 |
| 2916 | 0.0700 | 1.580e-01 | 1.28 | 8.757e-02 | 8.694e-02 |
| 19652 | 0.0371 | 1.191e-01 | 0.44 | 4.578e-02 | 4.546e-02 |
| 143748 | 0.0191 | 7.950e-02 | 0.61 | 2.835e-02 | 2.825e-02 |

| quantity | fitted rate | R^2 | levels | threshold | status |
|---|---|---|---|---|---|
| u L2 | 0.727 | 0.9465 | 4 | &ge; 0.50 | <span class="st-pass">pass</span> |
| p L2 shifted | 1.127 | 0.9588 | 4 | &ge; 0.50 | <span class="st-pass">pass</span> |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>MMS convergence (Re 100)</title>
<line x1="68" y1="141.6" x2="704" y2="141.6" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="60" y="145.6" text-anchor="end">1e-1</text>
<line x1="600.4" y1="16" x2="600.4" y2="284" stroke="var(--grid, #e4e1d8)" stroke-width="1"/>
<text class="mut" x="600.4" y="302" text-anchor="middle">1e-1</text>
<text class="mut" x="386.0" y="324" text-anchor="middle">h  (~ ndof^-1/3)</text>
<text class="mut" x="14" y="150.0" text-anchor="middle" transform="rotate(-90 14 150.0)">L2 error</text>
<circle cx="669.9" cy="38.3" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="493.1" cy="102.5" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="301.7" cy="126.7" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="102.1" cy="161.1" r="3.5" fill="var(--s1, #d97757)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<text x="76" y="30" fill="var(--s1, #d97757)">u L2: rate 0.73</text>
<path d="M102.1 166.8 L669.9 49.9" fill="none" stroke="var(--s2, #8c8880)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="46" fill="var(--s2, #8c8880)">  fit h^0.73</text>
<circle cx="669.9" cy="64.9" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="493.1" cy="153.5" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="301.7" cy="208.8" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<circle cx="102.1" cy="249.4" r="3.5" fill="var(--s3, #4a6fa5)" stroke="var(--surface, #ffffff)" stroke-width="1.5"/>
<text x="76" y="62" fill="var(--s3, #4a6fa5)">p L2 shifted: rate 1.13</text>
<path d="M102.1 261.7 L669.9 80.2" fill="none" stroke="var(--s4, #a8a29a)" stroke-width="2" stroke-dasharray="6 4"/>
<text x="76" y="78" fill="var(--s4, #a8a29a)">  fit h^1.13</text>
</svg>


## Boundary conditions

### Traction generalises the do-nothing outflow

A prescribed traction of zero must reproduce the do-nothing outflow exactly,
not approximately: the two take the same kernel branch and add a term that is
identically zero. Anything else means they are separate mechanisms that happen
to agree.

| quantity | do-nothing | traction t = 0 | difference |
|---|---|---|---|
| u_linf | 4.836e-02 | 4.836e-02 | 0.000e+00 |
| p_linf | 1.844e-01 | 1.844e-01 | 0.000e+00 |


### A pressure port fixes the level and nothing else -- flat mesh, exact linear solve

Holding a port at `p_bar` should shift the whole pressure field by
`p_bar - p_exact(outlet)` and leave the velocity alone. Both halves are
checked: the shift against that closed form, and the velocity against the
exact solution.

Run on both solver stacks the spike uses. A group that exercises one of
them certifies one of them, and this group spent a long time on a third
-- block-Jacobi on a flat mesh, inherited from unset defaults -- which
fails even at `p_bar = p_exact(outlet)`, where the boundary condition
asks for nothing unusual.

| p_bar | u_linf | p_linf | predicted shift | error | status |
|---|---|---|---|---|---|
| -0.16 | 5.572e-13 | 2.039e-14 | 0.000e+00 | 2.039e-14 | <span class="st-pass">pass</span> |
| -0.08 | 5.571e-13 | 8.000e-02 | 8.000e-02 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0 | 5.580e-13 | 1.600e-01 | 1.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.16 | 5.574e-13 | 3.200e-01 | 3.200e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.5 | 5.583e-13 | 6.600e-01 | 6.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1 | 5.587e-13 | 1.160e+00 | 1.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1.5 | 5.610e-13 | 1.660e+00 | 1.660e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 3 | 5.607e-13 | 3.160e+00 | 3.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>Pressure port linearity (direct)</title>
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


### A pressure port fixes the level and nothing else -- FGMRES + geometric multigrid

Holding a port at `p_bar` should shift the whole pressure field by
`p_bar - p_exact(outlet)` and leave the velocity alone. Both halves are
checked: the shift against that closed form, and the velocity against the
exact solution.

Run on both solver stacks the spike uses. A group that exercises one of
them certifies one of them, and this group spent a long time on a third
-- block-Jacobi on a flat mesh, inherited from unset defaults -- which
fails even at `p_bar = p_exact(outlet)`, where the boundary condition
asks for nothing unusual.

| p_bar | u_linf | p_linf | predicted shift | error | status |
|---|---|---|---|---|---|
| -0.16 | 1.904e-07 | 3.831e-08 | 0.000e+00 | 3.831e-08 | <span class="st-pass">pass</span> |
| -0.08 | 1.904e-07 | 8.000e-02 | 8.000e-02 | 4.000e-08 | <span class="st-pass">pass</span> |
| 0 | 1.904e-07 | 1.600e-01 | 1.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.16 | 1.904e-07 | 3.200e-01 | 3.200e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.5 | 1.904e-07 | 6.600e-01 | 6.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1 | 1.882e-07 | 1.160e+00 | 1.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1.5 | 1.882e-07 | 1.660e+00 | 1.660e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 3 | 1.882e-07 | 3.160e+00 | 3.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>Pressure port linearity (mg)</title>
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


### A pressure port fixes the level and nothing else -- FGMRES + Vanka, no multigrid

Holding a port at `p_bar` should shift the whole pressure field by
`p_bar - p_exact(outlet)` and leave the velocity alone. Both halves are
checked: the shift against that closed form, and the velocity against the
exact solution.

Run on both solver stacks the spike uses. A group that exercises one of
them certifies one of them, and this group spent a long time on a third
-- block-Jacobi on a flat mesh, inherited from unset defaults -- which
fails even at `p_bar = p_exact(outlet)`, where the boundary condition
asks for nothing unusual.

| p_bar | u_linf | p_linf | predicted shift | error | status |
|---|---|---|---|---|---|
| -0.16 | 1.904e-07 | 3.829e-08 | 0.000e+00 | 3.829e-08 | <span class="st-pass">pass</span> |
| -0.08 | 1.904e-07 | 8.000e-02 | 8.000e-02 | 4.000e-08 | <span class="st-pass">pass</span> |
| 0 | 1.904e-07 | 1.600e-01 | 1.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.16 | 1.904e-07 | 3.200e-01 | 3.200e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 0.5 | 1.904e-07 | 6.600e-01 | 6.600e-01 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1 | 1.882e-07 | 1.160e+00 | 1.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 1.5 | 1.882e-07 | 1.660e+00 | 1.660e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| 3 | 1.882e-07 | 3.160e+00 | 3.160e+00 | 0.000e+00 | <span class="st-pass">pass</span> |

<svg class="cvfig" viewBox="0 0 720 330" width="100%" role="img">
<title>Pressure port linearity (vanka)</title>
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


## The kinetic-energy budget

`dE/dt = P_in - P_out - eps_visc - eps_num`, with every term but the last
measured on the operator's own control volumes and sub-control surfaces, and
`eps_num` defined as the residual. On steady Poiseuille `dE/dt` is zero and the
flow is resolved, so `eps_num` is discretisation error: it must fall at the
scheme's order, and `closure` -- `|eps_num|` against the largest term it is
differenced from -- must fall with it.


### FGMRES + geometric multigrid

| ndof | E | eps_visc | eps_num | closure | div_l2 |
|---|---|---|---|---|---|
| 10692 | 1.066e+00 | 2.012e-01 | 8.750e-03 | 4.167e-02 | 1.231e-12 |
| 75140 | 1.067e+00 | 2.102e-01 | 2.344e-03 | 1.103e-02 | 4.457e-12 |
| 561924 | 1.067e+00 | 2.125e-01 | 6.055e-04 | 2.841e-03 | 4.501e-12 |


Observed order of `eps_num`: **2.022** (R^2 1.0000 over 3 levels, expect >= 1.65).


### FGMRES + Vanka, no multigrid

| ndof | E | eps_visc | eps_num | closure | div_l2 |
|---|---|---|---|---|---|
| 75140 | 1.067e+00 | 2.102e-01 | 2.344e-03 | 1.103e-02 | 3.377e-11 |


Solver independence at 75140 dof: `eps_num` 2.344e-03 (mg) against 2.344e-03 (vanka), difference 0.000e+00 of eps_visc.


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
| steady | -- | -- | +1.000000000 | +1.000000000 | 0.000e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| t1 | 0.125 | +0.7071 | +0.707106781 | +0.707106781 | 0.000e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| t2 | 0.250 | +1.0000 | +1.000000000 | +1.000000000 | 0.000e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| t4 | 0.500 | +0.0000 | +0.000000000 | +0.000000000 | 9.708e-18 | 9.708e-18 | <span class="st-pass">pass</span> |
| t6 | 0.750 | -1.0000 | -1.000000000 | -1.000000000 | 0.000e+00 | 0.000e+00 | <span class="st-pass">pass</span> |
| t8 | 1.000 | -0.0000 | -0.000000000 | -0.000000000 | 1.378e-17 | 1.378e-17 | <span class="st-pass">pass</span> |

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
| lshape | 7060 | 4.252e-18 | 3.827e-17 | 1.111e-01 | 1.345e-02 | 5.066e-01 | <span class="st-pass">pass</span> |
| re500 | 4540 | 1.077e-22 | 2.068e-17 | 4.226e-06 | 1.883e-01 | 1.944e-16 | <span class="st-pass">pass</span> |


## Solver behaviour

Convergence is asserted by the check above; the physics checks abstain on a run
that did not converge, because a verification result from such a run is not a
verification result. Timings carry the dof count they were measured on, and the
machine is named in Provenance.

| case | ndof | converged | Newton | linear its | t_solve (s) | Re reached | gauge |
|---|---|---|---|---|---|---|---|
| re1_n4 | 500 | yes | 3 | 138 | 0.042 | 2 | zero mean |
| re1_n8 | 2916 | yes | 2 | 203 | 0.100 | 2 | zero mean |
| re1_n16 | 19652 | yes | 2 | 453 | 0.507 | 2 | zero mean |
| re1_n32 | 143748 | yes | 2 | 1200 | 2.189 | 2 | zero mean |
| re100_n4 | 500 | yes | 5 | 918 | 0.314 | 200 | zero mean |
| re100_n8 | 2916 | yes | 5 | 1333 | 0.650 | 200 | zero mean |
| re100_n16 | 19652 | yes | 4 | 2223 | 2.449 | 200 | zero mean |
| re100_n32 | 143748 | yes | 4 | 4681 | 8.234 | 200 | zero mean |
| dirichlet | 4900 | yes | 0 | 3 | 0.035 | 100 | zero mean |
| natural | 33124 | yes | 3 | 8283 | 8.988 | 100 | determined by the do-nothing outflow |
| traction0 | 33124 | yes | 3 | 8283 | 8.788 | 100 | determined by the traction surface |
| direct_p-0.16 | 4900 | yes | 0 | 3 | 0.035 | 100 | determined by the prescribed pressure |
| direct_p-0.08 | 4900 | yes | 0 | 3 | 0.035 | 100 | determined by the prescribed pressure |
| direct_p0 | 4900 | yes | 0 | 3 | 0.035 | 100 | determined by the prescribed pressure |
| direct_p0.16 | 4900 | yes | 0 | 3 | 0.035 | 100 | determined by the prescribed pressure |
| direct_p0.5 | 4900 | yes | 0 | 3 | 0.035 | 100 | determined by the prescribed pressure |
| direct_p1.0 | 4900 | yes | 0 | 3 | 0.035 | 100 | determined by the prescribed pressure |
| direct_p1.5 | 4900 | yes | 0 | 3 | 0.035 | 100 | determined by the prescribed pressure |
| direct_p3.0 | 4900 | yes | 0 | 3 | 0.038 | 100 | determined by the prescribed pressure |
| mg_p-0.16 | 33124 | yes | 1 | 57 | 0.166 | 100 | determined by the prescribed pressure |
| mg_p-0.08 | 33124 | yes | 1 | 57 | 0.170 | 100 | determined by the prescribed pressure |
| mg_p0 | 33124 | yes | 1 | 58 | 0.176 | 100 | determined by the prescribed pressure |
| mg_p0.16 | 33124 | yes | 1 | 58 | 0.177 | 100 | determined by the prescribed pressure |
| mg_p0.5 | 33124 | yes | 1 | 51 | 0.167 | 100 | determined by the prescribed pressure |
| mg_p1.0 | 33124 | yes | 0 | 31 | 0.092 | 100 | determined by the prescribed pressure |
| mg_p1.5 | 33124 | yes | 0 | 32 | 0.095 | 100 | determined by the prescribed pressure |
| mg_p3.0 | 33124 | yes | 0 | 31 | 0.092 | 100 | determined by the prescribed pressure |
| vanka_p-0.16 | 33124 | yes | 1 | 450 | 0.336 | 100 | determined by the prescribed pressure |
| vanka_p-0.08 | 33124 | yes | 1 | 462 | 0.350 | 100 | determined by the prescribed pressure |
| vanka_p0 | 33124 | yes | 1 | 491 | 0.362 | 100 | determined by the prescribed pressure |
| vanka_p0.16 | 33124 | yes | 1 | 487 | 0.357 | 100 | determined by the prescribed pressure |
| vanka_p0.5 | 33124 | yes | 1 | 410 | 0.300 | 100 | determined by the prescribed pressure |
| vanka_p1.0 | 33124 | yes | 0 | 387 | 0.287 | 100 | determined by the prescribed pressure |
| vanka_p1.5 | 33124 | yes | 0 | 332 | 0.251 | 100 | determined by the prescribed pressure |
| vanka_p3.0 | 33124 | yes | 0 | 351 | 0.261 | 100 | determined by the prescribed pressure |
| mg_n2l4 | 10692 | yes | 0 | 16 | 0.038 | 100 | zero mean |
| mg_n2l8 | 75140 | yes | 0 | 27 | 0.192 | 100 | zero mean |
| mg_n4l8 | 561924 | yes | 0 | 71 | 2.298 | 100 | zero mean |
| vanka_n2l8 | 75140 | yes | 0 | 189 | 0.370 | 100 | zero mean |
| lshape | 7060 | yes | 3 | 16 | 0.371 | 20 | determined by the do-nothing outflow |
| steady | 2916 | yes | 3 | 13 | 0.049 | 20 | determined by the prescribed pressure |
| t1 | 2916 | yes | 3 | 13 | 0.051 | 20 | determined by the prescribed pressure |
| t2 | 2916 | yes | 3 | 16 | 0.062 | 20 | determined by the prescribed pressure |
| t4 | 2916 | yes | 6 | 26 | 0.099 | 20 | determined by the prescribed pressure |
| t6 | 2916 | yes | 4 | 35 | 0.133 | 20 | determined by the prescribed pressure |
| t8 | 2916 | yes | 5 | 44 | 0.169 | 20 | determined by the prescribed pressure |
| re500 | 4540 | yes | 5 | 24 | 0.228 | 500 | determined by the do-nothing outflow |

## Provenance

| field | value |
|---|---|
| generated | 2026-09-17 16:34:47 |
| machine | nid006549 |
| threads | 72 |
| commit | -- |
| linear solver | direct |
| element refine level | 1 |
| run directory | /Users/patrickzulian/Desktop/code/merge_git_repos/sfem/spikes/cvfem/verification_runs/grace-direct-4684979 |
| runs parsed | 47 of 47 |

Regenerate with `python3 python/cvfem_verify_report.py verification_runs/grace-direct-4684979`.
