# Verification setup from Farrell, Mitchell & Wechsung (2019)

*An Augmented Lagrangian Preconditioner for the 3D Stationary Incompressible Navier–Stokes
Equations at High Reynolds Number*, SIAM J. Sci. Comput. **41**(5), A3073–A3096,
[arXiv:1810.03315](https://arxiv.org/abs/1810.03315).

Every formula below was read off the typeset PDF page, not from a text extraction — the
extractor flattens superscripts and silently turns `10x²y²` into `10x2 y 2`. Section and
equation numbers refer to the paper.

## Reynolds number (§2)

> The Reynolds number, defined as `Re = UL/ν` where `U` is the characteristic velocity and
> `L` is the characteristic length scale of the flow.

Note their momentum equation carries `1/Re` on the viscous term, so the *pressure* in the
manufactured solution below is Reynolds-dependent. Ours is parameterised by `rho` and `mu`
with `Re = rho·U·L/mu`; the manufactured pressure has to be evaluated with the same `Re` the
operator is actually running at, not a nominal one.

**Parameter mapping for the MMS.** Theirs is `(u·∇)u − (1/Re)Δu + ∇p = f`; ours is
`rho(u·∇)u − μΔu + ∇p = f`. Run the manufactured case with **`rho = 1`, `mu = 1/Re`** and their
`p` formula is our exact pressure with no rescaling. Any other `(rho, mu)` pair needs `p`
rescaled and will otherwise corrupt the measured pressure convergence rate silently.

## 1. Method of manufactured solutions (§5.3, eq. 5.2)

Rescaled to the square `[0,2]²`. With `u = (u₁, u₂)`:

```
u₁(x,y) =  1/4 (x−2)² x² y (y²−2)
u₂(x,y) = −1/4 x (x²−3x+2) y² (y²−4)

p̃(x,y) = [ x y ( 3x⁴ − 15x³ + 10x²y² − 30x(y²−2) + 20(y²−2) ) ] / (5 Re)
         − 1/128 (x−2)⁴ x⁴ y² ( y⁴ − 2y² + 8 )

p(x,y) = p̃ − 1/4 ∫_[0,2]² p̃(x,y) dx = p̃ + 1408/33075 − 8/(5 Re)
```

Extended to three dimensions as `u(x,y,z) = (u₁(x,y), u₂(x,y), 0)`, pressure unchanged.

**Checked here, not taken on faith:** this field is exactly divergence-free.
`∂u₁/∂x = x(x−1)(x−2)(y³−2y)` and `∂u₂/∂y = −x(x−1)(x−2)(y³−2y)`, which cancel identically.
That matters because a manufactured solution that is only approximately solenoidal would
contaminate the pressure convergence and look like a discretisation defect.

The constant `1408/33075 − 8/(5Re)` fixes the pressure gauge to zero mean — **verified
exactly in rational arithmetic**: `(1/4)∫∫_[0,2]² p̃ = 8(6615 − 176·Re)/(33075·Re)`, which is
exactly `8/(5Re) − 1408/33075`.

A free cross-check between two of the three cases: `u₁(x, y=2) = x²(2−x)²`, which is exactly
the regularized cavity lid profile of §2 in the x-direction. If the MMS and the cavity
disagree on that boundary datum, one of the two transcriptions is wrong. With a pressure
pin rather than a zero-mean constraint, either the pin value must be set from this exact `p`
or the computed pressure compared only up to a constant.

**Expected convergence** (theirs, for `[P1 ⊕ B₃ᶠ]³−P0`): first order in pressure from the DG0
space; second order in velocity, degrading to first on coarse meshes because of SUPG and
recovering once `h⁻¹ ≳ Re`. Our discretisation is different — colocated FV with Rhie–Chow and
first-order upwinding — so **their rates are not our target**. We should expect first order in
velocity from the upwinding, and the MMS is worth having precisely because it measures what we
actually get rather than what we assume.

## 2. 3D lid-driven cavity (§5.5)

Ω = `[0,2]³`. No-slip on all boundaries except the top `{y = 2}`, where

```
u(x,y,z) = ( x²(2−x)² z²(2−z)² , 0 , 0 )ᵀ
```

This is the **regularized** lid: the velocity vanishes at the edges, removing the corner
singularity. Our current cavity case uses a constant lid velocity, so it is *not* comparable
to their numbers and the difference is not cosmetic — the singular corners are what limit the
attainable Reynolds number.

Reference Krylov iterations per Newton step (Table 5.6), 2.1×10⁶ → 1.1×10⁹ dof:

| refinements | dof | Re=10 | 100 | 1000 | 2500 | 5000 |
|---|---|---|---|---|---|---|
| 1 | 2.1×10⁶ | 4.50 | 4.00 | 5.00 | 4.50 | 4.00 |
| 2 | 1.7×10⁷ | 4.50 | 4.33 | 4.50 | 4.00 | 4.00 |
| 3 | 1.3×10⁸ | 4.50 | 4.33 | 4.00 | 3.50 | 7.00 |
| 4 | 1.1×10⁹ | 4.50 | 3.66 | 3.00 | 5.00 | 5.00 |

Run to failure at 1.7×10⁷ dof: iteration counts stay flat to Re=7000, rise thereafter, and
convergence fails at **Re=7700**.

## 3. 3D backward-facing step (§5.5)

```
Ω = ( ([0,10] × [1,2]) ∪ ([1,10] × [0,1]) ) × [0,1]
```

An L-shape in the x–y plane extruded one unit in z: the box `[0,10]×[0,2]×[0,1]` with the
notch `[0,1]×[0,1]×[0,1]` removed. Step at `x=1`, step height 1, inlet channel height 1,
outlet channel height 2 — an expansion ratio of 2.

Boundary conditions:

- **Inflow** on `{x = 0}`: `u(x,y,z) = ( 4(2−y)(y−1) z(1−z) , 0 , 0 )ᵀ`.
  Supported on `y ∈ [1,2]`, which is the whole inlet face, and parabolic in both `y` and `z`.

  **Corrected:** an earlier revision said this peaks at 1. It peaks at **1/4** — `4(2−y)(y−1)`
  peaks at 1 and `z(1−z)` peaks at 1/4, and the product at `(y,z) = (3/2, 1/2)` is 1/4. This
  is not pedantry: it sets `U`, and therefore which `mu` reproduces their Table 5.7 column
  labels. Report `Re` computed from both the peak and the mean and say which the run used.

  Two exact numbers worth having, both checked symbolically:
  **volumetric flux = 1/9**, and with an outlet area of 2 the **mean outflow speed = 1/18**.
  The flux is the oracle for the mass-balance check — it is what detects an unclosed step
  face, and it fails by a large margin rather than subtly.
- **Natural (do-nothing) outflow** on `{x = 10}`.
- **No-slip** on everything else, including the step faces.

Reference Krylov iterations per Newton step (Table 5.7):

| refinements | dof | Re=10 | 100 | 1000 | 2500 | 5000 |
|---|---|---|---|---|---|---|
| 1 | 2.1×10⁶ | 4.50 | 4.00 | 4.00 | 4.50 | 7.50 |
| 2 | 1.7×10⁷ | 5.00 | 4.00 | 3.33 | 4.00 | 10.00 |
| 3 | 1.3×10⁸ | 6.50 | 4.50 | 3.50 | 3.00 | 8.00 |
| 4 | 1.0×10⁹ | 7.50 | 3.50 | 2.50 | 3.00 | 6.00 |

## 4. Solver settings they used

- Continuation in Reynolds number as the only globalization, Newton with the PETSc `L2` line
  search, FGMRES outer.
- For the 3D runs specifically: SUPG stabilization **reduced by a factor 1/20**, linear
  solver relative tolerance relaxed to `1e-5`, and absolute tolerance for both linear and
  nonlinear solvers relaxed to `1e-8`.
- Runs stopped at Re=5000 for budget reasons, not because the solver failed.

## 5. What this needs that we do not have

Ordered by how much work each is, not by importance.

1. **A body force in the CVFEM operator.** The MMS is driven by
   `f = −(1/Re)Δu + (u·∇)u + ∇p`, and there is no source term anywhere in
   `src/hex8/cvfem_hex8_ns_core.hpp` or the operator. Without it the manufactured solution cannot be
   imposed at all. This is a kernel change: the forcing has to be integrated over each
   sub-control volume the same way the other terms are.
2. **Geometry-aware boundary faces.** `hex8_face_on_domain` in
   `src/hex8/cvfem_hex8_boundary_scs.hpp` decides a face is on the boundary by testing all four nodes
   against `x=0, x=Lx, y=0, y=Ly, z=0, z=Lz`. That is hard-coded for a box and is simply
   wrong on the L-shaped step domain, where the step faces at `x=1 (y<1)` and `y=1 (x<1)` are
   boundary but lie on no such plane. Needs a real boundary-face marking rather than a
   coordinate test.
3. **Natural outflow.** Every case so far is Dirichlet everywhere plus a pressure pin. The
   step needs an open outflow at `x=10`.

   **Correction.** An earlier revision of this document claimed that with `|Γ_N| > 0` the
   pressure pin must be dropped because the pressure becomes determined. That is true of the
   FEM do-nothing condition but **false for this discretisation.** Reading
   `src/hex8/cvfem_hex8_boundary_scs.hpp:170-174`, the boundary term is

   ```c
   const scalar_t mdot = rho * (ux[i]*ax + uy[i]*ay + uz[i]*az);
   r[i*4+0] += mdot * ux[i] + p[i] * ax - tau_x;      // and y, z
   r[i*4+3] += mdot;
   ```

   It keeps `p_i·a`. The area vectors over a closed control volume sum to zero, so a uniform
   pressure shift leaves every momentum residual unchanged: **the constant-pressure nullspace
   survives and the pin is still required.** Dropping the Dirichlet condition at `x=10` gives a
   well-defined, correctly linearised zero-gradient finite-volume outflow — not a do-nothing
   condition. Obtaining a gauge-fixing do-nothing means *replacing* the outflow traction
   (dropping `p_i·a − tau`, keeping `mdot·u_i`), which is a kernel physics change.

   **Backflow hazard in the same three lines:** `mdot·u_i` carries no upwind switch, unlike the
   interior kernel's `mpos`/`mneg`. For outflow (`mdot > 0`) `u_i` is the upwind value and this
   is correct; for inflow through an unconstrained face it is the *downwind* value — the classic
   finite-volume backflow instability, and a recirculating step outlet is where it bites.
4. **An L-shaped hex mesh in smesh.** The cheapest of the four: build the macro mesh as a
   structured grid over `[0,10]×[0,2]×[0,1]` with the notch cells dropped and the node
   numbering compacted, then hand it to the existing `to_semistructured`. No new
   semistructured machinery is required, because the macro mesh carries the geometry and the
   lattice refinement is the same everywhere.

Items 2 and 3 are the ones that actually gate the step benchmark; the mesh is not the hard
part despite being the visible one.

---

# Measured results

All on an Apple M1 (10 cores), 8 threads, under other load — **wall-clock figures are
indicative only; iteration counts are the comparable quantity.**

## 1. Manufactured solution — the discretisation is second order

`[0,2]³` at Re=1, volume-weighted `L²`, pressure compared up to a constant.

| h | dof | `u_l2` | order | `p_l2` (gauge-shifted) | order |
|---|---|---|---|---|---|
| 0.5000 | 500 | 3.970e-01 | — | 2.320e+00 | — |
| 0.2500 | 2,916 | 9.158e-02 | 2.12 | 9.321e-01 | 1.32 |
| 0.1250 | 19,652 | 2.015e-02 | 2.18 | 3.451e-01 | 1.43 |
| 0.0625 | 143,748 | 4.924e-03 | 2.03 | 1.345e-01 | 1.36 |

**Second order in velocity, ≈1.4 in pressure**, both stable across the ladder rather than
drifting. Re=1 is the informative point: upwinding is inactive there, so first order would
have indicated a consistency error rather than benign upwind diffusion. This is the first
measurement of this solver's order of accuracy.

Two measurement choices were necessary; the naive ones actively mislead. In `L∞` the same
data reads 1.92/2.35/1.43 for velocity and 0.90/0.53/0.38 for pressure — a pressure order
that appears to collapse. `L∞` is set by a single worst node, and the pin lets the gauge
drift (`dp_mean` 0.033 → −0.151), which is then charged as error at every node.

The pin was the obvious suspect for the pressure behaviour and was measured innocent: it sits
at (0,0,0), the worst velocity error is 21 h away and the worst pressure error 22.6 h away,
and excluding balls of radius 1h, 2h and 4h around it changes neither error in any digit.

Re=200 and Re=500 do not converge at these resolutions (cell Reynolds number ≈50 on the
coarse meshes). Expected without the AL preconditioner.

## 2. Regularized lid-driven cavity

`[0,2]³`, 32³ cells, 19,652 dof, fully 3D no-slip.

| Re | reached | Newton | Krylov/Newton | theirs (Table 5.6) | ratio |
|---|---|---|---|---|---|
| 10 | 10 | 7 | 192.9 | 4.50 | 43× |
| 100 | 100 | 13 | 86.2 | 4.00 | 22× |
| 1000 | 928.2 | 62 | 752.7 | 5.00 | 151× |

Their count is flat across two decades; ours improves to Re=100 then degrades ninefold by
Re=1000. **That divergence is the augmented-Lagrangian preconditioner's contribution**, which
is out of scope here — not a shortfall of the discretisation.

**The regularization matters, and by how much was checked rather than asserted:** at the same
mesh and Reynolds target the regularized lid reaches **928.2** where the constant lid reaches
**549.5**, a factor of 1.69. The corner singularity, not the solver, is what caps the
constant-lid case. Note the constant-lid case keeps `uz=0` slip spanwise walls, which is
correct for comparison against Ghia et al.'s 2D data and wrong for comparison against theirs.

## 3. Backward-facing step

40×8×4 macro at level 2, 47,268 dof, Re=20, `SFEM_GMG=0`.

**Global mass conservation: Σ continuity residual = 7.04e-14.** Every control volume is
closed, including the two step faces — which is exactly what fails if the boundary marking
misses them.

That check is quadrature-free and is the conservation test. A cruder plane-integrated flux
comparison gives 0.1077 in and 0.0896 out, a 17% "imbalance" that is entirely trapezoidal
error: the inlet is a smooth parabola (3% low against the exact 1/9) while the outlet profile
is developing and recirculating. The inlet figure is worth keeping as a check on the
prescribed profile; the outlet one is not a conservation measure.

The outflow required the genuine do-nothing condition. Dirichlet outlet: solves, Newton
quadratic. Unconstrained outlet with `p_i·a` retained: diverges, `|dx|_inf` 108 against 49.
Unconstrained and unpinned: linear solve diverges outright — confirming the nullspace
survives. Prescribing `(pI − τ)·n = 0` fixes the gauge and lets the pin come off.

**Resolved since this was written.** The paragraph here used to record that multigrid did
not work for this case because `src/ss/cvfem_ss_galerkin.hpp` called the boundary term
without the masks. Both halves of that are now out of date. The Galerkin coarse operator
passes both the face mask and the natural-outflow mask down through
`sscvfem_micro_face_mask` (`src/ss/cvfem_ss_galerkin.hpp:315-322`), and the case solves:

```
highest Re SOLVED = 20 of 20 target  (AT TARGET)
newton_converged: 1   16 Newton steps over 4 stages   7686 linear iterations   31.2 s
sum of continuity residual 5.111215e-15  (sum |.| 4.156488e-14)
```

quadratic in every stage, with mass conserved to 5e-15 — which is the quantity this
verification case is judged on. The missing masks were real but were not the last obstacle:
what actually unblocked it was switching the Vanka sweep to the additive form at an open
outlet. The multiplicative 8-colour sweep is the better smoother on a closed box (0.63 at
omega = 1 against 0.883) and that ranking reverses once the outlet opens, because colour
order and characteristic direction disagree where more than half the outlet faces are
reversed. See the commit that records the Brandt regime diagnostic behind it.
