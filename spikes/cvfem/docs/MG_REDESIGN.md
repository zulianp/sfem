# Multigrid redesign: lessons from a matrix-free stabilized Navier–Stokes solver

Source: Prieto Saavedra, Munch & Blais, *A matrix-free stabilized solver for the incompressible
Navier–Stokes equations* (SSRN 4981567, Oct 2024), and the Lethe implementation it describes
(`chaos-polymtl/lethe`, on deal.II).

Their solver is close enough to ours to be directly informative: **equal-order continuous
elements, stabilized so the pressure block is non-zero, solved monolithically by Newton–GMRES
with a matrix-free geometric multigrid preconditioner.** Our Rhie–Chow term plays the role
their PSPG term plays — it fills the pressure–pressure block and removes the saddle-point
structure, which is what makes monolithic GMG admissible in the first place. So the
architecture we already chose is the one they validate; what differs is in the details of the
smoother, and those details are where our multigrid currently fails.

## The number that says we are misconfigured, not merely untuned

They report **2–5 linear iterations per Newton step** with GMG, against >40 with ILU. Our
backward-facing step measures **3,899 per Newton step with block-Jacobi and 10,952 with
multigrid**, converging in neither case, at 13,644 dof. That is three orders of magnitude from
their yardstick. A configuration that far off is not short of tuning; something is structurally
wrong, and the paper names three candidates we can check.

## 1. The relaxation parameter must be computed per level, from the level operator

This is the highest-value item and the cheapest to try.

Their smoother is a damped relaxation `x ← x + ω P⁻¹ r`, with `ω = 2/(λ_min + λ_max)` of
`P⁻¹A`, where `λ_max` comes from **20 power iterations on that level's operator** and
`λ_min = λ_max/ψ` with `ψ` between 2 and 20. The paper is explicit about why this cannot be a
constant:

> The estimation of the eigenvalues for each level operator is crucial since the stabilization
> terms depend on the size of the element and it changes for each level.

**Our stabilization has exactly that property.** The Rhie–Chow coefficient
`c_s = α ρ |d|² |a|² / (2μ (a·d))` is a function of the element geometry, so the operator's
spectrum changes from level to level in the same way theirs does — and we apply a single
`SFEM_VANKA_OMEGA`, defaulting to 1, to every level. A previously recorded failure in this
spike is that the damping in use made the smoother *asymptotically divergent*; a fixed ω across
levels is a sufficient explanation for that, and an eigenvalue estimate per level is the
standard remedy rather than a novel one.

Action: estimate `λ_max` per level by power iteration at setup, derive ω, and apply it per
level. Cheap — 20 matvecs per level, once.

## 2. Which smoother works depends on steady versus transient, and on the order

Their results split cleanly, and the split matches our observations:

| regime | order | inverse-diagonal smoother | cell-block (ASM) smoother |
|---|---|---|---|
| steady | Q₂/Q₃ | works, 13–96× faster than matrix-based | works |
| steady | **Q₁Q₁** | **not robust** — results not shown | required |
| transient | **Q₁Q₁** | **robust**, 2.4–3.9 linear its/Newton to 540M dof | not needed |

We are the Q₁Q₁ row: lowest-order, equal-order, colocated. So:

- Our transient pump converging under block-Jacobi is **expected**, not luck.
- Our steady step failing under block-Jacobi is **expected**, and is not a defect in the case.
- For steady problems at our order a **cell-block smoother is not optional**.

We have one — Vanka — but it is used as a standalone multiplicative or additive sweep, where
theirs is an additive Schwarz *preconditioner inside the damped relaxation of item 1*. Our
additive variant is documented as needing ~480× the iterations and our multiplicative variant
diverges at an open outlet. Theirs is additive and works, which points at the missing ω rather
than at the Schwarz idea.

Their honest caveat, which applies to us: their ASM is implemented naively — it assembles the
sparse matrix to extract the blocks — so its memory is comparable to a matrix-based code, and
they flag this as the thing to improve.

## 3. The coarse operator differs in its stabilization scale, not in its state

An earlier draft of this section claimed we fail to transfer the linearization point to the
coarse levels. That is false, and the correction is worth stating because it moves the real
issue somewhere more interesting.

**The linearization point is already restricted to every level.** Once per Newton step the
driver runs `restrictions[i-1]->apply(states[i-1], states[i])`, divides by the transfer weight,
and calls `apply_constraints` on each level's state. The level operators are then updated
against it. We do what Lethe does here.

**The transient history provably cannot change the applied operator.**
`transient_diag_weight` returns `a0·ρ/dt` when a history is present and `1.5·ρ/dt` for BDF2
when it is not — and for BDF2, `a0 = 1.5`. The two are identical. The history enters the
*residual* and nothing else, and a coarse operator is only ever applied to a correction. The
existing comment declining to require a history on coarse levels is correct, not a loophole.

So Lethe transferring `uⁿ` and `uⁿ⁻¹` to all levels is about *their* formulation, where the
time derivative appears inside the SUPG/PSPG residual `∂ₜu + (u·∇)u + ∇p − νΔu − f` and
therefore does reach the Jacobian. Ours is a lumped diagonal term that does not.

### What actually differs

Both we and they rediscretize rather than Galerkin-coarsen, and for the same reason: the
stabilization parameter is a function of element size, so `PᵀAP` inherits the fine-grid
stabilization and is inconsistent. The difference is what happens to that parameter.

| coarse operator | selected by | Rhie–Chow `D_f = rc_scale·h²/(2μ)` |
|---|---|---|
| Lethe, matrix-free rediscretization | — | τ recomputed per level; consistent by construction |
| ours, `derefine_op` rediscretization | `SFEM_GMG_GALERKIN=0`, **the default** | recomputed at coarse `h`, so `D_f` quadruples per level (it carries `h²`), then hand-corrected by `SFEM_GMG_RC_DECAY`, which defaults to 1 — i.e. uncorrected |
| ours, element-wise Galerkin | `SFEM_GMG_GALERKIN=2` with `SFEM_GMG_EGAL=1` | frozen at the **fine** `h`; exact only under the frozen-pgrad Jacobian |

(An earlier draft of this table said element-wise Galerkin was the default. It is not:
`SFEM_GMG_EGAL` defaults to 1 but is gated on `galerkin_mode == 2`, and `SFEM_GMG_GALERKIN`
defaults to 0. The default hierarchy is rediscretized. The same draft said `D_f` rises 8× per
level; it is 4×, since the coefficient carries `h²` and not `h³`.)

## What is required for a consistent coarse space

A coarse operator is only ever applied to a *correction*, so the property that matters is
`A_H ≈ R A_h P` on the coarse space — not that `A_H` be a good discretization of the coarse
mesh. Those two requirements diverge exactly at the stabilization, and the code currently
pursues the second. Five things are required; the measurements below are from the cavity on
semi-structured macro-elements, 4 threads, and each is reproducible with `SFEM_GMG_CHECK=1`.

**1. The stabilization must be evaluated at the fine level's `h`.** On a smooth
coarse-representable mode the fine operator applies the fine `D_f` and the rediscretized
coarse operator applies 4× that, so the coarse solve returns a pressure correction 4× too
small. The commutation defect `|A_H R v − R A_h v| / |R A_h v|` per component, on a 4×4×4
macro mesh at L=4, 19,652 dof:

| `SFEM_GMG_RC_DECAY` | hop | ux | uy | uz | p |
|---|---|---|---|---|---|
| 1 (default) | 0 | 0.370 | 0.494 | 0.420 | **7.525** |
| 1 (default) | 1 | 0.495 | 0.389 | 0.410 | **8.413** |
| 0.25 | 0 | 0.370 | 0.494 | 0.420 | **1.409** |
| 0.25 | 1 | 0.495 | 0.389 | 0.410 | **0.599** |

The velocity figures are the rediscretization baseline: unchanged by the decay factor, by the
hierarchy (identical on a 2×1×1 mesh at L=8, 5,508 dof) and by `rho` from 1 to 0.01, so not
convective in origin. The pressure block sits an order of magnitude above that baseline at the
default and falls into it at 0.25. `SFEM_GMG_RC_DECAY=0.25` is a requirement, not a tuning
knob, and the default is the inconsistent setting.

**2. If the coarse operator is built element-wise, the fine operator must be element-local.**
`Σ_e P_eᵀ A_e P_e` equals `PᵀAP` only for an operator that is a sum of element contributions.
The exact Rhie–Chow Jacobian reconstructs the *direction's* nodal pressure gradient, which
reaches outside the element. On 2×1×1 at L=8, 5,508 dof:

| gate | `SFEM_RC_EXACT_JAC=1` (default) | `=0` (frozen pgrad) |
|---|---|---|
| element-locality, max abs response outside elem 0 | 1.2207e-02 — **NON-LOCAL** | 0.0000e+00 — LOCAL |
| `egal identity (q=1)`, must reproduce `A` itself | **5.954e-02 FAILED** | 1.469e-16 OK |
| `egal galerkin (0→1)` vs `PᵀAP` | **8.278e-01 FAILED** | 2.130e-16 OK |
| `egal level 1` vs probed composite | **9.583e-01 MISMATCH** | 1.874e-16 OK |
| `egal galerkin (0→2)`, `(0→3)` | 2.4e-16 OK | 2.4e-16 OK |

Levels 2 and 3 pass under both because they coarsen level 1's element matrices — self-consistent
and built on an 83%-wrong level. This failure mode was predicted in writing when the
construction landed ("changing Rhie-Chow to differentiate the pressure gradient would silently
invalidate the construction", `docs/README_alps.md`); `SFEM_RC_EXACT_JAC` subsequently came to
default to 1, and nothing runs the gate unless `SFEM_GMG_CHECK=1` is set. So
`SFEM_GMG_GALERKIN=2` requires `SFEM_RC_EXACT_JAC=0`, and the two defaults are mutually
inconsistent.

**3. `R = Pᵀ` on the residual, with the linearization point restricted separately.** Already
true: `g.Rmat[i] = g.Pmat[i]->transpose()`, while the state goes down through
`restrictions[]` divided by `state_weights[]`. Not a gap, and the two must stay distinct.

**4. The null space must be the same object on every level**, `P·N(A_H) ⊆ N(A_h)`. With a
pressure pin it is not: each level pins its own node, nothing makes it the same physical
point, and a coarse correction then arrives carrying an arbitrary constant offset — precisely
the near-null mode the smoother is worst at damping. The zero-mean gauge satisfies the
condition because `P` maps constants to constants; a pin needs `SFEM_GMG_PFILTER=1`, which
defaults to 0.

**5. The coarse space must contain what the smoother leaves behind.** Trilinear interpolation
on the macro-element lattice reproduces constants in each velocity component, the constant
pressure and smooth shear, so the space itself is not the problem here — the operator built
on it is.

### The consequence that inverts the current design

We already have a construction that is correction-consistent by definition and verified to
`2.1e-16`: `A_H = PᵀA_hP`, assembled element-wise, under the frozen Jacobian. It was rejected
because it "would inherit the fine-grid stabilisation and be inconsistent" — but inheriting
the fine-grid stabilization *is* requirement 1. That comment applies discretization-consistency
to an operator that only ever sees corrections. `SFEM_GMG_RC_DECAY=0.25` exists to
hand-simulate, with a single scalar, the property the Galerkin operator has exactly.

What none of this fixes: the ~0.4 velocity commutation defect is intrinsic to rediscretization
and does not shrink with problem size. A coarse correction is approximate by construction,
which is why item 1's per-level ω is not independent of this — the smoother has to be right
for the cycle to tolerate an approximate coarse solve.

Action, replacing the earlier one: set `SFEM_GMG_RC_DECAY=0.25` under rediscretization; if
element-wise Galerkin is used, pair it with `SFEM_RC_EXACT_JAC=0` and make that pairing a
build-time or startup assertion rather than a gate behind `SFEM_GMG_CHECK`; keep the zero-mean
gauge on every level. Then re-measure, because item 1's ω is calibrated against whichever
coarse operator these choices settle on.

## 4. Smaller things worth taking

- **Reuse the preconditioner across Newton steps.** They set it up once per *transient*
  iteration, not per Newton iteration. We rebuild the block diagonal every Newton step.
- **Smoothing sweeps: 5 pre and 5 post with the diagonal smoother, 2 and 2 with ASM.** We use
  3. Their point that "GMG performs in total 10 smoothing steps per iteration" is the context
  in which 2–5 linear iterations per Newton step should be read.
- **Chebyshev is available as an alternative smoother** (`PreconditionChebyshev` alongside
  `PreconditionRelaxation` in their operator header), selected at runtime. It needs the same
  eigenvalue estimate as item 1, so it comes almost free once that exists.
- **Coarse-grid solver** is one of: a direct solve, GMRES preconditioned by *p*-multigrid, or a
  single v-cycle of it. We use a direct coarse solve, which matches their default.

## What does not transfer

Their headline speedups (10–100×) come from **high-order** elements, where matrix-free
sum-factorization has the most to offer and where a matrix-based Jacobian is ruinous. At
Q₁Q₁ their matrix-free advantage is 1.6× steady and 3.4× transient. We are a low-order scheme
and should expect the low-order numbers, not the headline ones.

Their stabilization is residual-based (SUPG/PSPG) and ours is Rhie–Chow, which is not the same
operator; the structural consequence we rely on — a non-zero pressure block — is shared, but
none of their constants are ours.

## Suggested order of work

1. **Per-level ω from power iteration.** Smallest change, addresses a known-divergent smoother,
   and item 3 is hard to evaluate while the smoother is mistuned.
2. **Settle the coarse operator.** Measured, in item 3: the default hierarchy is
   rediscretized with `SFEM_GMG_RC_DECAY=1`, which leaves the coarse pressure block an order
   of magnitude off commuting with restriction, and element-wise Galerkin is wrong under the
   default exact Rhie–Chow Jacobian. Fix the two defaults before anything else is measured.
   Not a state-transfer problem: the linearization point already reaches every level.
3. **Re-measure the step case.** The yardstick is 2–5 linear iterations per Newton step; if it
   is still in the thousands after 1 and 2, the fault is elsewhere and this document is wrong
   about where.
4. Only then consider Chebyshev, sweep counts, and preconditioner reuse, which are
   optimisations of something that works.
