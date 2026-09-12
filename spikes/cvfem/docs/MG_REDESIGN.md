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

| coarse operator | applied operator | Rhie–Chow `D_f = rc_scale·h²/(2μ)` |
|---|---|---|
| Lethe, matrix-free rediscretization | `A_H(I_H û)` | τ recomputed per level; consistent by construction |
| ours, `derefine_op` | `A_H(R û)` | recomputed at coarse `h`, so `D_f` rises ~8× per level in 3D, then hand-corrected by `SFEM_GMG_RC_DECAY` |
| ours, element-wise Galerkin (`SFEM_GMG_EGAL=1`, **the default**) | `Σ_e P_eᵀ A_e P_e` | frozen at the **fine** `h` |

Two consequences.

First, our default coarse operator is the element-wise Galerkin one, which freezes the fine
grid's stabilization — the very inconsistency `derefine_op`'s own comment gives as the reason
not to use `PᵀAP`. Worse, element-wise Galerkin is exact only when the operator is a sum of
element contributions, and Rhie–Chow couples through a **nodal** pressure gradient, which is
not element-local unless it is frozen from the state. The driver already contains a gate that
tests exactly this and prints `LOCAL (element-wise Galerkin is exact)` or `NON-LOCAL`.

Second, `SFEM_GMG_RC_DECAY` is a hand-tuned scalar standing in for a per-level spectral
property — its comment says 0.25 "keeps `D_f` fixed at the fine level's value". That is the
same quantity item 1 proposes to *measure*. The paper does not tune a decay factor; it
estimates each level operator's spectrum and sets ω from it, for the stated reason that the
stabilization depends on element size and changes per level. Our decay knob and their power
iteration are two answers to one question, and only one of them is calibrated against the
operator it is correcting.

Action, replacing the earlier one: settle which coarse operator the hierarchy actually uses
(`SFEM_GMG_EGAL`, `galerkin_mode`), check the element-locality gate on a Rhie–Chow-active
case, and treat `SFEM_GMG_RC_DECAY` as a symptom — if per-level ω from item 1 works, the decay
factor should become unnecessary rather than merely better-chosen.

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
2. **Settle the coarse operator.** Which of element-wise Galerkin and rediscretization the
   hierarchy uses, whether the element-locality assumption holds with Rhie–Chow active, and
   whether `SFEM_GMG_RC_DECAY` survives a calibrated ω. Not a state-transfer problem: the
   linearization point already reaches every level.
3. **Re-measure the step case.** The yardstick is 2–5 linear iterations per Newton step; if it
   is still in the thousands after 1 and 2, the fault is elsewhere and this document is wrong
   about where.
4. Only then consider Chebyshev, sweep counts, and preconditioner reuse, which are
   optimisations of something that works.
