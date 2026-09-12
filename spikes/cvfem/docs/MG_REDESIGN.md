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

## 3. Coarse levels need the linearization point and the transient history

The paper states that the transfer operators carry, in addition to the defect and residual,

> the solutions of previous transient iterations, as required for the time-stepping scheme
> (uⁿ and uⁿ⁻¹), and the linearization point (û, p̂) to all the multigrid levels.

The code agrees: Lethe's `MFNavierStokesPreconditionGMG::initialize()` takes the previous
solution needed to evaluate the non-linear term and a vector of time derivatives of previous
solutions.

**We do the opposite.** Our `clone_onto` passes `dt` and the BDF order to coarse levels but
deliberately not the history, on the reasoning that a coarse operator is applied to a
correction and never asked for a residual. That reasoning is about the *residual*; it does not
license linearizing the coarse operator around a different state than the fine one. A coarse
operator carrying the right `dt` but the wrong convecting velocity is not a coarse version of
the fine operator, and a multigrid whose levels disagree about the operator is exactly the
failure mode we are seeing.

Action: transfer `û` (and `uⁿ`, `uⁿ⁻¹` when transient) to every level and rebuild the level
operators against them.

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
2. **Transfer the linearization point to coarse levels.** Correctness, not tuning.
3. **Re-measure the step case.** The yardstick is 2–5 linear iterations per Newton step; if it
   is still in the thousands after 1 and 2, the fault is elsewhere and this document is wrong
   about where.
4. Only then consider Chebyshev, sweep counts, and preconditioner reuse, which are
   optimisations of something that works.
