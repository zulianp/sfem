# What a transient FDA-nozzle campaign would cost

What a time step costs on one Grace socket, and what that implies for a Re_t 500 transient DNS.
Every number here is read back out of a run's own `diag.csv` and phase table — 72 Neoverse V2
cores, `OMP_PROC_BIND=true`, `OMP_PLACES=cores`, exclusive node, `real_t` fp64 and `geom_t` fp32
(confirmed in the generated `smesh_config.hpp` of both the local and the Alps prefix).

Two caveats belong at the top rather than in a footnote, because they bound everything below.
**The top rung has since been measured three times** — 45.365, 44.741 and 44.842 s/step, a spread
of **1.4%** and a standard deviation of **0.7%**, so on this configuration the machine is far
steadier than the 5–11% `CVFEM_Performance.md` warns about, and the original run was the slowest of
the three. The two smaller rungs remain single samples. And **the campaign's operating point lies
outside the measured range in size** — the step size has since been measured directly, but nothing
finer than 7.01M dof can be run here at all, for the reason given below.

Reproduce with `scripts/dns_cost_model.py <run-tree>` against the output of
`jobs/dns_cost.sbatch`.

## What a step costs

Nozzle at Re_t 500, `SFEM_GMG=1`, Vanka smoother, dt = 1e-3, 20 steps, upwind (blend off).

| run | ndof | t_step | sd | rel | newton/step | lin/step | vanka_setup/rebuild |
|---|---:|---:|---:|---:|---:|---:|---:|
| L4 | 116,212 | 1.253 s | 0.174 | 13.9% | 3.11 | 114.2 | 18.1 ms |
| L8 | 893,924 | 4.864 s | 0.208 | 4.3% | 4.00 | 84.3 | 126.8 ms |
| L16 | 7,014,340 | 45.365 s | 1.247 | 2.7% | 4.00 | 59.3 | 3604.4 ms |

Linear iterations *fall* with refinement (114 → 84 → 59): the multigrid hierarchy deepens, so the
work per dof does not rise as fast as the dof count. The small rung is the noisiest at 13.9%,
which matters because it carries equal weight in any fit.

**Step 1 is excluded from every mean.** A fresh run has no previous step to guess from and walks
the whole Reynolds ramp: it costs **4.77× / 4.40× / 5.77×** the settled per-step cost at L4 / L8 /
L16. Including it would inflate a campaign estimate by roughly the ramp's share of 20 steps.

## The model, and why it does not extrapolate

Set-up is **measured**, not fitted — the driver's phase table times `vanka_setup` directly — so
only the smoothing term is fitted, as `c · ndof · lin`. That is one free parameter against three
points rather than two, and it was a deliberate correction: fitting both terms gave leave-one-out
errors of −27%, +47%, −25%, a curve that could not reproduce any rung it had not been shown.

With set-up measured, leave-one-out still gives:

| held out | predicted | measured | error |
|---|---:|---:|---:|
| L4 | 0.934 s | 1.253 s | −25.5% |
| L8 | 6.712 s | 4.864 s | **+38.0%** |
| L16 | 45.182 s | 45.365 s | −0.4% |

The L8 miss is far outside that rung's own 4.3% scatter, so it is structural rather than noise.
The cause is visible once both mechanisms are normalised per dof:

| run | vanka_setup µs/dof | smooth[L0] µs/dof |
|---|---:|---:|
| L4 | 0.1559 | 0.0160 |
| L8 | 0.1419 | 0.0188 |
| L16 | **0.5139** | **0.0284** |

**Neither column is flat.** Set-up is flat from L4 to L8 and then 3.6× worse at L16 — a working-set
transition, with the implied exponent moving from 0.954 to 1.625 — while smoothing rises steadily
throughout. So `c` is itself a function of size, and no single exponent spans the ladder. At L16,
set-up is 31.6% of a step and `smooth[L0]` 54.2%; everything else in the cycle — all coarse levels,
restriction, prolongation, the coarse factorisation — is under 6% combined.

## The ladder cannot be extended, and this is the binding constraint

Three points cannot separate two size-dependent mechanisms, and **no fourth point is available on
this machine**:

* The operator refuses internal levels that are not powers of two (`src/op/cvfem_hex8_ns_op.cpp`).
  The hoisted macro-element geometry is computed once per macro element and reused for all L³ micro
  cells, which is exact in floating point only when the subdivision is a power of two — `geom_t` is
  fp32, so 1/2 and 1/4 are representable and 1/3 and 1/6 are not. The refusal carries its own
  measurements: naive-vs-macro operator agreement of 2.53e-16 at L=2 against **2.12e-06 at L=3**,
  which moves a converged Poiseuille solution from 1e-11 to **8e-02**. Confirmed empirically here:
  levels 3, 5 and 6 exit non-zero; level 2 runs.
* So the next rung is **L=32: ~56.1M dof, ~206 GB** extrapolated against L16's measured 25.7 GB —
  past one socket.

The guard is correct and should not be relaxed to obtain a data point. Lifting it needs the fix its
own comment names — computing geometry per micro cell in that sweep — which costs what the hoisting
was introduced to save and wants its own measurement.

## The step-size penalty runs the wrong way

Shrinking dt does **not** buy cheaper steps. At L8:

| dt | t_step | newton/step | lin/step | lin/newton |
|---:|---:|---:|---:|---:|
| 4.0e-3 | 3.916 s | 4.42 | 62.8 | 14.2 |
| 1.0e-3 | 4.864 s | 4.00 | 84.3 | 21.1 |
| 2.5e-4 | 6.022 s | 3.16 | 109.8 | **34.8** |

The transient term loads the velocity diagonal only — continuity carries no time derivative — so a
smaller dt unbalances the saddle point and stiffens the pressure coupling. A campaign therefore
pays twice for a small step: more steps *and* dearer steps.

## The averaging window, from measurement

Derived from the solver's own reported throat speed rather than from a quoted figure:

    u_throat  0.4143 m/s        (measured, L16 run)
    u_bulk    0.04603 m/s       by continuity, (r_throat/r_inlet)^2
    residence 3.476 s           over 0.16 m of 6 mm pipe
    window    17.38 s           five residence times

At dt = 1.75e-4 that is **99,307 steps**, independently reproducing the "order 10⁵ steps" the plan
assumed.

## What the campaign would cost

**Measured at the campaign operating point**, not extrapolated to it. `jobs/dt_campaign.sbatch`
ran L16 at dt = 1.75e-4 directly: **49.558 s/step** (19 steps, 10.2% scatter), against 45.365 s at
dt = 1e-3. Freeze strides are from `jobs/vanka_freeze.sbatch` and `jobs/vanka_freeze_tail.sbatch`,
where iteration counts do not move with the stride, so the saving is per-step and real.

| dt | steps | freeze=1 | freeze=2 | freeze=4 |
|---:|---:|---:|---:|---:|
| 1.75e-4 (campaign) | 99,307 | **57.0** | 48.0 | **43.3** |

socket-days — so **43 to 57 segments** of 24 hours.

**The dt penalty is size-dependent, and weaker exactly where it matters.** Shrinking dt from 1e-3
to the campaign value costs **+38.3%** per step at L8 but only **+9.2%** at L16, because at the
campaign dt lin/newton is 40.2 at L8 against 24.6 at L16 — the larger problem was already
iteration-bound, so the stiffer pressure coupling buys less. Extrapolating the L8 dt penalty onto
L16, which is the obvious move, would have **overstated** the campaign substantially. This is the
one place where declining to extrapolate changed the answer rather than merely hedging it.

For reference, the earlier floors computed from the dt = 1e-3 per-step cost were 52.1 / 43.9 /
39.6 socket-days. The measured figures land **9.2% above** them, so the floor was sound.

## Go / no-go

**Go, at 43–57 socket-days.**

The machinery is ready and verified. The restart chain reproduces an uninterrupted run exactly,
including running statistics whose summary line is character-identical across a segment seam; a
failed step aborts or cuts back rather than being silently accepted; the transient multigrid path
is gated against the direct solve; and end-to-end temporal order is confirmed at BDF1 1.043 and
BDF2 1.993. A campaign of 43–57 twenty-four-hour segments is exactly what that chain was built
for, and the cost is now measured at its own operating point rather than extrapolated to it.

The freeze stride is the one lever worth taking before launch: `SFEM_VANKA_FREEZE=4` is measured
to leave Newton and linear counts unmoved at every size and stride tested, and it takes the
campaign from 57 to 43 socket-days — two weeks of socket time for a setting whose iteration counts
do not move.

What remains genuinely unknown is the cost of any mesh **finer** than L16, and that is unknowable
here rather than merely unmeasured: the ladder cannot be extended on this machine at all. If the
campaign is ever to run finer than 7.01M dof, the per-micro-cell geometry fix named above has to
come first, and its cost wants its own measurement.

Two further things would change the picture and are worth knowing before launch rather than after:
the size scaling cannot be refined on this machine at all, so the cost at any mesh finer than L16
is unknown in principle; and `vanka_setup`'s 3.6× per-dof jump between L8 and L16 is unexplained —
if it is a working-set effect it may worsen again, and it is already 31.6% of a step at L16.
