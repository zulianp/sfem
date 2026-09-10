## The other way to apply the Jacobian

An SpMV of the assembled BSR is the alternative to the matrix-free action, so it belongs in
the cascade rather than in a table of its own — the question "what does a matvec cost" has
two answers and only one of them is matrix-free.

**462 MDOF/s at 4,121,204 dof, against 579 for the matrix-free action carrying the same
physics.** Matrix-free wins by 1.25x on the apply alone, and that is before counting the
assembly that produced the matrix: 87 MDOF/s on the colored layout and 39 on the atomic
one, which is 5 and 12 SpMVs' worth of work respectively, spent to build something that is
then slower to apply than not building it at all. The assembled operator earns its place as
a preconditioner — it is what block-Jacobi and the Schur diagonal are read out of — and not
as an apply.

**One row per problem size is enough, because the cost does not depend on the physics.**
The sparsity pattern is the mesh's node-to-node graph whatever terms the values carry, and
it stays that way by design: the exact Rhie–Chow term is deliberately kept out of the
assembled matrix precisely because it would couple pressures beyond nearest neighbours and
widen the pattern (`cvfem_hex8_ns_upwind_kernels.hpp:135`). That is an argument, though, not
a measurement, so the job measures it — bare, with Rhie–Chow, and with Rhie–Chow and the
boundary closure — and the three read 462, 461 and 457 MDOF/s, 1.1% apart. If they ever
stop agreeing, the pattern has widened and something more interesting than a throughput has
changed.

It varies mildly with size — 487 MDOF/s at 1,098,500 dof against 462 at 4,121,204 — and
that gap is 5%, only just outside the 3% these readings spread over, so it is worth
noticing and not worth explaining. What can be said is what it is *not*: the values are
about 840 bytes per dof at both sizes (878 MiB and 3329 MiB), so both matrices are far past
the 117 MiB of L3 and the SpMV is streaming from DRAM in both cases. Whatever the remaining
5% is, it is not a cache-residency effect.

That 840 bytes per dof is the durable point about this operator. The matrix-free action
reads the state and the mesh and recomputes everything else, so it moves a small constant
per dof no matter how much physics it carries; the SpMV moves the whole matrix every time.
That is why the SpMV's cost is flat across the physics — the traffic is the same — and why
adding arithmetic to the matrix-free operator narrowed the gap between them from 4.5x on
the bare kernel to 1.25x, without ever closing it.
