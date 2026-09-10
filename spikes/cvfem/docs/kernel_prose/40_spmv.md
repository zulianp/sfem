## The other way to apply the Jacobian

An SpMV of the assembled BSR is the alternative to the matrix-free action, so it belongs in
the cascade rather than in a table of its own — the question "what does a matvec cost" has
two answers and only one of them is matrix-free.

**446 MDOF/s at 4,121,204 dof, against 888 for the matrix-free action carrying the same
physics.** Matrix-free wins by 1.99x on the apply alone, and that is before counting the
assembly that produced the matrix: 83 MDOF/s on the colored layout and 38 on the atomic
one, which is 5 and 12 SpMVs' worth of work respectively, spent to build something that is
then slower to apply than not building it at all. The assembled operator earns its place as
a preconditioner — it is what block-Jacobi and the Schur diagonal are read out of — and not
as an apply.

That margin was 1.25x until recently, and it widened without the SpMV moving at all: the
matrix-free operator got faster. Which is the shape of this comparison in general — the
SpMV's cost is set by the matrix and does not respond to anything done to the kernel, so
every improvement on the matrix-free side is a full improvement in the ratio.

**One row per problem size is enough, because the cost does not depend on the physics.**
The sparsity pattern is the mesh's node-to-node graph whatever terms the values carry, and
it stays that way by design: the exact Rhie–Chow term is deliberately kept out of the
assembled matrix precisely because it would couple pressures beyond nearest neighbours and
widen the pattern (`cvfem_hex8_ns_upwind_kernels.hpp:135`). That is an argument, though, not
a measurement, so the job measures it — bare, with Rhie–Chow, and with Rhie–Chow and the
boundary closure — and the three read 446, 445 and 444 MDOF/s, 0.5% apart. If they ever
stop agreeing, the pattern has widened and something more interesting than a throughput has
changed.

It barely varies with size — 447 MDOF/s at 1,098,500 dof against 446 at 4,121,204, inside
the spread of either. That is itself worth a sentence, because the obvious explanation for
any size dependence would be cache residency and it is not available here: the values are
about 840 bytes per dof at both sizes (878 MiB and 3329 MiB), so both matrices are far past
the 117 MiB of L3 and the SpMV streams from DRAM in both cases.

That 840 bytes per dof is the durable point about this operator. The matrix-free action
reads the state and the mesh and recomputes everything else, so it moves a small constant
per dof no matter how much physics it carries; the SpMV moves the whole matrix every time.
That is why the SpMV's cost is flat across the physics — the traffic is the same — and why
adding arithmetic to the matrix-free operator narrows the gap between them from 4.5x on
the bare kernel to 2.0x, without ever closing it.
