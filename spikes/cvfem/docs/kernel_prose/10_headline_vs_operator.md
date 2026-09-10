## What the headline number leaves out

The throughput this spike quotes is measured on an operator the solver never runs. That is
a deliberate and useful thing to measure — it isolates the element kernel from everything
around it, which is how a change to the arithmetic is seen without the rest of the solve
moving underneath it — but it is not the cost of an apply in the Newton loop.
`src/hex8/cvfem_hex8_ns_core.hpp:5` says the two families "differ in physics, not just in
layout"; the tables above are what that sentence costs.

**The residual's headline overstates the real operator by 2.0x.** 2916 MDOF/s is the
element kernel on the packed layout; the same kernel carrying Rhie–Chow, the boundary
closure and the transient term is 1445. Quoting the first as the operator's throughput is
not wrong, but it needs the qualifier, and this report is what the qualifier should point
at.

**For the Jacobian action the factor is 3.6x, not 2.0x**, and that asymmetry is the thing
worth taking away. 2032 bare against 571 for the operator the Krylov loop evaluates. The
action pays for something the residual has no analogue of: the *exact* Rhie–Chow Jacobian
differentiates through the nodal pressure-gradient reconstruction, so every matvec rebuilds
that reconstruction for the Krylov direction — a full element sweep that cannot be hoisted
out of anything, because the direction changes with every iteration. Turning the term on
alone takes the action from 2032 to 618; everything after that is comparatively cheap.

An earlier version of this analysis reported ~1.9x and measured the residual only, because
no benchmark row for the action with Rhie–Chow existed to measure. It does now, and the
number it produces is nearly twice as large. The residual figure it gave, 2579 against
1377, agrees with the 1.87x this run reads for the same two rows.

**Where the assembled matrix sits.** The SpMV of the assembled Jacobian runs at 449 MDOF/s
against 584 for the matrix-free action carrying the same physics, so matrix-free wins by
1.3x on the apply alone — before counting the assembly that produced the matrix, which
costs 39 MDOF/s on the atomic layout. The assembled operator earns its place as a
preconditioner, not as an apply.
