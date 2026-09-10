## What the headline number leaves out

The throughput this spike quotes is measured on an operator the solver never runs. That is
a deliberate and useful thing to measure — it isolates the element kernel from everything
around it, which is how a change to the arithmetic is seen without the rest of the solve
moving underneath it — but it is not the cost of an apply in the Newton loop.
`src/hex8/cvfem_hex8_ns_core.hpp:5` says the two families "differ in physics, not just in
layout"; the tables above are what that sentence costs.

**The residual's headline overstates the real operator by 2.11x**, 2912 MDOF/s against 1377
for the same kernel carrying Rhie–Chow, the boundary closure and the transient term. **For
the Jacobian action it is 2.43x**, 2018 against 830.

That second figure used to be 3.68x, and the difference is the point of the most recent
work rather than a change in the physics. The exact Rhie–Chow Jacobian differentiates
through the nodal pressure-gradient reconstruction, so every matvec rebuilds that
reconstruction for the Krylov direction — a term that cannot be hoisted out of anything,
because the direction changes with every iteration. That pass used to take 5.06 ms of a
7.34 ms matvec, 69% of it, for arithmetic that amounts to one small gradient per element.
It now takes 1.04 ms: the denominator it was recomputing every time is pure geometry and is
cached, and the sweep runs over packs with a ghost reduction instead of over the flat mesh
with 24 atomics per element. **The operator the solver's Krylov loop applies is 1.57x faster
as a result** — 576 to 906 MDOF/s at 8,586,756 dof, measured against the previous binary in
one allocation.

What is left of the exact term's cost is now in the element sweep rather than beside it:
turning Rhie–Chow on takes the action from 2018 to 936, and the boundary closure and the
transient term account for the rest. That is where a stored element tangent would act, and
it is the next thing worth measuring.
