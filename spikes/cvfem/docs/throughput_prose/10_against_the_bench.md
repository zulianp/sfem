## Against the bench: what the missing factor was

perf/baseline_grace.csv records `jac_action_packed_sumfact` at 1933 MDOF/s, and
re-measured on one allocation with three repeats it is 1695 (spread 3%). The scope
above read 607-625 when this was first investigated, and reads 851-911 now that the
Rhie-Chow coefficient is hoisted. These are the same kernel, and the trace figure
already EXCLUDES the boundary pass and the nodal gradient -- they are sibling
scopes, listed separately in the tables below -- so the cascade in
docs/CVFEM_Operator_Cascade.md did not account for the difference.

The bench could not measure the configuration the solver runs: it refused
`--rhie-chow` with `--jac-action` on `--layout packed`, because the bench's packed
staging never carried the term. That staging now exists (`--rhie-chow` with
`--jac-action`, layout packed or store, geom affine), and it agrees with the atomic
reference to 7.9e-16 relative, so the two measurements can finally be put on the
same axis.

Grace, one node, 72 threads, OMP_PROC_BIND=true, --exclusive, 11,212,884 dof
(n=140), 20 repeats after 5 warmup:

| packed Jacobian action | MDOF/s | vs. the plain kernel |
|---|---:|---:|
| element kernel, no Rhie-Chow (what the baseline quotes) | 1912 | 1.00x |
| element kernel, with Rhie-Chow, coefficient computed in the face loop | 882 | 2.17x slower |
| element kernel, with Rhie-Chow, coefficient hoisted | 1460 | 1.31x slower |
| the whole matvec: hoisted kernel + the direction's nodal gradient | 602 | 3.18x slower |

Rhie-Chow appeared to cost the packed element kernel 2.17x, and most of that turned
out not to be the term at all. The mass-flux coefficient carries a
degenerate-geometry guard, and with that guard inside the twelve sub-control-surface
face loops GCC 13.3 refuses to vectorise them -- `-fopt-info-vec-all` says
"vectorization is not profitable" -- while without it all twelve vectorise. The
guard's arithmetic is not the cost: measured against builds differing only in that
function's body, its sqrt is worth 1.3% and its two divides 0.7%, and rewriting the
early returns branchlessly recovers only 9%. Its presence in the loop is the cost.

The coefficient is pure geometry, so it is now built once per element per Newton step
and read in the face loop -- the arrangement the semi-structured path has always used
(`SSMacroGeom::coeff`), which is why that path never paid this. The guard still runs,
at build time, so the value is unchanged: the packed action still agrees with the
atomic reference at 7.9e-16, exactly as before. **That leaves the true cost of
Rhie-Chow in the element kernel at 1.31x, not 2.17x.**

The hoist is applied to the Jacobian action and deliberately not to the residual. The
residual's face loop is smaller, its cost model lands on the other side of the same
threshold, and its checksum is bit-identical either way -- so its vectorisation never
changed and it would pay the gather for nothing. Measured, it did: 1614 -> 1545
MDOF/s. With the residual left alone it is 1614 -> 1605, inside the spread.

What now dominates a matvec is the other half of Rhie-Chow: the direction's nodal
gradient reconstruction, which cannot be hoisted out of a Krylov solve because the
direction changes every iteration. At 1460 MDOF/s for the kernel and 602 for the
matvec it is **59% of every matvec** -- the largest single item on the flat path, and
the next thing worth attacking.

The residual discrepancy is 871 (bench) against 607-625 (solver), a factor of 1.4
rather than 3.1, and the two are no longer in different configurations -- they run
on different meshes (a plain cube against the constrained channel) and at different
sizes. That is an ordinary gap; the factor of three was not.

Two further things this settles:

  * Scaling is not the problem. With Rhie-Chow the sweep goes 7.53 -> 463.8 MDOF/s
    from 1 to 72 threads (61.6x); without it, 30.8 -> 1881 (61.1x). The Rhie-Chow
    path parallelises just as well. The cost is arithmetic and a second sweep.
  * The thread binding was never the explanation. OMP_PROC_BIND=true and close
    measure 1711 and 1718 MDOF/s, equal within a 3% spread; spread is the slow one
    at 1490.

Quoting ~2000 MDOF/s for this kernel still means the version without Rhie-Chow,
and the solver does not run that version. The number to quote for what the solver
runs is 602 MDOF/s at 11.2M dof on 72 Grace cores.

The configuration without Rhie-Chow -- every row in perf/baseline_grace.csv -- is
untouched by all of this, checked in the same runs: 1910 -> 1912 MDOF/s for the
Jacobian action and 2642 -> 2640 for the residual, with bit-identical checksums.

Reproduce with `jobs/fused_rc.sbatch` (the cascade) and `jobs/rc_hoist.sbatch` (the
hoist, against the binary from the parent commit).
