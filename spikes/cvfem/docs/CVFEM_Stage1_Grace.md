# What the reconstruction fix is worth to the solver, on Grace

Stage 1 rewrote the nodal-gradient reconstruction that the Jacobian action calls once per
matvec: the denominator is cached geometry rather than a fresh 8 MB allocation, the sweep
runs over packs with a ghost reduction rather than over the flat element table with atomics,
and the denominator is folded into the writes so the separate normalisation pass is gone.

This page is the end-to-end measurement of that, against the binary immediately before it
(commit `7e290fa06`), on one Grace socket. Both binaries are built from the same sources
except for five headers, with the same compiler and flags, and run on an exclusive node at
72 threads with `OMP_PROC_BIND=true` and `OMP_PLACES=cores`.

The comparable quantity throughout is **microseconds per call** from the compiled-in trace,
not wall clock. The two binaries reduce the reconstruction in different orders, so they take
different numbers of linear iterations on the same problem -- 1539 against 2930 on one case
-- and a wall clock divided by different iteration counts is not a comparison.

## The matvec, by scope

Manufactured solution, 1,098,500 dof, 72 threads. `apply_jacobian_action_accumulate` is the
full operator: the element sweep, the reconstruction and the boundary closure inside it.

| scope | before (us/call) | after (us/call) | speedup |
|---|---:|---:|---:|
| `nodal_grad_strided` | 969.2 | 137.8 | **7.03x** |
| `apply_jacobian_action_packed` | 988.9 | 1007.4 | 0.98x |
| `apply_boundary_scs_jacobian_action` | 95.8 | 97.1 | 0.99x |
| `copy_constrained_dofs` | 36.6 | 36.4 | 1.01x |
| `apply_jacobian_action_accumulate` | 2057.9 | 1239.7 | **1.66x** |
| `Function::apply` | 2137.7 | 1320.4 | **1.62x** |

The reconstruction fell from 45.9% of the matvec to 10.6%. Everything else is within 2%,
which is the check that the gain is the reconstruction rather than a difference in the run.

## Against problem size

The channel, 72 threads, one row per size. The element sweep saturates at about 1,065
MDOF/s by a million dofs, so the two larger sizes are the ones to quote and 242,500 is
below saturation.

| dof | reconstruction before | after | speedup | matvec before | after | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 242,500 | 261.1 | 69.2 | 3.77x | 757.8 | 573.0 | 1.32x |
| 1,082,564 | 897.8 | 146.2 | 6.14x | 2049.9 | 1303.5 | **1.57x** |
| 2,924,100 | 2344.5 | 512.9 | 4.57x | 5290.7 | 3503.9 | **1.51x** |

In MDOF/s the reconstruction goes 929 -> 3503, 1206 -> 7404, 1247 -> 5702. The before column
is flat across a twelvefold change in size, which is what an atomically reduced sweep that
is bound by its atomics looks like; the after column rises and then falls back at 2.9M dof,
where it is bandwidth bound instead.

The speedup the solver sees end to end is smaller and should be: `us_per_lin_it` moves 1.11x,
1.13x and 1.13x across the same three sizes, because it also carries the multigrid
preconditioner and the Krylov vector operations, and Stage 1 touched neither.

Measuring this below saturation understates it badly. At 143,748 dof -- the top of the
verification ladder -- the reconstruction is only 28% of the matvec before the change, and
the matvec speedup is 1.22x rather than 1.66x.

## Verification

The 22-case matrix passes 6 of 6 checks on both binaries, with the fitted rates identical to
four digits: u L2 2.330 (R^2 = 0.9987) and p L2 shifted 1.514 (R^2 = 0.9995) over 500,
2,916, 19,652 and 143,748 dof.

One case differs between them, and it is not a regression. The pressure port at p_bar = 3.0,
33,124 dof, is the far end of a sweep whose other seven values are between -0.16 and 1.5,
and its linear solve is fragile in both binaries:

| threads | 1 | 8 | 18 | 36 | 72 |
|---|---|---|---|---|---|
| after | 922 it | 789 it | diverged | 2225 it | 822 it |
| before | 779 it | 868 it | diverged | 816 it | 1066 it |

Both fail at 18 threads and both converge at 72. The same binary at 72 threads diverged in
one run of this case and converged in another, so the outcome is not reproducible run to
run: neither reduction order is bit-reproducible with itself, and this case sits on the
edge. Iteration counts on a case that does converge swing by a factor of three for the same
reason.
