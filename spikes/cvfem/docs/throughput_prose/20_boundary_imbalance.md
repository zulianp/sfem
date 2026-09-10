## The boundary closure was bound by its own load imbalance

The flat operator closes its boundary control volumes in a second sweep after the
element kernel, and that sweep walked the WHOLE mesh -- gathering coordinates, fields
and, for the Jacobian, the direction, 88 doubles per element -- in order to do work on
the shell alone. At N=96 the shell is about 9% of the elements, so 91% of the gathers
bought nothing.

The obvious fix is to test the face mask and skip. It bought nothing at all, and that
is the interesting part:

| packed Jacobian action, boundary scope | N=48, 1,853,572 dof | N=96, 14,489,860 dof |
|---|---:|---:|
| as it was | 1059 us/call | 7331 us/call |
| skipping elements with no boundary face | 1070 us/call | 6212 us/call |
| iterating a compacted list of them | **180 us/call** | **714 us/call** |
| | **5.9x** | **10.3x** |

The skip removes ~91% of the work and leaves the wall time where it was, because the
boundary elements are the shell of the mesh: under `schedule(static)` they fall into a
few threads' chunks, and the pass is bound by whichever thread owns them, not by the
total. Compacting the elements into a list first restores the balance -- every thread
gets an equal share of the elements that do something -- and only then does the work
reduction show up as time.

The share of a matvec spent closing the boundary goes from 23.5% to 5.0% at N=48 and
from 22.2% to 2.7% at N=96. The element sweep and the gradient reconstruction are
unchanged in the same runs (13.13 -> 13.26 s and 10.05 -> 10.23 s at N=96), so nothing
was moved around.

The skip is exact rather than an approximation. `fmask` is read in exactly one place in
each of the three boundary kernels, the per-face inclusion test, so an element with no
boundary face contributes nothing -- which tests/cvfem_boundary_mask_test.cpp asserts
directly ("fmask 0 contributes nothing"). Where no sideset mask exists, and that is the
default since one is only compiled under SFEM_BOUNDARY_MASK=1, the list is built from
the same bounding-box test the kernels would have run face by face.

The benchmark could not have found this: its own `--boundary` pass already skipped
zero-mask elements, so it had the work reduction and the imbalance together and showed
neither. The measurement had to come from the solver's trace.

Reproduce with `jobs/bnd_skip.sbatch`.
