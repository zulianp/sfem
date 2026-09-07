# The library's own dof rates, as the baseline to beat

Measured with SFEM's `bench_op`, which is the right instrument for this: it times
`Op::apply` through the operator layer a solve actually goes through, and it
already registers the generated kernels next to the hand-written ones.  Its rate
is `rows / elapsed` -- degrees of freedom per second -- which is the same
normalisation `bench_split.cpp` and `bench_mixed.cpp` use, so the numbers are
directly comparable once the build and thread count match.

All rows below: `build64`, `OMP_NUM_THREADS=8`, `OMP_PROC_BIND=true`, Apple
M-series laptop, `SFEM_REPEAT=5`, matrix ops suppressed with
`SFEM_MAX_NODES_MATRIX=1 SFEM_MAX_NODES_BLOCK_MATRIX=1` so the comparison is
matrix-free throughout.  All figures MDOF/s.

| element | ndof | elements | Laplacian | PackedLaplacian | LinearElasticity | NeoHookeanOgden | NeoHookeanOgdenPacked | GeneratedNeoHookeanOgden |
|---|---|---|---|---|---|---|---|---|
| TET4  | 10498683 | 20736000 | 165.95 | 144.24 | 130.55 |  30.51 |     -- | 45.16 |
| TET10 | 10498683 |  2592000 | 150.40 | 200.86 |  98.88 | 121.72 | 133.09 | 22.72 |
| HEX8  | 10328853 |  3375000 | 136.43 | 179.43 | 143.76 | 142.04 | 154.90 | 44.36 |

    TET4   SFEM_ELEM_TYPE=TET4  SFEM_BASE_RESOLUTION=120
    TET10  SFEM_ELEM_TYPE=TET4  SFEM_BASE_RESOLUTION=60  SFEM_PROMOTE_TO_P2=1
    HEX8   SFEM_ELEM_TYPE=HEX8  SFEM_BASE_RESOLUTION=150

## What it says about the generated kernels

| element | best hand-written | generated | generated / hand-written |
|---|---|---|---|
| TET4  |  30.51 | 45.16 | **1.48x ahead** |
| TET10 | 133.09 | 22.72 | **5.86x behind** |
| HEX8  | 154.90 | 44.36 | **3.49x behind** |

The setup column of the raw output explains the whole pattern.  The hand-written
neohookean reports 0.70 s of setup on TET10 and 0.395 s on HEX8, and none at all
on TET4.  That setup is partial assembly: `frontend/ops/sfem_NeoHookeanOgden.cpp`
turns it on automatically for HEX8 and TET10, stores `S_ikmn` per element, and
applies from the store.  TET4 does not use it.

So the generated kernels are ahead exactly where the library does not use the
technique, and behind exactly where it does.  The gap is not a code-quality gap;
it is the technique.  The object the hand-written path stores is the same
45-component `S_ikmn` under the same major symmetry that
`plans/inexact_apply.py` now emits, so closing the gap is what the split kernels
are for, and these are the numbers they have to beat:

    TET10   133.09 MDOF/s
    HEX8    154.90 MDOF/s

and on TET4, 45.16 MDOF/s is already the number to preserve rather than beat.

## Two things worth not over-reading

**Packed is not uniformly the faster layout.**  It leads on TET10 (200.86
against 150.40) and HEX8 (179.43 against 136.43), and on TET4 at 3.13M dof
(170.42 against 122.65).  But at 10.5M dof on TET4 it *trails* -- 144.24 against
165.95 -- so its advantage there is size-dependent, presumably cache residency.
Quote it per element and per size, not as a general property.

**Do not mix builds.**  An earlier pass at this used the non-OpenMP `build/`,
which reported the HEX8 hand-written neohookean at 5.84 MDOF/s with 9.39 s of
setup against `build64`'s 142.04 and 0.395 s.  Threads alone do not explain a
24x change in setup; the builds differ in more than OpenMP.  The single-threaded
figures were discarded rather than mixed into the table above.

## Comparing the split kernels against this

The split benchmarks in this directory currently call the generated kernels
directly, with the geometry precomputed, rather than through `Op::apply`.  That
measures a different thing -- at 170373 dof `bench_op` reports the generated
neohookean at 1.463 MDOF/s where the direct call reports about 5 -- so figures
from the two harnesses must not be quoted against each other.  A fair comparison
needs the split registered as an op and driven by `bench_op`; until then this
file is the reference, not a scoreboard.

One consequence of the scale is worth stating in advance.  At 20.7M TET4
elements the stored tangent is 45 numbers per element: 3.7 GB at f32 and 1.9 GB
at f16, against the 3.4 GB the whole benchmark currently resides in.  For the
two-unit Mooney-Rivlin Kelvin-Voigt material it is 126 numbers, so 10.4 GB at
f32.  At this size the store is a first-order cost, which the 206763-dof
measurements could not show.

# The generated split against the baseline

`bench_op_split.exe.cpp` / `run_bench_op_split.sh`.  A spike copy of `bench_op`,
deliberately outside the driver tree so the baseline above cannot drift: it
builds the same mesh from the same environment variables, times the library's
operators through `Op::apply` exactly as `bench_op` does, and then times the
generated split kernels on that same mesh in the same process -- which removes
the harness mismatch that made the earlier `spikes/` figures incomparable to
these.

It separates the two costs `bench_op` folds together.  The tangent is assembled
once per Newton step and applied once per Krylov iteration, so an apply rate
without its assembly is half the story.

Both at ~10.4M dof, 8 threads, `build64` libraries, MDOF/s.

**TET10, 2592000 elements, 10498683 dof**

| | setup [s] | apply [s] | rate |
|---|---|---|---|
| library `NeoHookeanOgden`, hand-written partial assembly | 0.6525 | 7.643e-02 | 137.37 |
| library `GeneratedNeoHookeanOgden`, fused                | --     | 3.854e-01 |  27.24 |
| generated split, assembly                                 | 0.3922 | --        |  --    |
| generated split, stored apply f64                         | 0.3922 | 4.927e-02 | 213.09 |
| generated split, stored apply f32                         | 0.3922 | 4.781e-02 | 219.61 |
| generated split, stored apply f16                         | 0.3922 | 4.865e-02 | 215.80 |

**HEX8, 3375000 elements, 10328853 dof**

| | setup [s] | apply [s] | rate |
|---|---|---|---|
| library `NeoHookeanOgden`, hand-written partial assembly | 0.4273 | 6.341e-02 | 162.90 |
| library `GeneratedNeoHookeanOgden`, fused                | --     | 2.174e-01 |  47.52 |
| generated split, assembly                                 | 0.4143 | --        |  --    |
| generated split, stored apply f32                         | 0.4143 | 1.031e-01 | 100.21 |

**TET4, 768000 elements, 398763 dof** (small: see the memory note below)

| | setup [s] | apply [s] | rate |
|---|---|---|---|
| library `NeoHookeanOgden`, no partial assembly           | --     | 1.187e-02 |  33.59 |
| library `GeneratedNeoHookeanOgden`, fused                | --     | 8.319e-03 |  47.93 |
| generated split, assembly                                 | 0.0129 | --        |  --    |
| generated split, stored apply f16                         | 0.0129 | 3.410e-03 | 116.95 |

## What it says

**TET10: the generated split beats hand-written partial assembly on both halves.**
The apply is 1.60x faster (219.61 against 137.37) and the assembly is 1.66x
cheaper (0.392 s against 0.653 s).  The baseline had the generated kernel 5.86x
behind on this element; the split turns that deficit into a lead.

**HEX8: the split doubles the generated kernel but stays 1.63x behind the
hand-written one** (100.21 against 162.90, from 47.52).  The assembly times are
level -- 0.414 s against 0.427 s -- so the whole remaining gap is the
contraction: `hex8_SdotHdotG` beats the staged rank-factored contraction, which
is expensive precisely on HEX8, where `Wbar` has high rank.  That is the next
thing to fix, and it is a contraction problem rather than a partial-assembly one.

**TET4: 2.44x over the fused kernel**, on the element where the library does not
use partial assembly at all.  There is nothing hand-written to catch here; this
is new throughput.

## Two things to read carefully

**Break-even needs the right reference.**  It only means something against an
operator that performs no setup of its own.  The hand-written `NeoHookeanOgden`
enables partial assembly for HEX8 and TET10 and pays for it in its own setup
column, so charging the split for an assembly the reference also performs would
be wrong -- the first version of this spike did exactly that and reported 15.1
applies for TET10 where the honest figure against the fused kernel is 1.16.  The
tool now prints both framings and names which reference each applies to.  Against
an operator that also assembles, compare apply to apply and setup to setup.

**The store is a first-order cost at this scale.**  The bench holds f64, f32 and
f16 stores at once, 630 bytes per element.  That is 1.6 GB on TET10 and 2.1 GB on
HEX8 at the sizes above, but 13 GB for TET4 at the baseline resolution of 120,
which is why the TET4 row is measured at 768000 elements rather than 20.7M.
Reaching the full baseline size on TET4 needs the precisions allocated and freed
one at a time.  In a solver only one precision is ever resident, so this is a
property of the benchmark; the underlying per-element cost -- 45 numbers, 180
bytes at f32 -- is what a real use pays.
