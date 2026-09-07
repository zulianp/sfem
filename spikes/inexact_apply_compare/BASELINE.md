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
