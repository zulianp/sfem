# Comparing the inexact apply against the exact one

The projected apply of `plans/inexact_apply.py` is exact on an affine simplex,
because the deformation gradient does not vary over the cell.  That is the
gate on the whole construction, and this is it, run on real generated code
rather than symbolically.

```sh
python - <<'PY'
import dataclasses, sys
sys.path.insert(0, "python/codegen/framework/materials")
from sfem import gen
from linear_elasticity import material
gen.generate(dataclasses.replace(material, inexact_apply=True),
             "/tmp/ia", elements=("TET4",), clean=True)
PY

c++ -std=c++17 -O2 -o compare spikes/inexact_apply_compare/compare.cpp \
    /tmp/ia/d3/tet4/linear_elasticity_tet4_operator.cpp \
    -I /tmp/ia/d3/tet4 -I /tmp/ia -I /tmp/ia/d3 \
    -I base -I algebra -I <build> -I <build>/external/smesh \
    $(find external/smesh/src -type d | sed 's/^/-I /')
./compare
```

Measured on a Freudenthal-split unit cube, 1296 elements, 343 nodes, 1029
degrees of freedom, single-threaded:

```
exact   l1 = 1.334688845028226e+01
inexact l1 = 1.334688845028226e+01
relative l1 difference = 3.174e-16
```

Round-off, which is what "the projection is exact here" has to mean in
floating point.  `linear_elasticity` is used because its tangent does not
depend on the state, so the exact kernel takes no state argument and the
comparison passes zero for it; the inexact kernel builds the same constant
tangent from that zero state.

The comparison this does *not* make is the interesting one for a curved
element: on HEX8 or TET10 the projection is a genuine approximation and the
right measurement is a relative error against the exact apply, not agreement.
That needs the kernel emitted for those elements and a driver that sweeps mesh
resolution, and it is not done.

## Building

CMake, from the spike directory. It finds MPI and OpenMP, generates the kernels for
the material and element you ask for, and builds whichever harnesses those kernels
support.

    cmake -S spikes/inexact_apply_compare -B build-spike \
          -DSPIKE_MATERIAL=neohookean_ogden -DSPIKE_ELEMENT=HEX8
    cmake --build build-spike -j
    ./build-spike/bench_split 5

The cache variables are `SPIKE_MATERIAL`, `SPIKE_ELEMENT` (TET4, HEX8, TET10),
`SPIKE_MESH_ORDER`, `SPIKE_PACK_SIZE`, `SPIKE_ARCH_FLAGS`, and the paths
`SPIKE_SFEM`, `SPIKE_SFEM_BUILD` and `SPIKE_GEN_ROOT`. On Apple silicon pass
`-DSPIKE_ARCH_FLAGS=-mcpu=apple-m1` and `-DOpenMP_ROOT=$(brew --prefix libomp)`.

Which harnesses appear depends on what the material publishes: a single-unit material
gets `bench_split`, `warp_sweep` and `scale_probe`, and the two-unit Mooney-Rivlin
Kelvin-Voigt material gets `bench_mixed`. `bench_op_split` and `op_inexact_check` go
through the Op wrappers rather than the kernels and appear only when `libsfem` is found
in `SPIKE_SFEM_BUILD`.

**Kernels are generated while CMake configures, not while it builds.** The set of
generated sources is not known until the generator has run, and CMake resolves source
lists at configure time; generating during the build would need a second configure to
see the results. Reconfigure after changing the material or the element. Generation is
skipped when the output is already in `SPIKE_GEN_ROOT`, which matters -- it is 75
seconds for TET4 and 1380 for a Mooney-Rivlin HEX8.

The pack size is a cache variable *and* an environment variable: `PACK_SIZE=256
./build-spike/bench_mixed 5` re-partitions without rebuilding, which is the only
practical way to sweep it when the translation unit takes forty minutes to compile.

Where the material publishes a `matrix_formats` entry the harnesses also assemble a BSR
matrix and time its apply beside the matrix-free and partially-assembled ones, so the
three ways of applying the same Jacobian appear in one table with their setup costs and
their memory. The graph is the mesh's node-to-node graph, built in `bsr_matrix.inc`
rather than borrowed from the library, because the comparison is between kernels and not
between two surrounding frameworks.

The `run_*.sh` scripts predate this and still work, but they are behind: they hand-write
the same compiler invocation, the copies had drifted apart -- one knew about the packed
layout and another did not, one found `mpi.h` and another did not -- and none of them
carries the BSR column. Prefer CMake.

## The split kernels

`bench_split.cpp` / `run_split.sh` measure the form that pays: `Sbar` assembled
once into a store, then applied. The generated header carries three entry points
per element —

    <material>_<element>_inexact_apply_tangent_a_msoa
        geometry, state and parameters in, `Sbar` out.  Once per tangent.

    <material>_<element>_inexact_apply_stored_a_msoa
    <material>_<element>_inexact_apply_compressed_a_msoa
        `Sbar` and the vector.  No geometry, no state, no material parameters.

`Sbar` is 45 numbers per element in three dimensions, the same packing the
hand-written `*_S_IKMN_SIZE` operators use, under the major symmetry
`S[i,k,m,n] == S[k,i,n,m]`. The store's type is a template parameter, so
`double`, `metric_tensor_t` (float) and `compressed_t` (half, with one
`scaling_t` per element) are the same kernel. Its two strides let the caller
choose element-major or component-major without a second kernel.

`warp_sweep.cpp` / `run_warp.sh` characterise the approximation instead of its
speed: the displacement is warped away from a constant-gradient one by a severity
parameter, and the deviation is measured against it.  At zero warp the tangent is
constant and the projection is exact, which makes that row a control on the node
ordering and the reference integral as well.

See `RESULTS.md` for the measured throughput, the break-even, the warp sweep, and
why a smooth increment makes a converging projection look flat.
