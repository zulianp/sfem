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
