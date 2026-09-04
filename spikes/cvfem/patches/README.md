# Patches for external/smesh

Not applied. `external/smesh` is a separate checkout, so these are left for manual
application.

## 0001-sshex8-deterministic-coordinates.patch

Makes the semi-structured HEX8 coordinate generation reproducible.

**Apply from the smesh root:**

```sh
cd external/smesh
git apply -p1 ../../spikes/cvfem/patches/0001-sshex8-deterministic-coordinates.patch
# or: patch -p1 < ../../spikes/cvfem/patches/0001-sshex8-deterministic-coordinates.patch
```

Then rebuild smesh, and rebuild anything linking it -- the spike links a prebuilt smesh,
so the fills come from the library and the patch has no effect until that is rebuilt.

**What is wrong.** `sshex8_fill_points` and `sshex8_fill_points_1D_map` write

```c
points[d][elements[lidx][e]] = acc;
```

for every macro-element `e` that contains the node. A lattice node on a macro-element
face, edge or corner belongs to several elements, and each computes `acc` by
interpolating from its own eight corners with its own weights. The value is the same
mathematically and differs in the last bits, so the coordinate that survives is whichever
store landed last. The loop is `#pragma omp parallel for collapse(4)` over the lattice
with the element loop inside, and a shared node has a different `lidx` in each element
that owns it, so the competing writes come from different parallel iterations. Which one
wins follows the thread schedule.

**How it shows up.** On a 27-macro-element mesh the x coordinate checksum varied between
runs on 8 threads (16562.000025525689, 16562.000030174851) and was stable on one
(16562.000023022294). y and z were unaffected, but only because their values landed on
exactly representable numbers where summation order cannot matter. Everything downstream
inherits it: a field seeded from the coordinates differs before any solver has run, and in
the CVFEM spike this was the largest remaining source of run-to-run variation in iteration
counts.

**The fix.** Give each node a single writer: the lowest-numbered element containing it.
`sshex8_build_node_owner` builds that table with a minimum, which is order independent, so
the table is itself deterministic. Each fill then skips nodes it does not own.

Three properties worth noting:

- A single-threaded run produces exactly what it produced before, since the lowest element
  was already the last to write a node it shares only with higher-numbered elements. For
  multi-threaded runs the values change in the last bits -- that is the point.
- It is less work, not more: a shared node is interpolated once rather than once per
  incident element, which at level 4 is 37% of the nodes and at level 16 about 12%.
- The extra cost is one pass over (elements x nodes-per-element) and one `ptrdiff_t` per
  node, paid once at mesh construction.

**Checked:** `c++ -fsyntax-only` on `smesh_sshex8_mesh.cpp`, which explicitly instantiates
both fills for every (idx, geom, ref) combination the library uses.

**Not checked:** the runtime effect, because that needs smesh rebuilt. The determinism
probe is `SFEM_GMG_CHECK=1` in `cvfem_hex8_ns_ssgmg`, which prints `mesh coords checksum`;
run it twice on 8 threads at `SFEM_N=3` and the three numbers should now agree.
