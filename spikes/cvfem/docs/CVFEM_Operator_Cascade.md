# What the benchmark leaves out

The throughput this spike quotes is measured on an operator the solver never runs. That is
a deliberate and useful thing to measure — it isolates the element kernel from everything
around it — but it is not the cost of an apply in the Newton loop, and the gap is larger
than the prose suggested.

`src/hex8/cvfem_hex8_ns_core.hpp:5` says the two families "differ in physics, not just in
layout: the benchmark's carry no boundary sub-control-surface or Rhie-Chow terms". This
puts numbers on that.

## Measurement

`drivers/cvfem_hex8_ns_upwind_bench.cpp` with `--rhie-chow` and `--boundary`, sumfact
kernel, `--n 128` → **8,586,756 dofs**, one Grace socket, 72 cores, `--exclusive`,
`OMP_PROC_BIND=close OMP_PLACES=cores`, medians of 3, 20 timed applies after 3 warmups.

## Packed layout

| configuration | MDOF/s | of kernel |
|---|---|---|
| element kernel only — what the gate and every quoted figure measure | 2578.8 | 100% |
| + boundary closure | 1999.6 | 78% |
| + Rhie–Chow, nodal gradient hoisted | 1636.6 | 63% |
| + Rhie–Chow + boundary — **the operator the solver runs** | 1376.9 | **53%** |
| + Rhie–Chow with the gradient rebuilt inside every apply | 719.9 | 28% |

## Atomic layout

| configuration | MDOF/s | of kernel |
|---|---|---|
| element kernel only | 842.3 | 100% |
| + boundary closure | 749.5 | 89% |
| + Rhie–Chow, gradient hoisted | 693.7 | 82% |
| + Rhie–Chow + boundary | 633.8 | 75% |
| + Rhie–Chow, gradient per apply | 437.2 | 52% |

## Three things worth taking from this

**The headline number overstates the real operator by about 1.9x.** 2579 MDOF/s is the
element kernel; the same kernel with the physics the solver needs is 1377. Quoting the
first as the operator's throughput is not wrong, but it needs the qualifier, and this table
is what the qualifier should point at.

**Caching the nodal pressure gradient is worth 2.27x on the apply** (1636.6 against 719.9).
That is the apply alone; `docs/README_alps.md` records the same option as 1.26x off the
whole linear solve, which is consistent — the solve is more than its applies. Anything that
forces the gradient to be rebuilt per apply gives back more than half the operator.

**The packed layout's advantage shrinks as the physics is added.** Against atomic it is
3.06x on the bare kernel, 2.17x with Rhie–Chow and the boundary, and 1.65x once the
gradient is rebuilt per apply. The packed layout is still the right choice, but a layout
comparison made on the bare kernel overstates the margin by nearly two.

## Reproducing

```sh
source scripts/alps_env.sh
cvfem_build --target cvfem_hex8_ns_upwind_bench
sbatch jobs/perf_regression.sbatch     # the gate now carries three of these configurations
```

The three `residual_packed_rc*` entries in `scripts/perf_regression.sh` gate exactly the
rows above, so a change that makes the real operator slower without touching the bare
kernel is now visible.
