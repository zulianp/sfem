## The packed layout's advantage shrinks as the physics is added

Against the atomic layout at 4,121,204 dof:

| operator | packed | atomic | packed / atomic |
|---|---:|---:|---:|
| residual, element kernel only | 2912 | 826 | **3.53x** |
| + Rhie–Chow, gradient hoisted | 1656 | 618 | 2.68x |
| + Rhie–Chow + boundary | 1490 | 624 | 2.39x |
| Jacobian action, element kernel only | 1837 | 785 | 2.34x |
| + Rhie–Chow + boundary | 888 | 453 | 1.96x |

The packed layout is still the right choice — it wins in every row — but a layout
comparison made on the bare kernel overstates the margin by roughly 1.5 to 2. The reason is
structural rather than incidental: packing buys its advantage in the element sweep, through
SIMD over a pack and a ghost reduction in place of atomics, and most of what is added here
is either a separate pass over the mesh or arithmetic that vectorises less well.

One row deserves a caveat rather than a reading. With the gradient rebuilt inside every
apply the ratio reads 2.99x, *higher* than the bare kernel's — but that is an artefact of
where the reconstruction runs. It now sweeps over packs, and the benchmark builds no pack
for a plain `--layout atomic` residual, so the atomic side of that particular row is still
paying for the old flat atomic sweep while the packed side is not. The solver has a pack in
both cases and would not show the same gap.

The same caution applies to any two rows in the tables above: they are comparable only when
the completeness column matches, and only when the gap between them exceeds the `spread`
each was measured with.
