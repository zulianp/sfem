## The packed layout's advantage shrinks as the physics is added

Against the atomic layout, on the residual at 4,121,204 dof:

| operator | packed | atomic | packed / atomic |
|---|---:|---:|---:|
| element kernel only | 2945 | 881 | **3.34x** |
| + Rhie–Chow, gradient hoisted | 1759 | 703 | 2.50x |
| + Rhie–Chow + boundary | 1559 | 675 | 2.31x |
| + Rhie–Chow, gradient per apply | 729 | 439 | 1.66x |

and on the Jacobian action, 2.47x on the bare kernel (2066 against 835) against 1.54x with
Rhie–Chow and the boundary closure (579 against 376).

The packed layout is still the right choice — it wins in every row — but a layout
comparison made on the bare kernel overstates the margin by about two. The reason is
structural rather than incidental: packing buys its advantage in the element sweep, through
SIMD over a pack and a ghost reduction in place of atomics, and every term added here is
either a separate pass over the mesh (the boundary closure, the transient term, the
gradient reconstruction) or arithmetic that does not vectorise as well (the Rhie–Chow
coefficient). None of those is helped by the pack, so each one dilutes what the pack is
for.

The same caution applies to any two rows in the tables above: they are comparable only when
the completeness column matches, and only when the gap between them exceeds the `spread`
each was measured with.
