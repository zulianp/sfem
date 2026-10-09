# `tet4/affine/` — empty, because every TET4 kernel is already affine and cannot be otherwise

DESIGN.md asks each format for `affine` / `isoparametric` / `axis_aligned`. For the TET4
microkernels the separation is not work to be done but a property of the element: a
four-node tetrahedron's reference map is **linear**, so its Jacobian is constant over the
element by construction. There is no isoparametric TET4 to separate from an affine one — the
two coincide.

Checked rather than assumed: no kernel under `../` takes node coordinates in place of an
adjugate and a determinant, and neither the hand-written header nor the generated one contains
the string `isoparam`.

So the TET4 kernels stay in `../`, and this note is here instead of a directory holding a copy
of them. The same argument retires `isoparametric/` and `axis_aligned/` for this element: an
axis-aligned tetrahedron is not a meaningful category, since a tetrahedron has no axis-aligned
edge triple that is also a valid element.

The HEX8 case is different and that is why it has the folders: a trilinear hexahedron's Jacobian
varies over the element unless the element is a parallelepiped, so affine there is an assumption
about the mesh rather than a property of the element, and the isoparametric kernels exist to
carry the case where the assumption fails.
