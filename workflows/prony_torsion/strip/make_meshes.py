#!/usr/bin/env python3
"""The strip's meshes (cases/epdm_strip_turns.yaml): a 100 x 30 x 6 mm box, x along the strip.

    make_meshes.py <out>    writes <out>/<element>_<n>/ for every mesh below

n is the number of elements along the strip; the section has 0.3 n by 0.06 n of them, so the
elements stay near cubes (1.67 x 1.67 x 1.5 mm at n = 60). HEX27 at n has the nodes of HEX8 at 2n;
TET10 is the TET4 box promoted by the case (mesh.promote_to_p2), with the nodes of HEX8 at 2n.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "verification_and_validation"))
from common.geometry import box_mesh, tensor_product_mesh  # noqa: E402
from common.mesh import write_mesh  # noqa: E402

L, W, T = 0.1, 0.03, 0.006
MESHES = [("HEX8", 60), ("HEX8", 120), ("HEX27", 30), ("HEX27", 60), ("TET4", 60)]


def main(out):
    for element, n in MESHES:
        nx, ny, nz = n, 3 * n // 10, n // 15
        mesh = (tensor_product_mesh(L, W, T, nx, ny, nz, element) if element == "HEX27"
                else box_mesh(L, W, T, nx, ny, nz, element))
        write_mesh(os.path.join(out, "%s_%d" % (element, n)), mesh)
        print("%s_%d: %d x %d x %d elements, %d nodes" % (element, n, nx, ny, nz, len(mesh.points)))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
