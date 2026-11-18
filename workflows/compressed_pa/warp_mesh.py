"""A smesh mesh of the unit box (raw layout: meta.yaml, i<k>.int32, x/y/z.float32) displaced by a smooth
field that vanishes on the box's faces, for the compressed partial assembly paper's study of curved and
distorted elements:
    x <- x + a h s(x) (sin 2 pi y, sin 2 pi z, sin 2 pi x),   s(x) = sin(pi x) sin(pi y) sin(pi z),
h the mean length of the elements' first edges and a the severity.  With --edges only the nodes that are
no element's vertex move (a TET10 mesh: its edges curve, its vertices stay; the first four nodes of an
element are its vertices); otherwise every node moves (HEX8: distorted, non-affine hexahedra).  The faces
of the box stay planes, so sidesets found by plane are unchanged.  Prints the smallest ratio of an
element's signed vertex volume to the undisplaced one (TET4/TET10 and HEX8 corners).
    python warp_mesh.py <in> <out (new)> <a> [--edges]"""
import os
import shutil
import sys

import numpy as np


def main(src, out, a, edges):
    meta = open(os.path.join(src, "meta.yaml")).read()
    k = 0
    idx = []
    while os.path.exists(os.path.join(src, f"i{k}.int32")):
        idx.append(np.fromfile(os.path.join(src, f"i{k}.int32"), dtype=np.int32))
        k += 1
    e = np.stack(idx, axis=1)
    X = np.stack([np.fromfile(os.path.join(src, f"{c}.float32"), dtype=np.float32).astype(np.float64) for c in "xyz"], axis=1)
    lo, hi = X.min(axis=0), X.max(axis=0)
    U = (X - lo) / (hi - lo)
    h = float(np.mean(np.linalg.norm(X[e[:, 1]] - X[e[:, 0]], axis=1)))
    s = np.sin(np.pi * U[:, 0]) * np.sin(np.pi * U[:, 1]) * np.sin(np.pi * U[:, 2])
    D = a * h * s[:, None] * np.stack([np.sin(2 * np.pi * U[:, 1]), np.sin(2 * np.pi * U[:, 2]), np.sin(2 * np.pi * U[:, 0])], axis=1)
    if edges:
        moving = np.ones(len(X), dtype=bool)
        moving[np.unique(e[:, :4])] = False
        D[~moving] = 0
    Y = X + D

    def volumes(P):  # the signed volume of the first four vertices (a hexahedron's corner 0,1,3,4)
        c = [0, 1, 2, 3] if e.shape[1] in (4, 10) else [0, 1, 3, 4]
        p = [P[e[:, i]] for i in c]
        return np.einsum("ij,ij->i", p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0]))

    ratio = volumes(Y) / volumes(X)
    os.makedirs(out, exist_ok=False)
    for i in range(k):
        shutil.copyfile(os.path.join(src, f"i{i}.int32"), os.path.join(out, f"i{i}.int32"))
    for c, col in zip("xyz", Y.T):
        col.astype(np.float32).tofile(os.path.join(out, f"{c}.float32"))
    open(os.path.join(out, "meta.yaml"), "w").write(meta)
    print(f"{out}: a={a} h={h:.4g} moved {int(np.count_nonzero(np.any(D != 0, axis=1)))} of {len(X)} nodes, "
          f"largest displacement {np.abs(D).max() / h:.3f} h, smallest vertex volume ratio {ratio.min():.3f}")
    if ratio.min() <= 0:
        sys.exit("warp_mesh: an element inverts")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), "--edges" in sys.argv[4:])
