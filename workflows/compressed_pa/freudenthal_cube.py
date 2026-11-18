"""The paper's tetrahedral benchmark cube: the unit cube of n^3 cells, each split into six tetrahedra
along its main diagonal (Freudenthal/Kuhn), positively oriented, in smesh's raw layout (i0..i3.int32,
x/y/z.float32).  The library's own cube (Mesh::create_cube) has twelve tetrahedra and a centre node per
cell, another matrix; workflows/compressed_pa/throughput.sbatch runs bench_inexact MESH on this one.
    python freudenthal_cube.py <n> <out folder (new)>"""
import os
import sys
from itertools import permutations

import numpy as np


def main(n, out):
    m = n + 1
    g = np.arange(n)
    cx, cy, cz = np.meshgrid(g, g, g, indexing="ij")
    base = (cx + m * cy + m * m * cz).ravel().astype(np.int64)
    step = [1, m, m * m]  # unit steps along x, y, z
    tets = []
    for p in permutations(range(3)):
        a = base
        b = a + step[p[0]]
        c = b + step[p[1]]
        d = c + step[p[2]]
        positive = np.linalg.det(np.eye(3)[list(p)]) > 0
        tets.append(np.stack([a, b, c, d] if positive else [a, c, b, d], axis=1))
    e = np.concatenate(tets).astype(np.int32)
    os.makedirs(out, exist_ok=False)
    for i in range(4):
        e[:, i].tofile(os.path.join(out, f"i{i}.int32"))
    h = np.linspace(0, 1, m, dtype=np.float32)
    X, Y, Z = np.meshgrid(h, h, h, indexing="ij")
    for c, A in zip("xyz", (X, Y, Z)):  # node x + m y + m^2 z
        A.transpose(2, 1, 0).ravel().astype(np.float32).tofile(os.path.join(out, f"{c}.float32"))
    print(f"{out}: {len(e)} elements, {m ** 3} nodes")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
