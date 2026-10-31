"""A TET4 mesh of an application for the benches (QP_MESH): the blocks a case computes, from its
smesh database, in the raw layout element_mesh.inc reads (i0..i3.int32, x/y/z.float32).

    python real_meshes.py <smesh mesh folder> <output folder> [--blocks A,B] [--refine k]

The nodes the blocks use are numbered compactly in their original order (the benches reorder them).
The coordinates are translated to the origin and scaled so that the largest extent is 1, so that the
benches' state (a linear map plus sines of the coordinates, written for the unit cube) deforms the
part as much as it deforms the cube; a uniform scaling multiplies the tangent by a constant and
changes no relative error.  --refine splits each tetrahedron into eight, k times (the inner
octahedron along its shortest diagonal).  Prints the mesh's size and its blocks per matrix row.
"""
import argparse
import os

import numpy as np
import yaml


def read(folder, blocks):
    with open(os.path.join(folder, "meta.yaml")) as f:
        meta = yaml.safe_load(f)
    xyz = [np.fromfile(os.path.join(folder, meta["points"][d][c]), dtype=np.float64)
           for d, c in enumerate("xyz")]
    names = [b["name"] for b in meta["blocks"]]
    chosen = blocks or names
    missing = set(chosen) - set(names)
    if missing:
        raise SystemExit(f"no block {sorted(missing)} in {folder} (blocks: {names})")
    elems = []
    for b in meta["blocks"]:
        if b["name"] not in chosen:
            continue
        if b["element_type"] != "TET4":
            raise SystemExit(f"block {b['name']} is {b['element_type']}, not TET4")
        cols = [np.fromfile(os.path.join(folder, list(e.values())[0]), dtype=np.int32) for e in b["elements"]]
        elems.append(np.stack(cols, axis=1))
    return np.concatenate(elems).astype(np.int64), np.stack(xyz, axis=1), chosen


def compact(elems, points):
    used, inverse = np.unique(elems, return_inverse=True)
    return inverse.reshape(elems.shape), points[used]


def volumes(elems, points):
    a, b, c, d = (points[elems[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6


def refine(elems, points):
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = np.sort(np.concatenate([elems[:, [i, j]] for i, j in pairs]), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    mid = len(points) + inverse.reshape(len(pairs), len(elems)).T  # [element][edge]
    points = np.concatenate([points, 0.5 * (points[unique[:, 0]] + points[unique[:, 1]])])
    v = [elems[:, i] for i in range(4)]
    m01, m02, m03, m12, m13, m23 = (mid[:, k] for k in range(6))
    corners = [(v[0], m01, m02, m03), (m01, v[1], m12, m13), (m02, m12, v[2], m23), (m03, m13, m23, v[3])]
    # The inner octahedron, split along a diagonal: four tetrahedra around it, the other four
    # midpoints taken in order around the diagonal's equator.
    diagonals = [((m01, m23), (m02, m03, m13, m12)), ((m02, m13), (m01, m03, m23, m12)),
                 ((m03, m12), (m01, m02, m23, m13))]
    length = np.stack([np.linalg.norm(points[p] - points[q], axis=1) for (p, q), _ in diagonals], axis=1)
    pick = np.argmin(length, axis=1)
    inner = []
    for t in range(4):
        tet = []
        for c in range(4):
            options = [((p, q) + (ring[t], ring[(t + 1) % 4]))[c] for (p, q), ring in diagonals]
            tet.append(np.choose(pick, options))
        inner.append(tuple(tet))
    children = np.concatenate([np.stack(t, axis=1) for t in corners + inner])
    return children, points


def blocks_per_row(elems, n):
    pairs = np.stack([elems[:, [i, j]] for i in range(4) for j in range(4)]).reshape(-1, 2)
    key = np.unique(pairs[:, 0] * n + pairs[:, 1])
    return len(key) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--blocks", default="")
    ap.add_argument("--refine", type=int, default=0)
    a = ap.parse_args()
    elems, points, chosen = read(a.mesh, [b for b in a.blocks.split(",") if b])
    elems, points = compact(elems, points)
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0)).max()
    vol = np.abs(volumes(elems, points)).sum()
    for _ in range(a.refine):
        elems, points = refine(elems, points)
    vol_refined = np.abs(volumes(elems, points)).sum()
    if abs(vol_refined - vol) > 1e-9 * vol:
        raise SystemExit(f"refinement changed the volume: {vol} -> {vol_refined}")
    os.makedirs(a.out, exist_ok=True)
    for i in range(4):
        elems[:, i].astype(np.int32).tofile(os.path.join(a.out, f"i{i}.int32"))
    for d, c in enumerate("xyz"):
        points[:, d].astype(np.float32).tofile(os.path.join(a.out, f"{c}.float32"))
    n = len(points)
    print(f"{a.out}: blocks {','.join(chosen)}, refine {a.refine}: {len(elems)} elements, {n} nodes, {3 * n} dof, "
          f"{blocks_per_row(elems, n):.1f} blocks per row")


if __name__ == "__main__":
    main()
