#!/usr/bin/env python3
"""Compare two SDF grids where a containment solve evaluates them.

usage: sdf_compare.py <grid_a> <grid_b> <mesh> [--displacement=<raw mesh with point_data/disp_*>]
                      [--blocks=A,B] [--shift=s]

  <grid_a>, <grid_b>  grid folders (meta.yaml + data) or legacy metadata_*.yml files
  <mesh>              raw mesh (exodusII_to_raw.py layout); its skin nodes over the selected blocks
                      are the points sampled
  --displacement      also sample at X + u of these nodal displacements (e.g. utopia's result)
  --blocks            only these blocks of <mesh>
  --shift             the gap offset the solve uses (utopia: 1e-5)

Reports, per point set: how many points lie inside each grid, the sign agreement of a and b,
percentiles of |a - b| and of a and b themselves. For containment the final positions must be on
the free side of the obstacle, which fixes the sign convention.
"""

import getopt
import os
import sys

import numpy as np


def read_yaml_flat(path):
    out = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


class Grid:
    def __init__(self, location):
        meta_path = location if os.path.isfile(location) else os.path.join(location, "meta.yaml")
        m = read_yaml_flat(meta_path)
        self.n = np.array([int(m["nx"]), int(m["ny"]), int(m["nz"])])
        self.o = np.array([float(m["ox"]), float(m["oy"]), float(m["oz"])])
        self.h = np.array([float(m["dx"]), float(m["dy"]), float(m["dz"])])
        data = m["path"]
        if not os.path.isabs(data):
            data = os.path.join(os.path.dirname(meta_path), data)
        dtype = np.float64 if m.get("type", "float") in ("double", "float64") or data.endswith("float64") else np.float32
        self.v = np.fromfile(data, dtype=dtype).reshape(self.n[2], self.n[1], self.n[0])
        self.name = location

    def sample(self, p):
        """Trilinear values at points p (3, n); NaN outside the grid."""
        t = (p.T - self.o) / self.h
        inside = np.all((t >= 0) & (t <= self.n - 1), axis=1)
        i = np.clip(np.floor(t).astype(np.int64), 0, self.n - 2)
        f = t - i
        out = np.zeros(p.shape[1])
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) *
                         (f[:, 2] if dz else 1 - f[:, 2]))
                    out += w * self.v[i[:, 2] + dz, i[:, 1] + dy, i[:, 0] + dx]
        out[~inside] = np.nan
        return out


def load_mesh(folder, blocks):
    coords = []
    for ax in "xyz":
        f = [e for e in os.listdir(folder) if e.split(".")[0] == ax][0]
        dtype = np.dtype(f.split(".")[-1]) if f.split(".")[-1] != "raw" else np.float32
        coords.append(np.fromfile(os.path.join(folder, f), dtype=dtype).astype(np.float64))
    names = sorted(os.listdir(os.path.join(folder, "blocks")))
    if blocks:
        missing = set(blocks) - set(names)
        if missing:
            sys.exit(f"sdf_compare: {folder} has no blocks {sorted(missing)} (has {names})")
        names = [n for n in names if n in blocks]
    conn = []
    for n in names:
        d = os.path.join(folder, "blocks", n)
        conn.append(np.vstack([np.fromfile(os.path.join(d, f"i{k}.int32"), dtype=np.int32) for k in range(4)]))
    return np.vstack(coords), np.hstack(conn)


def skin_nodes(tets):
    """Nodes on faces used by exactly one tetrahedron."""
    faces = np.hstack([tets[[0, 1, 2]], tets[[0, 1, 3]], tets[[0, 2, 3]], tets[[1, 2, 3]]])
    faces = np.sort(faces, axis=0).T
    _, idx, counts = np.unique(faces, axis=0, return_index=True, return_counts=True)
    return np.unique(faces[idx[counts == 1]])


def report(label, a, b, shift):
    ok = ~np.isnan(a) & ~np.isnan(b)
    n = a.size
    print(f"{label}: {n} points, inside a {np.sum(~np.isnan(a))}, inside b {np.sum(~np.isnan(b))}")
    if not ok.any():
        return
    aa, bb = a[ok], b[ok]
    agree = np.sign(aa) == np.sign(bb)
    flipped = np.sign(aa) == -np.sign(bb)
    print(f"  sign: same {agree.mean():.4f}, opposite {flipped.mean():.4f}")
    q = [0, 1, 50, 99, 100]
    print(f"  |a - b| percentiles {q}: {np.percentile(np.abs(aa - bb), q)}")
    print(f"  |a + b| percentiles {q}: {np.percentile(np.abs(aa + bb), q)}")
    print(f"  a percentiles {q}: {np.percentile(aa, q)}")
    print(f"  b percentiles {q}: {np.percentile(bb, q)}")
    if shift:
        print(f"  a - shift < 0 at {np.mean(aa - shift < 0):.4f}, b - shift < 0 at {np.mean(bb - shift < 0):.4f}")


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        sys.exit(1)
    opts, _ = getopt.getopt(argv[4:], "", ["displacement=", "blocks=", "shift="])
    displacement, blocks, shift = None, None, 0.0
    for o, v in opts:
        if o == "--displacement":
            displacement = v
        elif o == "--blocks":
            blocks = v.split(",")
        elif o == "--shift":
            shift = float(v)
    a, b = Grid(argv[1]), Grid(argv[2])
    print(f"a: {a.name} {a.n} origin {a.o} spacing {a.h}")
    print(f"b: {b.name} {b.n} origin {b.o} spacing {b.h}")
    x, tets = load_mesh(argv[3], blocks)
    nodes = skin_nodes(tets)
    p = x[:, nodes]
    report("skin nodes, undeformed", a.sample(p), b.sample(p), shift)
    if displacement:
        u = np.vstack([np.fromfile(os.path.join(displacement, "point_data", f"disp_{c}.float64")) for c in "xyz"])
        q = p + u[:, nodes]
        report("skin nodes, displaced", a.sample(q), b.sample(q), shift)


if __name__ == "__main__":
    main(sys.argv)
