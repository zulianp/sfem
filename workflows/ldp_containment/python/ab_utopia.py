#!/usr/bin/env python3
"""A containment solved by SFEM against the utopia result of the same case.

usage: ab_utopia.py <case> [--output=output] [--sdf=db/sdf] [--blocks=A,B] [--shift=1e-5]
                    [--young=9.174e6] [--poisson=0.4] [--csv=ab.csv] [--label=name]

  <case>/db/solid           the whole solid mesh (exodusII_to_raw layout)
  <case>/db/utopia          the utopia result converted the same way (point_data/disp_*)
  <case>/<output>           the playground output: mesh/ (node_mapping), out/disp.{0,1,2}.<step>
  --sdf                     the grid both states are measured on (positive inside the obstacle)
  --blocks                  the blocks SFEM computed (default all)

Measured on the nodes of those blocks (S = SFEM's last state, U = utopia's):
  displacement   max and L2 of |u_S - u_U| relative to U's, also after removing the best-fit rigid
                 motion of each block (a free valve may settle in a different rigid position)
  penetration    the largest sdf(X + u) over the skin nodes (>0 is inside the obstacle), and the
                 largest violation of the shifted constraint, sdf + shift
  contact        skin nodes within 10 shifts of the obstacle, for each, and their overlap (Jaccard)
  energy         the NeoHookeanOgden elastic energy of each state (TET4, one point per element)
One CSV row is appended to --csv.
"""

import csv
import getopt
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdf_compare import Grid, skin_nodes  # noqa: E402


def read_axis(folder, stem):
    for e in os.listdir(folder):
        if e.split(".")[0] == stem:
            ext = e.split(".")[-1]
            return np.fromfile(os.path.join(folder, e), dtype=np.dtype(ext) if ext != "raw" else np.float32)
    raise FileNotFoundError(f"{folder}/{stem}.*")


def load_blocks(solid, blocks):
    names = sorted(os.listdir(os.path.join(solid, "blocks")))
    if blocks:
        missing = set(blocks) - set(names)
        if missing:
            sys.exit(f"ab_utopia: {solid} has no blocks {sorted(missing)} (has {names})")
        names = [n for n in names if n in blocks]
    out = {}
    for n in names:
        d = os.path.join(solid, "blocks", n)
        out[n] = np.vstack([np.fromfile(os.path.join(d, f"i{k}.int32"), dtype=np.int32) for k in range(4)])
    return out


def sfem_displacement(output, n_parent):
    out = os.path.join(output, "out")
    pat = re.compile(r"^disp\.(\d)\.(\d{9})\.(float\d+)$")
    files = {}
    for e in os.listdir(out):
        m = pat.match(e)
        if m:
            files.setdefault(int(m.group(2)), {})[int(m.group(1))] = (os.path.join(out, e), m.group(3))
    step = max(files)
    comps = [np.fromfile(files[step][c][0], dtype=np.dtype(files[step][c][1])) for c in range(3)]
    mesh = os.path.join(output, "mesh")
    mapping = None
    for e in os.listdir(mesh):
        if e.startswith("node_mapping."):
            mapping = np.fromfile(os.path.join(mesh, e), dtype=np.dtype(e.split(".")[-1])).astype(np.int64)
    if mapping is None:
        mapping = np.arange(len(comps[0]))
    u = np.zeros((3, n_parent))
    for c in range(3):
        u[c, mapping] = comps[c]
    return u, step


def rigid_residual(x, a, b):
    """b - (R a + t) for the rotation R and translation t that best fit b to a on points x:
    a, b displacements of the points x. Kabsch on the deformed positions."""
    p, q = x + a, x + b
    pc, qc = p.mean(axis=1, keepdims=True), q.mean(axis=1, keepdims=True)
    h = (p - pc) @ (q - qc).T
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    fitted = r @ (p - pc) + qc
    return q - fitted


def energy(x, tets, u, mu, lmbda):
    """sum over elements of V (mu/2 (I1 - 3) - mu ln J + lmbda/2 ln^2 J), F = I + grad u."""
    X = x[:, tets]  # 3 x 4 x ne
    U = u[:, tets]
    dX = X[:, 1:, :] - X[:, :1, :]  # 3 x 3 x ne (columns are edges)
    dU = U[:, 1:, :] - U[:, :1, :]
    dX = np.moveaxis(dX, 2, 0)
    dU = np.moveaxis(dU, 2, 0)
    Xinv = np.linalg.inv(dX)
    F = np.eye(3) + dU @ Xinv
    J = np.linalg.det(F)
    V = np.abs(np.linalg.det(dX)) / 6
    I1 = np.einsum("eij,eij->e", F, F)
    if np.any(J <= 0):
        return np.nan, int(np.sum(J <= 0))
    psi = mu / 2 * (I1 - 3) - mu * np.log(J) + lmbda / 2 * np.log(J) ** 2
    return float(np.sum(V * psi)), 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    case = argv[1]
    opts, _ = getopt.getopt(argv[2:], "", ["output=", "sdf=", "blocks=", "shift=", "young=", "poisson=", "csv=", "label="])
    o = dict(output="output", sdf="db/sdf", blocks="", shift="1e-5", young="9.174e6", poisson="0.4", csv="", label="")
    for k, v in opts:
        o[k[2:]] = v
    shift = float(o["shift"])
    E, nu = float(o["young"]), float(o["poisson"])
    mu, lmbda = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

    solid = os.path.join(case, "db", "solid")
    x = np.vstack([read_axis(solid, c).astype(np.float64) for c in "xyz"])
    blocks = load_blocks(solid, [b for b in o["blocks"].split(",") if b])
    uU = np.vstack([np.fromfile(os.path.join(case, "db", "utopia", "point_data", f"disp_{c}.float64")) for c in "xyz"])
    uS, step = sfem_displacement(os.path.join(case, o["output"]), x.shape[1])
    grid = Grid(os.path.join(case, o["sdf"]))

    row = {"case": os.path.basename(os.path.abspath(case)), "label": o["label"], "step": step}
    all_nodes = np.unique(np.concatenate([t.ravel() for t in blocks.values()]))
    d = uS[:, all_nodes] - uU[:, all_nodes]
    nU = np.linalg.norm(uU[:, all_nodes], axis=0)
    row["nodes"] = len(all_nodes)
    row["max_u_utopia"] = nU.max()
    row["max_u_sfem"] = np.linalg.norm(uS[:, all_nodes], axis=0).max()
    row["rel_max_diff"] = np.linalg.norm(d, axis=0).max() / nU.max()
    row["rel_l2_diff"] = np.linalg.norm(d) / np.linalg.norm(uU[:, all_nodes])
    rig = []
    for name, tets in blocks.items():
        nodes = np.unique(tets)
        rig.append(rigid_residual(x[:, nodes], uU[:, nodes], uS[:, nodes]))
    rig = np.hstack(rig)
    row["rel_max_diff_rigid_removed"] = np.linalg.norm(rig, axis=0).max() / nU.max()

    skin = skin_nodes(np.hstack(list(blocks.values())))
    for tag, u in (("utopia", uU), ("sfem", uS)):
        s = grid.sample(x[:, skin] + u[:, skin])
        row[f"outside_grid_{tag}"] = int(np.sum(np.isnan(s)))
        s = np.nan_to_num(s, nan=-np.inf)
        row[f"max_sdf_{tag}"] = s.max()
        row[f"max_violation_{tag}"] = max(0.0, (s + shift).max())
        row[f"contact_{tag}"] = set(skin[s > -10 * shift].tolist())
        e, inverted = energy(x, np.hstack(list(blocks.values())), u, mu, lmbda)
        row[f"energy_{tag}"] = e
        row[f"inverted_{tag}"] = inverted
    cu, cs = row.pop("contact_utopia"), row.pop("contact_sfem")
    row["contact_nodes_utopia"], row["contact_nodes_sfem"] = len(cu), len(cs)
    row["contact_jaccard"] = len(cu & cs) / max(1, len(cu | cs))

    for k, v in row.items():
        print(f"{k:28s} {v:.6g}" if isinstance(v, float) else f"{k:28s} {v}")
    if o["csv"]:
        new = not os.path.exists(o["csv"])
        with open(o["csv"], "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main(sys.argv)
