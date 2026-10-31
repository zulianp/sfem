# Part of qp/viz/twist_movies.sh.
"""XDMF time series for the raw output of the playground's FieldExport (mesh/ + out/disp.<c>.<step>).

    python make_xdmf.py CASE_DIR [--macro N]

Writes CASE_DIR/xdmf/{conn.int32, disp_<step>.float64, points.float32} and CASE_DIR/case.xdmf.  With
--macro N the mesh is the refined trilinear grid of N^3 triquadratic elements: CASE_DIR/macro.xdmf
then holds the macro hexahedra (corner nodes) on the same points and displacements.
"""

import glob
import os
import re
import sys

import numpy as np

TOPOLOGY = {"TET4": "Tetrahedron", "TET10": "Tetrahedron_10", "HEX8": "Hexahedron"}


def meta(path):
    m = {}
    for line in open(path):
        k = re.match(r"(\w+):\s*(\S+)", line)
        if k:
            m[k.group(1)] = k.group(2)
    return m


def grid(name, topo, nconn, npe, nn, step, t):
    extra = f' NodesPerElement="{npe}"' if topo == "Polyline" else ""
    return f"""    <Grid Name="{name}" GridType="Uniform">
      <Time Value="{t}"/>
      <Topology TopologyType="{topo}" NumberOfElements="{nconn}"{extra}>
        <DataItem Dimensions="{nconn} {npe}" NumberType="Int" Precision="4" Format="Binary">xdmf/{name}_conn.int32</DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="{nn} 3" NumberType="Float" Precision="4" Format="Binary">xdmf/points.float32</DataItem>
      </Geometry>
      <Attribute Name="disp" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{nn} 3" NumberType="Float" Precision="8" Format="Binary">xdmf/disp_{step}.float64</DataItem>
      </Attribute>
    </Grid>
"""


def write_xdmf(path, name, topo, conn, nn, steps):
    with open(path, "w") as f:
        f.write('<?xml version="1.0" ?>\n<Xdmf Version="3.0">\n  <Domain>\n')
        f.write('   <Grid Name="series" GridType="Collection" CollectionType="Temporal">\n')
        for k, s in enumerate(steps):
            f.write(grid(name, topo, conn.shape[0], conn.shape[1], nn, s, k / (len(steps) - 1)))
        f.write("   </Grid>\n  </Domain>\n</Xdmf>\n")


def main():
    case = sys.argv[1]
    macro = int(sys.argv[sys.argv.index("--macro") + 1]) if "--macro" in sys.argv else 0
    m = meta(os.path.join(case, "mesh", "meta.yaml"))
    nn, npe = int(m["n_nodes"]), int(m["elem_num_nodes"])
    xyz = np.stack([np.fromfile(os.path.join(case, "mesh", c + ".float32"), dtype=np.float32) for c in "xyz"], axis=1)
    conn = np.stack([np.fromfile(os.path.join(case, "mesh", f"i{k}.int32"), dtype=np.int32) for k in range(npe)], axis=1)
    out = os.path.join(case, "xdmf")
    os.makedirs(out, exist_ok=True)
    xyz.astype(np.float32).tofile(os.path.join(out, "points.float32"))
    conn.astype(np.int32).tofile(os.path.join(out, "mesh_conn.int32"))
    steps = sorted({int(re.search(r"disp\.0\.(\d+)\.float64", p).group(1))
                    for p in glob.glob(os.path.join(case, "out", "disp.0.*.float64"))})
    for s in steps:
        d = np.stack([np.fromfile(os.path.join(case, "out", f"disp.{c}.{s:09d}.float64"), dtype=np.float64)
                      for c in range(3)], axis=1)
        d.tofile(os.path.join(out, f"disp_{s}.float64"))
    write_xdmf(os.path.join(case, "case.xdmf"), "mesh", TOPOLOGY[m["element_type"]], conn, nn, steps)
    # The true element edges of the quadratic elements, as 3-node polylines through the mid-edge node
    # (curved with the deformation), for drawing over a smooth surface.
    edges = []
    if macro:
        # The refined grid has 2 * macro intervals per direction; macro edges run along the even lattice.
        n = 2 * macro
        ijk = np.rint(xyz * n).astype(int)
        lookup = {tuple(p): i for i, p in enumerate(ijk)}
        for axis in range(3):
            for u in range(0, n + 1, 2):
                for v in range(0, n + 1, 2):
                    for w in range(0, n, 2):
                        pts = []
                        for o in range(3):
                            p = [0, 0, 0]
                            p[axis] = w + o
                            p[(axis + 1) % 3] = u
                            p[(axis + 2) % 3] = v
                            pts.append(lookup[tuple(p)])
                        edges.append(pts)
    elif m["element_type"] == "TET10":
        seen = set()
        for e in conn:
            for a, b, mid in ((0, 1, 4), (1, 2, 5), (0, 2, 6), (0, 3, 7), (1, 3, 8), (2, 3, 9)):
                key = (min(e[a], e[b]), max(e[a], e[b]))
                if key not in seen:
                    seen.add(key)
                    edges.append([e[a], e[mid], e[b]])
    if edges:
        # Only the edges on the cube's boundary: all three nodes on one face plane of the reference cube.
        edges = np.array(edges, dtype=np.int32)
        on = np.zeros(len(edges), dtype=bool)
        for axis in range(3):
            for v in (0.0, 1.0):
                on |= np.all(np.abs(xyz[edges][:, :, axis] - v) < 1e-6, axis=1)
        edges = edges[on]
        if m["element_type"] == "TET10":
            # The surface of quadratic tetrahedra is drawn curved: sample each edge along its quadratic
            # curve on the deformed geometry, per step (nodes a, mid, b at s = 0, 1/2, 1).
            S = 9
            sv = np.linspace(0, 1, S)
            Na, Nm, Nb = (1 - sv) * (1 - 2 * sv), 4 * sv * (1 - sv), sv * (2 * sv - 1)
            conn_s = np.arange(len(edges) * S, dtype=np.int32).reshape(len(edges), S)
            conn_s.tofile(os.path.join(out, "cedges_conn.int32"))
            with open(os.path.join(case, "edges.xdmf"), "w") as fx:
                fx.write('<?xml version="1.0" ?>\n<Xdmf Version="3.0">\n  <Domain>\n')
                fx.write('   <Grid Name="series" GridType="Collection" CollectionType="Temporal">\n')
                for k, st in enumerate(steps):
                    d = np.fromfile(os.path.join(out, f"disp_{st}.float64"), dtype=np.float64).reshape(nn, 3)
                    X = xyz.astype(np.float64) + d
                    P = (Na[None, :, None] * X[edges[:, 0]][:, None, :] + Nm[None, :, None] * X[edges[:, 1]][:, None, :]
                         + Nb[None, :, None] * X[edges[:, 2]][:, None, :])
                    P.astype(np.float32).tofile(os.path.join(out, f"cedges_{st}.float32"))
                    fx.write(f'''    <Grid Name="edges" GridType="Uniform">
      <Time Value="{k / (len(steps) - 1)}"/>
      <Topology TopologyType="Polyline" NumberOfElements="{len(edges)}" NodesPerElement="{S}">
        <DataItem Dimensions="{len(edges)} {S}" NumberType="Int" Precision="4" Format="Binary">xdmf/cedges_conn.int32</DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="{len(edges) * S} 3" NumberType="Float" Precision="4" Format="Binary">xdmf/cedges_{st}.float32</DataItem>
      </Geometry>
    </Grid>
''')
                fx.write("   </Grid>\n  </Domain>\n</Xdmf>\n")
        else:
            edges.tofile(os.path.join(out, "edges_conn.int32"))
            write_xdmf(os.path.join(case, "edges.xdmf"), "edges", "Polyline", edges, nn, steps)
    print(case, m["element_type"], nn, "nodes", conn.shape[0], "cells", len(steps), "steps",
          "max |u|", float(np.max(np.linalg.norm(d, axis=1))))


if __name__ == "__main__":
    main()
