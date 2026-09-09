#!/usr/bin/env python3
"""Create a ParaView XDMF file from an SFEM CVFEM HEX8 output folder.

Reads <folder>/mesh (SoA HEX8) plus a field folder, stacks velocity as an N x 3
vector for glyphs, and writes XDMF.

Three layouts, detected rather than configured, because two drivers write
different ones and a transient run writes a third:

    <folder>/out/{u.0,u.1,u.2,p}      cvfem_hex8_ns_steady
    <folder>/{vel.0,vel.1,vel.2,p}    cvfem_hex8_ns_ssgmg, final state only
    <folder>/step_NNNN/{vel.*,p}      cvfem_hex8_ns_ssgmg with SFEM_WRITE_STEPS=1

The last becomes a TIME SERIES -- one XDMF that ParaView animates -- which is the
only useful way to look at a case whose whole point is the cycle. Each step
carries a time.txt so the frames are placed at their real times rather than at
their indices. The mesh is written once and shared by every frame: this is
transpiration on a fixed mesh, so there is one geometry for all of them.

usage: create_xdmf.py <output_folder> [output.xdmf]
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

EXT_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "raw": np.float32,
}


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def _ext(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _detect(pattern: str, extensions) -> list[str]:
    files = glob.glob(pattern)
    return [f for f in files if _ext(f) in extensions]


def _read_array(path: str) -> np.ndarray:
    dtype = EXT_DTYPE.get(_ext(path))
    if dtype is None:
        _die(f"unknown binary extension: {path}")
    return np.fromfile(path, dtype=dtype)


def _find_one(folder: str, stem: str, extensions) -> str:
    matches = _detect(os.path.join(folder, f"{stem}.*"), extensions)
    if not matches:
        _die(f"missing {stem}.* in {folder}")
    matches.sort()
    return matches[0]


def _read_coords(mesh_folder: str) -> np.ndarray:
    coords = []
    for name in ("x", "y", "z"):
        path = _detect(
            os.path.join(mesh_folder, f"{name}.*"),
            ("float16", "float32", "float64", "raw"),
        )
        if not path:
            break
        coords.append(_read_array(path[0]).astype(np.float64, copy=False))
    if len(coords) != 3:
        _die(f"expected x/y/z coordinates in {mesh_folder}")
    n = coords[0].size
    if any(c.size != n for c in coords):
        _die(f"coordinate length mismatch in {mesh_folder}")
    return np.column_stack(coords)


def _read_hex8_cells(mesh_folder: str) -> np.ndarray:
    idx = []
    for i in range(8):
        path = _detect(
            os.path.join(mesh_folder, f"i{i}.*"),
            ("raw", "int16", "int32", "int64"),
        )
        if not path:
            _die(f"expected HEX8 connectivity i0..i7 in {mesh_folder} (missing i{i}.*)")
        idx.append(_read_array(path[0]))
    n = idx[0].size
    if any(a.size != n for a in idx):
        _die(f"connectivity length mismatch in {mesh_folder}")
    return np.column_stack(idx).astype(np.int64, copy=False)


def _read_field(out_folder: str, stem: str) -> np.ndarray:
    path = _find_one(out_folder, stem, ("float64", "float32", "raw"))
    return _read_array(path).astype(np.float64, copy=False)


def _write_xdmf(mesh, path: str) -> None:
    import meshio

    try:
        mesh.write(path)
    except Exception as exc:
        print(f"HDF XDMF write failed ({exc}); writing XML XDMF", file=sys.stderr)
        mesh.write(path, data_format="XML")


def _find_field_dirs(root: str):
    """(label, folder, stem_prefix, time) per frame, newest layout first."""
    steps = sorted(glob.glob(os.path.join(root, "step_[0-9]" * 1 + "[0-9][0-9][0-9]")))
    if steps:
        frames = []
        for i, d in enumerate(steps):
            t = float(i)
            tf = os.path.join(d, "time.txt")
            if os.path.exists(tf):
                with open(tf) as fh:
                    try:
                        t = float(fh.read().strip())
                    except ValueError:
                        pass
            frames.append((os.path.basename(d), d, "vel", t))
        return frames
    out = os.path.join(root, "out")
    if os.path.isdir(out) and _detect(os.path.join(out, "u.0.*"), ("float64", "float32", "raw")):
        return [("out", out, "u", 0.0)]
    if _detect(os.path.join(root, "vel.0.*"), ("float64", "float32", "raw")):
        return [("final", root, "vel", 0.0)]
    return []


def _read_frame(folder: str, prefix: str, nnodes: int):
    ux = _read_field(folder, prefix + ".0")
    uy = _read_field(folder, prefix + ".1")
    uz = _read_field(folder, prefix + ".2")
    p = _read_field(folder, "p")
    for name, arr in ((prefix + ".0", ux), (prefix + ".1", uy), (prefix + ".2", uz), ("p", p)):
        if arr.size != nnodes:
            _die(f"{name} length {arr.size} != nnodes {nnodes}")
    return np.column_stack((ux, uy, uz)), p


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(argv) >= 2 else 1

    root = os.path.abspath(argv[1])
    mesh_folder = os.path.join(root, "mesh")
    if not os.path.isdir(mesh_folder):
        _die(f"missing mesh folder: {mesh_folder}")
    frames = _find_field_dirs(root)
    if not frames:
        _die(f"no fields under {root}: expected out/u.0.*, vel.0.*, or step_NNNN/")

    out_path = (
        os.path.abspath(argv[2])
        if len(argv) > 2
        else os.path.join(root, "output.xdmf")
    )
    if not out_path.endswith(".xdmf"):
        out_path = out_path + ".xdmf"

    points = _read_coords(mesh_folder)
    cells = _read_hex8_cells(mesh_folder)
    nnodes = points.shape[0]
    import meshio

    if len(frames) == 1:
        _, folder, prefix, _ = frames[0]
        velocity, p = _read_frame(folder, prefix, nnodes)
        mesh = meshio.Mesh(points, [("hexahedron", cells)], point_data={"u": velocity, "p": p})
        _write_xdmf(mesh, out_path)
        print(f"wrote: {out_path}")
        print(f"  nnodes={nnodes}  nelements={cells.shape[0]}")
        print(f"  u range=[{velocity.min():.6g}, {velocity.max():.6g}]")
        return 0

    # A time series. The geometry is written once and every frame refers to it, which is
    # both smaller and correct: the mesh does not move.
    #
    # Written from inside the destination directory, and that is not fussiness.
    # TimeSeriesWriter puts the heavy data in a sidecar .h5 which the .xdmf references by
    # BARE FILENAME, but it creates that file relative to the process's working directory --
    # so writing to another folder leaves an .xdmf pointing at an .h5 that is not beside it,
    # and the pair only opens from whichever directory the generator happened to run in.
    lo, hi = float("inf"), float("-inf")
    out_dir = os.path.dirname(out_path) or "."
    cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        with meshio.xdmf.TimeSeriesWriter(os.path.basename(out_path)) as writer:
            writer.write_points_cells(points, [("hexahedron", cells)])
            for label, folder, prefix, t in frames:
                velocity, p = _read_frame(folder, prefix, nnodes)
                writer.write_data(t, point_data={"u": velocity, "p": p})
                lo, hi = min(lo, float(velocity.min())), max(hi, float(velocity.max()))
    finally:
        os.chdir(cwd)
    print(f"wrote: {out_path}  ({len(frames)} frames)")
    print(f"  nnodes={nnodes}  nelements={cells.shape[0]}")
    print(f"  t range=[{frames[0][3]:.6g}, {frames[-1][3]:.6g}]")
    print(f"  u range=[{lo:.6g}, {hi:.6g}]")
    print(f"  p range=[{p.min():.6g}, {p.max():.6g}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
