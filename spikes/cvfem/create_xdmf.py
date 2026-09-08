#!/usr/bin/env python3
"""Create a ParaView XDMF file from an SFEM CVFEM HEX8 output folder.

Reads <folder>/mesh (SoA HEX8) and <folder>/out (u.0, u.1, u.2, p),
stacks velocity as an N x 3 vector for glyphs, and writes XDMF.

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


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if len(argv) >= 2 else 1

    root = os.path.abspath(argv[1])
    mesh_folder = os.path.join(root, "mesh")
    out_folder = os.path.join(root, "out")
    if not os.path.isdir(mesh_folder):
        _die(f"missing mesh folder: {mesh_folder}")
    if not os.path.isdir(out_folder):
        _die(f"missing out folder: {out_folder}")

    out_path = (
        os.path.abspath(argv[2])
        if len(argv) > 2
        else os.path.join(root, "output.xdmf")
    )
    if not out_path.endswith(".xdmf"):
        out_path = out_path + ".xdmf"

    points = _read_coords(mesh_folder)
    cells = _read_hex8_cells(mesh_folder)
    ux = _read_field(out_folder, "u.0")
    uy = _read_field(out_folder, "u.1")
    uz = _read_field(out_folder, "u.2")
    p = _read_field(out_folder, "p")
    nnodes = points.shape[0]
    for name, arr in (("u.0", ux), ("u.1", uy), ("u.2", uz), ("p", p)):
        if arr.size != nnodes:
            _die(f"{name} length {arr.size} != nnodes {nnodes}")

    velocity = np.column_stack((ux, uy, uz))
    import meshio

    mesh = meshio.Mesh(
        points,
        [("hexahedron", cells)],
        point_data={"u": velocity, "p": p},
    )
    _write_xdmf(mesh, out_path)
    print(f"wrote: {out_path}")
    print(f"  nnodes={nnodes}  nelements={cells.shape[0]}")
    print(f"  u range=[{velocity.min():.6g}, {velocity.max():.6g}]")
    print(f"  p range=[{p.min():.6g}, {p.max():.6g}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
