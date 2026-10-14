#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import numpy as np


def detect(folder: Path, stem: str, extensions: tuple[str, ...]) -> Path | None:
    for ext in extensions:
        path = folder / f"{stem}.{ext}"
        if path.exists():
            return path
    return None


def dtype_from_suffix(path: Path, fallback):
    suffix = path.suffix.lstrip(".")
    if suffix == "float64":
        return np.float64
    if suffix == "float32":
        return np.float32
    if suffix == "float16":
        return np.float16
    if suffix == "int64":
        return np.int64
    if suffix == "int32":
        return np.int32
    if suffix == "int16":
        return np.int16
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("output")
    args = parser.parse_args()

    mesh = Path(args.mesh)
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    meta = mesh / "meta.yaml"
    if meta.exists():
        shutil.copyfile(meta, output / "meta.yaml")

    coords = []
    for stem in ("x", "y", "z"):
        path = detect(mesh, stem, ("float32", "float64", "float16", "raw"))
        if path is None:
            break
        coords.append(np.fromfile(path, dtype=dtype_from_suffix(path, np.float32)).astype(np.float32))

    if len(coords) < 2:
        raise RuntimeError(f"expected at least x/y coordinates in {mesh}")

    n_nodes = len(coords[0])
    if len(coords) == 2:
        coords.append(np.zeros(n_nodes, dtype=np.float32))

    for d, stem in enumerate(("x", "y", "z")):
        coords[d].tofile(output / f"{stem}.float32")

    i = 0
    while True:
        path = detect(mesh, f"i{i}", ("int32", "int64", "int16", "raw"))
        if path is None:
            break
        idx = np.fromfile(path, dtype=dtype_from_suffix(path, np.int32)).astype(np.int32)
        idx.tofile(output / f"i{i}.int32")
        i += 1

    if i == 0:
        raise RuntimeError(f"expected connectivity streams in {mesh}")

    print(f"Prepared XDMF visualization mesh: {output}")


if __name__ == "__main__":
    main()
