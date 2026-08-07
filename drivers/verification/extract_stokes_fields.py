#!/usr/bin/env python3
"""Extract numerical, exact, and error fields for a Stokes verification run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from .stokes_mms import CASES, case_by_name, describe_cases
except ImportError:
    from stokes_mms import CASES, case_by_name, describe_cases


REAL_DTYPE = np.float64
GEOM_DTYPE = np.float32


def read_nodal_field(directory: Path, name: str, dtype, fallback_dtypes=()) -> np.ndarray:
    extensions = []
    if np.dtype(dtype) == np.dtype(np.float32):
        extensions.append("float32")
    elif np.dtype(dtype) == np.dtype(np.float64):
        extensions.append("float64")
    extensions.extend(fallback_dtypes)
    extensions.append("raw")

    seen = set()
    for extension in extensions:
        if extension in seen:
            continue
        seen.add(extension)
        path = directory / ("%s.%s" % (name, extension))
        if path.exists():
            return np.fromfile(path, dtype=dtype)
    raise FileNotFoundError(str(directory / ("%s.<float32|float64|raw>" % name)))


def extract_fields(case_name: str, mesh_dir: Path, solution_dir: Path, pressure_mean_free: bool) -> dict:
    case = case_by_name(case_name)

    coord_names = ("x", "y", "z")[: case.dim]
    coords = tuple(
        read_nodal_field(mesh_dir, name, GEOM_DTYPE, ("float64",)).astype(REAL_DTYPE)
        for name in coord_names
    )
    velocity = tuple(
        read_nodal_field(solution_dir, "u%d" % d, REAL_DTYPE, ("float32",))
        for d in range(case.dim)
    )
    p = read_nodal_field(solution_dir, "p", REAL_DTYPE, ("float32",))
    lengths = tuple(len(values) for values in (*coords, *velocity, p))
    if any(length != lengths[0] for length in lengths):
        raise ValueError("mesh and solution arrays have inconsistent lengths")

    exact_velocity = case.velocity(*coords)
    p_exact = case.pressure(*coords)
    if pressure_mean_free:
        p = p - np.mean(p)
        p_exact = p_exact - np.mean(p_exact)

    velocity_error_components = tuple(
        numerical - exact for numerical, exact in zip(velocity, exact_velocity)
    )
    p_error = p - p_exact
    velocity_error = np.sqrt(
        sum(component * component for component in velocity_error_components)
    )
    fields = {
        name: values for name, values in zip(coord_names, coords)
    }
    for d, values in enumerate(velocity):
        fields["u%d" % d] = values
    fields["p"] = p
    for d, values in enumerate(exact_velocity):
        fields["u%d_exact" % d] = values
    fields.update({
        "p": p,
        "p_exact": p_exact,
    })
    for d, values in enumerate(velocity_error_components):
        fields["u%d_error" % d] = values
    fields.update(
        {
            "p_error": p_error,
            "velocity_error": velocity_error,
            "pressure_error_abs": np.abs(p_error),
        }
    )
    return fields


def write_csv(fields: dict, path: Path) -> None:
    names = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for values in zip(*(fields[name] for name in names)):
            writer.writerow(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print available manufactured-solution cases and exit",
    )
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--solution-dir", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-npz", type=Path)
    parser.add_argument(
        "--pressure-mean-free",
        action="store_true",
        help="subtract nodal mean before pressure comparison",
    )
    args = parser.parse_args(argv)
    if args.list_cases:
        print(describe_cases())
        return 0
    if args.case is None:
        parser.error("--case is required unless --list-cases is used")
    if args.mesh_dir is None:
        parser.error("--mesh-dir is required unless --list-cases is used")
    if args.solution_dir is None:
        parser.error("--solution-dir is required unless --list-cases is used")
    if args.out_csv is None:
        parser.error("--out-csv is required unless --list-cases is used")

    fields = extract_fields(args.case, args.mesh_dir, args.solution_dir, args.pressure_mean_free)
    write_csv(fields, args.out_csv)
    if args.out_npz is not None:
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_npz, **fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
