#!/usr/bin/env python3
"""Collect nodal error tables for generated Stokes verification runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    from .stokes_mms import CASES, case_by_name, describe_cases
except ImportError:
    from stokes_mms import CASES, case_by_name, describe_cases


REAL_DTYPE = np.float64
GEOM_DTYPE = np.float32


@dataclass(frozen=True)
class Level:
    name: str
    h: float
    mesh_dir: Path
    solution_dir: Path


def parse_level(raw: str) -> Level:
    parts = raw.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "level must have form <name>:<h>:<mesh-dir>:<solution-dir>"
        )
    name, h, mesh_dir, solution_dir = parts
    h_value = float(h)
    if h_value <= 0:
        raise argparse.ArgumentTypeError("level h must be positive")
    return Level(name, h_value, Path(mesh_dir), Path(solution_dir))


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


def relative_l2(numerical: np.ndarray, exact: np.ndarray) -> Tuple[float, float]:
    diff = numerical - exact
    abs_error = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sqrt(np.mean(exact * exact)))
    rel_error = abs_error / denom if denom > 0 else abs_error
    return abs_error, rel_error


def level_errors(case_name: str, level: Level, pressure_mean_free: bool) -> dict:
    case = case_by_name(case_name)
    coord_names = ("x", "y", "z")[: case.dim]
    coords = tuple(
        read_nodal_field(level.mesh_dir, name, GEOM_DTYPE, ("float64",)).astype(REAL_DTYPE)
        for name in coord_names
    )
    velocity = tuple(
        read_nodal_field(level.solution_dir, "u%d" % d, REAL_DTYPE, ("float32",))
        for d in range(case.dim)
    )
    p = read_nodal_field(level.solution_dir, "p", REAL_DTYPE, ("float32",))

    lengths = tuple(len(values) for values in (*coords, *velocity, p))
    if any(length != lengths[0] for length in lengths):
        raise ValueError("mesh and solution arrays have inconsistent lengths for %s" % level.name)

    exact_velocity = case.velocity(*coords)
    exact_p = case.pressure(*coords)
    if pressure_mean_free:
        p = p - np.mean(p)
        exact_p = exact_p - np.mean(exact_p)

    component_errors = [
        relative_l2(numerical, exact)
        for numerical, exact in zip(velocity, exact_velocity)
    ]
    p_abs, p_rel = relative_l2(p, exact_p)
    velocity_abs = float(
        np.sqrt(sum(abs_error * abs_error for abs_error, _ in component_errors))
    )
    velocity_ref = float(
        np.sqrt(np.mean(sum(component * component for component in exact_velocity)))
    )
    velocity_rel = velocity_abs / velocity_ref if velocity_ref > 0 else velocity_abs

    row = {
        "case": case.name,
        "level": level.name,
        "h": level.h,
        "nnodes": lengths[0],
        "velocity_l2_abs": velocity_abs,
        "velocity_l2_rel": velocity_rel,
        "pressure_l2_abs": p_abs,
        "pressure_l2_rel": p_rel,
    }
    for d, (abs_error, rel_error) in enumerate(component_errors):
        row["u%d_l2_abs" % d] = abs_error
        row["u%d_l2_rel" % d] = rel_error
    return row


def convergence_rates(rows: Iterable[dict], key: str) -> list:
    ordered = sorted(rows, key=lambda row: float(row["h"]), reverse=True)
    rates = [""]
    for prev, curr in zip(ordered, ordered[1:]):
        e0 = float(prev[key])
        e1 = float(curr[key])
        h0 = float(prev["h"])
        h1 = float(curr["h"])
        if e0 > 0 and e1 > 0 and h0 != h1:
            rates.append(np.log(e0 / e1) / np.log(h0 / h1))
        else:
            rates.append("")
    return rates


def write_csv(rows: list, path: Path) -> None:
    rows = sorted(rows, key=lambda row: float(row["h"]), reverse=True)
    velocity_rates = convergence_rates(rows, "velocity_l2_abs")
    pressure_rates = convergence_rates(rows, "pressure_l2_abs")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    fields += ["velocity_rate", "pressure_rate"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, velocity_rate, pressure_rate in zip(rows, velocity_rates, pressure_rates):
            enriched = dict(row)
            enriched["velocity_rate"] = velocity_rate
            enriched["pressure_rate"] = pressure_rate
            writer.writerow(enriched)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print available manufactured-solution cases and exit",
    )
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument(
        "--level",
        action="append",
        type=parse_level,
        help="refinement level as <name>:<h>:<mesh-dir>:<solution-dir>",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--pressure-mean-free",
        action="store_true",
        help="subtract nodal mean before pressure error calculation",
    )
    args = parser.parse_args(argv)
    if args.list_cases:
        print(describe_cases())
        return 0
    if args.case is None:
        parser.error("--case is required unless --list-cases is used")
    if not args.level:
        parser.error("--level is required unless --list-cases is used")
    if args.out is None:
        parser.error("--out is required unless --list-cases is used")

    rows = [
        level_errors(args.case, level, args.pressure_mean_free)
        for level in args.level
    ]
    write_csv(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
