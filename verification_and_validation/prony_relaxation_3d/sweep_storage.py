#!/usr/bin/env python3
"""How reduced history storage degrades the Prony relaxation, format by format.

Runs the relaxation of a resolved `prony_relaxation_3d` variant on the generated
operator once per `HistoryStorage` format of the internal variables H_i -- the
only stored history; the relaxed stress is evaluated from the displacement at
each step boundary -- at the finest time step, and reports for each:

  - the reaction history against the full-storage run (what the storage costs),
  - the reaction history against the analytical oracle (what the user sees),
  - the long-time modulus error,
  - the bytes of history per element.

The recursion re-encodes H_i every step, so rounding accumulates over the run;
this is the measurement that says which formats are safe for a given spectrum
and step count.  Nothing here asserts: it is a characterisation, read beside the
bound `sfem_PronyRelaxationTest` holds the codecs to.

    sweep_storage.py --variant-dir <run_all output>/prony_relaxation_3d/generated_multi_term_reference \
                     --driver <build>/playground --out <dir>
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import yaml

CASE_DIR = Path(__file__).resolve().parent
SUITE_DIR = CASE_DIR.parent
for path in (SUITE_DIR, CASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.metrics import relative_l2_error  # noqa: E402
from common.run_prony_relaxation import run_level  # noqa: E402
from oracle import elastic_axial_reaction, parse_series, relaxation_factor  # noqa: E402
from verify import reaction_history  # noqa: E402

FORMATS = ("full", "float32", "float16_scaled", "int16_scaled", "int8_scaled")

# Bytes per stored value, and whether a per-(element, tensor) float scale rides along.
STORAGE_BYTES = {"full": 8, "float32": 4, "float16_scaled": 2, "int16_scaled": 2, "int8_scaled": 1}


def history_bytes_per_element(storage, n_terms, n_components=6, n_qp=8):
    """H_i only; the previous displacement adds 3 values per node."""
    scales = n_terms * 4 if storage not in ("full", "float32") else 0
    return n_terms * n_components * n_qp * STORAGE_BYTES[storage] + scales


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant-dir", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    case = yaml.safe_load((args.variant_dir / "resolved_case.yaml").read_text(encoding="utf-8"))
    variant = case["selected_variant"]
    resolution = variant["resolution"]
    environment = os.environ.copy()
    placeholders = {"resolution_%s" % key: value for key, value in resolution.items()}
    placeholders.update({"material_%s" % key: value for key, value in case["material"].items()})
    environment.update(
        {key: str(value).format(**placeholders) for key, value in case["driver"]["environment"].items()}
    )
    temperature = float(str(resolution["temperatures"]).split(",")[0])
    duration = float(case["time"]["duration"])
    dt = float(case["time"]["time_steps"][-1])
    steps = round(duration / dt)
    arguments = [str(value) for value in case["driver"]["arguments"]]
    run_args = argparse.Namespace(
        driver=args.driver,
        mesh=args.variant_dir / "mesh",
        dirichlet=args.variant_dir / "dirichlet.yaml",
        overrides=[arguments[i + 1] for i, value in enumerate(arguments[:-1]) if value == "--set"],
    )

    g = parse_series(str(resolution["prony_g"]))
    tau = parse_series(str(resolution["prony_tau"]))
    material = case["material"]
    elastic = elastic_axial_reaction(
        case["loading"]["strain"], material["c10"], material["c01"], material["bulk_modulus"]
    )
    expected_times = np.arange(1, steps + 1, dtype=np.float64) * dt

    results = {}
    for storage in FORMATS:
        output = args.out / storage
        output.mkdir(parents=True, exist_ok=True)
        run_environment = dict(environment)
        run_environment["SFEM_PRONY_HISTORY_STORAGE"] = storage
        run_level(run_args, output, duration, dt, temperature, run_environment)
        times, reaction = reaction_history(output, expected_times)
        results[storage] = (times, reaction / elastic)

    full = results["full"][1]
    expected = relaxation_factor(expected_times, g, tau, temperature=temperature, wlf=None)
    long_time = 1.0 - np.sum(g)
    print("variant %s: %d terms, %d steps of dt=%g" % (variant["id"], len(g), steps, dt))
    print("%-16s %12s %12s %12s %11s" % ("H storage", "vs full", "vs oracle", "long-time", "bytes/elem"))
    for storage in FORMATS:
        factor = results[storage][1]
        print(
            "%-16s %12.3e %12.3e %12.3e %11d"
            % (
                storage,
                relative_l2_error(factor, full),
                relative_l2_error(factor, expected),
                abs(factor[-1] - long_time) / long_time,
                history_bytes_per_element(storage, len(g)),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
