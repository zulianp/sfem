#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import shutil
import sys

import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common import playground
from common.transient import TIME_LEVELS


def temperature_label(value):
    return "T" + format(float(value), ".8g").replace("-", "m").replace(".", "p")


def generated_operator_config(environment, temperature):
    """The YAML the generated Prony operator is configured with, from the case's variables."""
    g = [float(value) for value in environment["SFEM_PRONY_G"].split(",") if value]
    tau = [float(value) for value in environment["SFEM_PRONY_TAU"].split(",") if value]
    group = {"prony": [{"g": gi, "tau": ti} for gi, ti in zip(g, tau)]}
    if environment.get("SFEM_USE_WLF", "0") not in ("", "0"):
        group["wlf"] = {
            "C1": float(environment["SFEM_WLF_C1"]),
            "C2": float(environment["SFEM_WLF_C2"]),
            "T_ref": float(environment["SFEM_WLF_T_REF"]),
        }
        group["temperature"] = float(temperature)
    if environment.get("SFEM_PRONY_HISTORY_STORAGE"):
        group["history_storage"] = environment["SFEM_PRONY_HISTORY_STORAGE"]
    return {
        "type": environment["SFEM_OPERATOR"],
        "C10": float(environment["SFEM_MOONEY_RIVLIN_C10"]),
        "C01": float(environment["SFEM_MOONEY_RIVLIN_C01"]),
        "K": float(environment["SFEM_MOONEY_RIVLIN_K"]),
        "relaxation": {"iso": group},
    }


def run_level(args, output, duration, dt, temperature, environment=None):
    """One relaxation run; `environment` holds the case's material variables (os.environ by default)."""
    environment = os.environ if environment is None else environment
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    steps = round(duration / dt)
    if abs(steps * dt - duration) > 1.0e-12:
        raise ValueError(f"duration {duration} is not divisible by dt={dt}")
    case = {
        "mesh": {"path": str(args.mesh), "packed": False},
        "material": generated_operator_config(environment, temperature),
        "dirichlet_conditions": yaml.safe_load(args.dirichlet.read_text(encoding="utf-8"))["dirichlet_conditions"],
        "regime": "quasi_static",
        "time": {"dt": dt, "steps": steps},
        "output": {"path": str(output), "export_freq": steps, "fields": ["material_reaction"], "quantities": True},
    }
    completed = playground.run(args.driver, case, output / "case.yaml", args.overrides, os.environ.copy())
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode:
        raise RuntimeError(f"playground exited with status {completed.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run Prony relaxation and history-commit checks")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--dirichlet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="playground case override key=value")
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    duration = float(config["time"]["duration"])
    time_steps = [float(value) for value in config["time"]["time_steps"]]
    temperatures = [float(value) for value in os.environ.get("SFEM_PRONY_TEMPERATURES", "20").split(",")]
    if len(time_steps) != len(TIME_LEVELS):
        raise ValueError("Prony relaxation requires exactly three time-step sizes")
    if len(temperatures) not in (1, 2):
        raise ValueError("Prony variants require one reference temperature or one temperature pair")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    for temperature in temperatures:
        root = args.output / temperature_label(temperature)
        for name, dt in zip(TIME_LEVELS, time_steps):
            run_level(args, root / name, duration, dt, temperature)
            print("SFEM_TRANSIENT_LEVEL_COMPLETE")

    print("SFEM_PRONY_RELAXATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
