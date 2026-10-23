#!/usr/bin/env python3

"""Run the linear Kelvin-Voigt damped mode at three time-step sizes on the elasticity playground."""

import argparse
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

SUITE_DIR = Path(__file__).resolve().parents[1]
if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

from common import playground
from common.mesh import read_mesh
from common.sets import write_nodeset
from common.transient import TIME_LEVELS, write_modal_initial_fields


def modal_initial_acceleration(material, length, amplitude, velocity):
    wave_number = np.pi / float(length)
    axial_modulus = float(material["bulk_modulus"]) + 2.0 * float(material["shear_stiffness"]) / 3.0
    viscous_modulus = 2.0 * float(material["damping"]) / 3.0
    density = float(material["density"])
    omega_0_squared = wave_number**2 * axial_modulus / density
    twice_decay = viscous_modulus * wave_number**2 / density
    return -twice_decay * float(velocity) - omega_0_squared * float(amplitude)


def generated_material(material):
    """The case's law, sigma = K tr(eps) I + k dev(eps) + eta dev(eps_dot), in the generated
    material's parameters: sigma = 2 mu eps + lmbda tr(eps) I + 2 eta_s dev(eps_dot)."""
    k = float(material["shear_stiffness"])
    K = float(material["bulk_modulus"])
    return {
        "type": "GeneratedLinearKelvinVoigt",
        "mu": k / 2,
        "lmbda": K - k / 3,
        "eta_s": float(material["damping"]) / 2,
        "eta_b": 0.0,
    }


def mode_case(args, material, density, output, dirichlet, dt=None, steps=None, initial=None):
    case = {
        "mesh": {"path": str(args.mesh), "refine": args.refine, "packed": False},
        "material": material,
        "regime": "dynamic",
        # The consistent mass, as the hand-written operator this case was written against had.
        "dynamics": {"scheme": "newmark", "density": density, "mass": "consistent"},
        "time": {"dt": dt, "steps": steps},
        "output": {"path": str(output), "export_freq": 1},
    }
    if dirichlet is not None:
        case["dirichlet_conditions"] = {"file": str(dirichlet)}
    if initial is not None:
        displacement, velocity, acceleration = initial
        case["initial_conditions"] = {
            "displacement": {"components": displacement.split(",")},
            "velocity": {"components": velocity.split(",")},
            "acceleration": {"components": acceleration.split(",")},
        }
    return case


def run(args, case, output):
    completed = playground.run(args.driver, case, Path(output) / "case.yaml", args.overrides, os.environ.copy())
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode:
        raise RuntimeError(f"playground exited with status {completed.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run the Kelvin-Voigt mode at three time-step sizes")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--dirichlet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refine", type=int, default=0, help="semi-structured refinement level of the mesh")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="playground case override key=value")
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    duration = float(config["time"]["duration"])
    time_steps = [float(value) for value in config["time"]["time_steps"]]
    amplitude = float(config["initial_conditions"]["modal_amplitude"])
    velocity = float(config["initial_conditions"]["modal_velocity"])
    length = float(config["geometry"]["length"])
    material = generated_material(config["material"])
    density = float(config["material"]["density"])
    acceleration = modal_initial_acceleration(config["material"], length, amplitude, velocity)
    if len(time_steps) != len(TIME_LEVELS):
        raise ValueError("damped-mode verification requires exactly three time-step sizes")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    run_dirichlet = args.dirichlet
    if args.refine > 1:
        # The refined mesh's nodes are numbered by the refinement: one step without conditions
        # exports it, and the node sets and the initial fields are built on it.
        prepare = args.output / "prepared"
        run(args, mode_case(args, material, density, prepare, None, time_steps[0], 1), prepare)
        initial_mesh = read_mesh(prepare / "mesh")
        refined_sets = args.output / "refined_sets"
        left = np.flatnonzero(np.isclose(initial_mesh.points[:, 0], 0.0, rtol=0.0, atol=1.0e-7))
        right = np.flatnonzero(np.isclose(initial_mesh.points[:, 0], length, rtol=0.0, atol=1.0e-7))
        all_nodes = np.arange(initial_mesh.n_points, dtype=np.int64)
        left_path = refined_sets / "left.int32.raw"
        right_path = refined_sets / "right.int32.raw"
        all_path = refined_sets / "all.int32.raw"
        write_nodeset(left_path, left)
        write_nodeset(right_path, right)
        write_nodeset(all_path, all_nodes)
        run_dirichlet = args.output / "refined_dirichlet.yaml"
        run_dirichlet.write_text(
            yaml.safe_dump(
                {
                    "dirichlet_conditions": [
                        {"type": "nodeset", "format": "file", "path": str(left_path), "value": 0, "component": 0},
                        {"type": "nodeset", "format": "file", "path": str(right_path), "value": 0, "component": 0},
                        {
                            "type": "nodeset",
                            "format": "file",
                            "path": str(all_path),
                            "value": [0, 0],
                            "component": [1, 2],
                        },
                    ]
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    else:
        initial_mesh = read_mesh(args.mesh)

    initial = write_modal_initial_fields(
        args.output / "initial", initial_mesh, amplitude, velocity, acceleration, length
    )
    for name, dt in zip(TIME_LEVELS, time_steps):
        steps = round(duration / dt)
        if abs(steps * dt - duration) > 1.0e-12:
            raise ValueError(f"duration {duration} is not divisible by dt={dt}")
        output = args.output / name
        run(args, mode_case(args, material, density, output, run_dirichlet, dt, steps, initial), output)
        print("SFEM_TRANSIENT_LEVEL_COMPLETE")

    print("SFEM_LINEAR_KV_MODE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
