#!/usr/bin/env python3

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

from common.mesh import read_mesh
from common.sets import write_nodeset
from common.transient import TIME_LEVELS, run_checked, write_modal_initial_fields


def modal_initial_acceleration(material, length, amplitude, velocity):
    wave_number = np.pi / float(length)
    axial_modulus = float(material["bulk_modulus"]) + 2.0 * float(material["shear_stiffness"]) / 3.0
    viscous_modulus = 2.0 * float(material["damping"]) / 3.0
    density = float(material["density"])
    omega_0_squared = wave_number**2 * axial_modulus / density
    twice_decay = viscous_modulus * wave_number**2 / density
    return -twice_decay * float(velocity) - omega_0_squared * float(amplitude)


def main():
    parser = argparse.ArgumentParser(description="Run the Kelvin-Voigt mode at three time-step sizes")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--dirichlet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.case.read_text(encoding="utf-8"))
    duration = float(config["time"]["duration"])
    time_steps = [float(value) for value in config["time"]["time_steps"]]
    amplitude = float(config["initial_conditions"]["modal_amplitude"])
    velocity = float(config["initial_conditions"]["modal_velocity"])
    length = float(config["geometry"]["length"])
    acceleration = modal_initial_acceleration(config["material"], length, amplitude, velocity)
    if len(time_steps) != len(TIME_LEVELS):
        raise ValueError("damped-mode verification requires exactly three time-step sizes")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    refine_level = int(os.environ.get("SFEM_ELEMENT_REFINE_LEVEL", "0"))
    run_dirichlet = args.dirichlet
    if refine_level > 1:
        prepare = args.output / "prepared"
        environment = os.environ.copy()
        environment.update({"SFEM_T_END": "0", "SFEM_NEWMARK_ENABLE_OUTPUT": "0"})
        environment.pop("SFEM_INITIAL_DISPLACEMENT", None)
        environment.pop("SFEM_INITIAL_DISPLACEMENT_COMPONENTS", None)
        environment.pop("SFEM_INITIAL_VELOCITY", None)
        environment.pop("SFEM_INITIAL_VELOCITY_COMPONENTS", None)
        environment.pop("SFEM_INITIAL_ACCELERATION", None)
        environment.pop("SFEM_INITIAL_ACCELERATION_COMPONENTS", None)
        run_checked([args.driver, args.mesh, args.dirichlet, "NONE", prepare], environment)
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

    displacement_paths, velocity_paths, acceleration_paths = write_modal_initial_fields(
        args.output / "initial", initial_mesh, amplitude, velocity, acceleration, length
    )

    for name, dt in zip(TIME_LEVELS, time_steps):
        steps = round(duration / dt)
        if abs(steps * dt - duration) > 1.0e-12:
            raise ValueError(f"duration {duration} is not divisible by dt={dt}")
        environment = os.environ.copy()
        environment.update(
            {
                "SFEM_DT": str(dt),
                "SFEM_T_END": str(duration),
                "SFEM_EXPORT_FREQ": "1",
                "SFEM_NEWMARK_ENABLE_OUTPUT": "1",
                "SFEM_INITIAL_DISPLACEMENT_COMPONENTS": displacement_paths,
                "SFEM_INITIAL_VELOCITY_COMPONENTS": velocity_paths,
                "SFEM_INITIAL_ACCELERATION_COMPONENTS": acceleration_paths,
            }
        )
        run_checked(
            [args.driver, args.mesh, run_dirichlet, "NONE", args.output / name],
            environment,
            marker="SFEM_TRANSIENT_LEVEL_COMPLETE",
        )
    print("SFEM_LINEAR_KV_MODE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
