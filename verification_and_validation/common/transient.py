"""Shared artifacts and strict result handling for transient solid cases."""

from pathlib import Path
import glob
import shutil
import subprocess

import numpy as np
import yaml

from .geometry import box_mesh, rectangle_mesh
from .mesh import read_mesh, write_mesh
from .raw import write_raw
from .sets import (
    boundary_sides,
    nodeset_from_sideset,
    select_boundary_axis,
    write_nodeset,
    write_sideset,
)


TIME_LEVELS = ("coarse", "medium", "fine")


def generate_bar_mesh(output, element_type, nx, ny, nz=None, length=1.0):
    """Generate an axis-aligned bar and the sets used by transient cases."""

    output = Path(output)
    if output.exists():
        shutil.rmtree(output)

    element_type = str(element_type).upper()
    if element_type in ("TRI3", "QUAD4"):
        mesh = rectangle_mesh(length, 1.0, int(nx), int(ny), element_type=element_type)
    elif element_type in ("TET4", "HEX8"):
        if nz is None:
            raise ValueError("three-dimensional bars require nz")
        mesh = box_mesh(length, 1.0, 1.0, int(nx), int(ny), int(nz), element_type=element_type)
    else:
        raise ValueError(f"unsupported transient bar element: {element_type}")

    write_mesh(output, mesh)
    left = select_boundary_axis(mesh, 0, 0.0)
    right = select_boundary_axis(mesh, 0, float(length))
    exterior = boundary_sides(mesh)
    sets = output / "sets"
    write_sideset(sets / "left", mesh, left)
    write_sideset(sets / "right", mesh, right)
    write_nodeset(sets / "left.int32.raw", nodeset_from_sideset(mesh, left))
    write_nodeset(sets / "right.int32.raw", nodeset_from_sideset(mesh, right))
    write_nodeset(sets / "boundary.int32.raw", nodeset_from_sideset(mesh, exterior))
    write_nodeset(sets / "all.int32.raw", np.arange(mesh.n_points, dtype=np.int64))

    metadata = {
        "element_type": mesh.element_type,
        "dimension": mesh.dimension,
        "resolution": [int(nx), int(ny)] + ([] if nz is None else [int(nz)]),
        "length": float(length),
        "n_points": mesh.n_points,
        "n_elements": mesh.n_elements,
    }
    (output / "generation.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return mesh


def write_modal_initial_fields(output, mesh, amplitude, velocity, acceleration, length=1.0):
    """Write full-node component files for the axial sine mode."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    shape = np.sin(np.pi * mesh.points[:, 0] / float(length))
    displacement_paths = []
    velocity_paths = []
    acceleration_paths = []
    for component in range(mesh.dimension):
        displacement = float(amplitude) * shape if component == 0 else np.zeros(mesh.n_points)
        initial_velocity = float(velocity) * shape if component == 0 else np.zeros(mesh.n_points)
        initial_acceleration = float(acceleration) * shape if component == 0 else np.zeros(mesh.n_points)
        displacement_path = output / f"displacement.{component}.float64.raw"
        velocity_path = output / f"velocity.{component}.float64.raw"
        acceleration_path = output / f"acceleration.{component}.float64.raw"
        write_raw(displacement_path, displacement, np.float64, require_finite=True)
        write_raw(velocity_path, initial_velocity, np.float64, require_finite=True)
        write_raw(acceleration_path, initial_acceleration, np.float64, require_finite=True)
        displacement_paths.append(str(displacement_path))
        velocity_paths.append(str(velocity_path))
        acceleration_paths.append(str(acceleration_path))
    return ",".join(displacement_paths), ",".join(velocity_paths), ",".join(acceleration_paths)


def require_time_history(folder, expected_times):
    folder = Path(folder)
    time_path = folder / "time.txt"
    if not time_path.is_file():
        raise FileNotFoundError(f"missing transient time history: {time_path}")
    times = np.loadtxt(time_path, dtype=np.float64, ndmin=1)
    expected_times = np.asarray(expected_times, dtype=np.float64)
    if times.shape != expected_times.shape:
        raise ValueError(f"{time_path} contains {len(times)} times; expected {len(expected_times)}")
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"time history is not strictly increasing: {time_path}")
    if not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-11):
        raise ValueError(f"time history is incomplete or has incorrect sample times: {time_path}")
    return times


def read_field_history(folder, name, dimension, node_count, expected_times):
    """Read one vector history and reject missing or extra component states."""

    folder = Path(folder)
    times = require_time_history(folder, expected_times)
    components = []
    for component in range(int(dimension)):
        paths = sorted(glob.glob(str(folder / f"{name}.{component}.*.float64")))
        if len(paths) != len(times):
            raise ValueError(
                f"{folder} contains {len(paths)} {name} states for component {component}; expected {len(times)}"
            )
        values = [np.fromfile(path, dtype=np.float64) for path in paths]
        if any(value.shape != (node_count,) or not np.all(np.isfinite(value)) for value in values):
            raise ValueError(f"invalid {name} history for component {component} in {folder}")
        components.append(np.asarray(values))
    return times, np.stack(components, axis=-1)


def run_checked(command, environment, marker=None):
    completed = subprocess.run(
        [str(value) for value in command],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode:
        raise RuntimeError(f"transient driver exited with status {completed.returncode}")
    if marker:
        print(marker)
    return completed.stdout


def load_generated_mesh(folder):
    return read_mesh(Path(folder))
