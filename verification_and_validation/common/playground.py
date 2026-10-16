"""Run the SFEM elasticity playground on a case composed from a V&V case's inputs."""

from pathlib import Path
import subprocess

import yaml


def _load(path, key):
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return document[key] if key in document else document


def static_case(mesh, operator, output, dirichlet=None, neumann=None):
    """A steady case: the mesh, the operator YAML, the conditions' YAML files, and output that
    exports the displacement and the material reaction and writes quantities.yaml."""
    case = {
        "mesh": {"path": str(mesh)},
        "material": _load(operator, "operator"),
        "regime": "steady",
        "output": {"path": str(output), "fields": ["material_reaction"], "quantities": True},
    }
    if dirichlet is not None:
        case["dirichlet_conditions"] = _load(dirichlet, "dirichlet_conditions")
    if neumann is not None:
        case["loads"] = {"neumann": _load(neumann, "neumann_conditions")}
    return case


def load_ramp_case(mesh, material, output, dirichlet, neumann, dt, steps, ramp, export_freq=1):
    """A quasi-static case whose Neumann loads ramp linearly to their full value over `ramp`
    seconds (none: at once), in `steps` steps of `dt`. `material` is the operator YAML as a dict.
    Exports the displacement every `export_freq` steps and at the last."""
    profile = (
        {"type": "linear_ramp", "start_time": 0, "end_time": ramp, "start_value": 0, "end_value": 1}
        if ramp
        else {"type": "constant", "value": 1}
    )
    return {
        "mesh": {"path": str(mesh), "packed": False},
        "material": material,
        "dirichlet_conditions": {"file": str(dirichlet)},
        "loads": {"neumann": {"file": str(neumann)}, "profile": profile},
        "regime": "quasi_static",
        "time": {"dt": dt, "steps": steps},
        "output": {"path": str(output), "export_freq": export_freq},
    }


def material(operator, parameters):
    """The operator YAML for `operator` with `parameters`, a list of name=value."""
    document = {"type": operator}
    for parameter in parameters:
        name, value = parameter.split("=", 1)
        document[name] = float(value)
    return document


def run(executable, case, case_path, overrides, environment):
    """Writes `case` to `case_path` and runs the playground on it with `overrides` (key=value)."""
    case_path = Path(case_path)
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    command = [str(executable), str(case_path)]
    for override in overrides:
        command += ["--set", override]
    return subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
