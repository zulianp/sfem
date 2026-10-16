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
