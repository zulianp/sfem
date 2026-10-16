"""What the Prony torsion tools read from a playground case: the regime, the torsion of the first
rotation in `constraints` (the one the torque probe watches), and the iso relaxation series."""

from __future__ import annotations

from pathlib import Path

import yaml


class TorsionCase:
    def __init__(self, case: dict):
        rotation = (case.get("constraints") or [{}])[0]
        profile = rotation.get("profile") or {}
        iso = ((case.get("material") or {}).get("relaxation") or {}).get("iso") or {}

        self.dynamic = case.get("regime", "steady") == "dynamic"
        self.creep = "control" in rotation
        self.amplitude = float(rotation.get("angle", 0.0))
        self.ramp_time = float(profile.get("end_time", 0.0)) if profile.get("type") in ("linear_ramp", "ramp") else 0.0
        self.period = float(profile["period"]) if profile.get("type") == "sine" else None
        self.terms = iso.get("prony") or []
        self.wlf = iso.get("wlf")
        self.temperature = iso.get("temperature")


def load(path) -> TorsionCase:
    return TorsionCase(yaml.safe_load(Path(path).read_text()))
