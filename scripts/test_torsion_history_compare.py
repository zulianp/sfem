"""Run with: venv/bin/python scripts/test_torsion_history_compare.py."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

from run_torsion_history_compare import CASE, POLICIES, compare_runs, load_case, read_history, run, scalar_errors


def check(root):
    original = yaml.safe_load(CASE.read_text())
    assert original["dynamics"] == {"type": "newmark", "density": 1.0, "beta": 0.64, "gamma": 0.6}
    assert original["material"]["prony"] == [
        {"g": 0.4, "tau": 1}, {"g": 0.4, "tau": 2}, {"g": 0.1, "tau": 5}, {"g": 0.05, "tau": 10}]
    shortened = load_case(0.025)
    assert shortened["time"]["t_end"] == 0.025
    shortened["time"]["t_end"] = original["time"]["t_end"]
    assert shortened == original == load_case()
    for invalid in (0, -1, float("nan"), float("inf"), 0.007, 81):
        try:
            load_case(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Must reject invalid end time: {invalid}")
    case = {"time": {"dt": 1, "t_end": 2}, "solver": {"newton": {"tol": 1e-8}},
            "dynamics": original["dynamics"],
            "output": {"path": "results_newmark", "history_csv": "history.csv"},
            "torsion": {"release": {"time": 2}}}
    (root / "case.yaml").write_text(yaml.safe_dump(case))
    ref = pd.DataFrame({"time": [1, 2], "angle": [0.6, 0], "released": [0, 1],
                        "torque": [2, 0], "ux": [0, 0], "uy": [3, 1], "uz": [4, 0],
                        "newton_it": [2, 3], "lin_it": [10, 20], "gnorm": [1e-10, 1e-11]})
    candidate = ref.copy()
    candidate["torque"] += 0.02
    candidate["ux"] += 0.05
    errors = scalar_errors(ref, candidate)
    # Finite even after release (reference torque is zero); normalize by the peak.
    assert np.allclose(errors.torque_peak_normalized_percent, 1)
    assert np.allclose(errors.control_peak_normalized_percent, 1)

    for i, name in enumerate(POLICIES):
        folder = root / name / "results_newmark"
        (folder / "out").mkdir(parents=True)
        data = ref.copy()
        data["torque"] += i * 0.02
        data["ux"] += i * 0.05
        data.to_csv(folder / "history.csv", index=False)
        np.savetxt(folder / "out/time.txt", [0, 1, 2])
        for frame in range(3):
            for component in range(3):
                values = np.array([1, 2], dtype=np.float64) * frame * (1 + i * 0.01)
                values.tofile(folder / "out" / f"disp.{component}.{frame}.float64")
    compare_runs(root)
    summary = pd.read_csv(root / "compare/torsion_summary.csv")
    assert np.allclose(summary.peak_torque_error_percent_of_ref_peak, range(5))
    assert np.allclose(summary.peak_control_error_percent_of_ref_peak, range(5))
    for name in ("torsion_comparison.pdf", "coupled_comparison.pdf", "torsion_errors.csv"):
        assert (root / "compare" / name).stat().st_size > 0

    path = root / "fp64/results_newmark/history.csv"
    for bad, message in ((ref.iloc[:1], "incomplete"),
                         (ref.assign(gnorm=1e-5), "unconverged"),
                         (ref.assign(torque=np.nan), "non-finite"),
                         (ref.assign(released=0), "release schedule")):
        bad.to_csv(path, index=False)
        try:
            read_history(path, case)
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError(f"Must reject {message}")
    try:
        run(root, Path("does-not-matter"))
    except FileExistsError:
        pass
    else:
        raise AssertionError("Must preserve existing results")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="sfem-torsion-compare-") as directory:
        check(Path(directory))
    print("Torsion comparison checks passed")
