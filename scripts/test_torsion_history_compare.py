"""Run with: venv/bin/python scripts/test_torsion_history_compare.py."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

from run_torsion_history_compare import (
    CASE, POLICIES, RESOLUTIONS, collect_runs, compare_runs, load_case, plot_results, read_history, run, scalar_errors,
)


def check(root):
    assert RESOLUTIONS == {"coarse": (41, 9, 9), "medium": (81, 17, 17), "fine": (161, 33, 33)}
    for setting, resolution in zip((None, "0", "1", "1"), (None, "coarse", "medium", "fine")):
        environment = dict(os.environ)
        environment.update(PRONY_NX="16", PRONY_NY="5", PRONY_NZ="5")
        environment.pop("SFEM_HISTORY_CHECK", None)
        if setting is not None:
            environment["SFEM_HISTORY_CHECK"] = setting
        expected = "1" if setting is None else setting
        folder = root / f"diagnostics_{setting}_{resolution}"
        with patch.dict(os.environ, environment, clear=True), \
             patch("run_torsion_history_compare.subprocess.run") as launch, \
             patch("run_torsion_history_compare.subprocess.check_output", return_value="test-commit"), \
             patch("run_torsion_history_compare.read_history"), \
             patch("run_torsion_history_compare.compare_runs"):
            launch.return_value.returncode = 0
            run(folder, Path(sys.executable), 0.025, resolution)
            assert launch.call_count == 1 + len(POLICIES)
            command = launch.call_args_list[0].args[0]
            nodes = tuple(int(command[command.index(flag) + 1]) for flag in ("-x", "-y", "-z"))
            assert nodes == (RESOLUTIONS[resolution] if resolution else (16, 5, 5))
            assert all(call.kwargs["env"]["SFEM_HISTORY_CHECK"] == expected
                       for call in launch.call_args_list)
        assert json.loads((folder / "manifest.json").read_text())["history_check"] == expected
        assert json.loads((folder / "manifest.json").read_text())["resolution"] == resolution

    with patch("run_torsion_history_compare.subprocess.run") as launch, \
         patch("run_torsion_history_compare.subprocess.check_output", return_value="test-commit"), \
         patch("run_torsion_history_compare.read_history"), \
         patch("run_torsion_history_compare.compare_runs") as compare:
        launch.return_value.returncode = 0
        selected = root / "selected_element"
        run(selected, Path(sys.executable), 0.025, "coarse", "per_elem", ["fp16_tensor"], True)
        assert launch.call_count == 2 and not compare.called
        assert launch.call_args.kwargs["env"]["SFEM_HISTORY_MODE"] == "per_elem"
        assert launch.call_args.kwargs["env"]["SFEM_HISTORY_SCALING"] == "tensor"
        saved = json.loads((selected / "manifest.json").read_text())
        assert saved["requested_cases"] == ["fp16_tensor"]
        assert saved["runs"]["fp16_tensor"]["complete"]

    for mode, cases in (("wrong", ["fp64"]), ("per_qp", ["fp64", "fp64"]), ("per_elem", [])):
        try:
            run(root / "invalid", Path(sys.executable), history_mode=mode, cases=cases, run_only=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Reject invalid mode/cases before launching")

    original = yaml.safe_load(CASE.read_text())
    assert original["time"] == {"dt": 0.005, "t_end": 30.0}
    assert original["dynamics"] == {"type": "newmark", "density": 1.0e-5, "beta": 0.64, "gamma": 0.6}
    assert original["torsion"]["angle"] == 5.0
    material = original["material"]
    weights = [term["g"] for term in material["prony"]]
    np.testing.assert_allclose(weights, np.array([0.08, 0.08, 0.06, 0.02]) / 1.24, rtol=1e-14)
    assert [term["tau"] for term in material["prony"]] == [2.09, 13.03, 117.80, 349.87]
    np.testing.assert_allclose(
        np.array([material["C10"], material["C01"]]) * (1 - sum(weights)),
        [1.555, 0.05], rtol=1e-14)
    assert material["K"] == 190.0
    shortened = load_case(0.025)
    assert shortened["time"]["t_end"] == 0.025
    shortened["time"]["t_end"] = original["time"]["t_end"]
    assert shortened == original == load_case()
    for invalid in (0, -1, float("nan"), float("inf"), 0.007, 30.005):
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

    # Only compact CSV/JSON files travel to the local machine; plotting cannot read raw fields.
    portable = root / "portable"
    (portable / "compare").mkdir(parents=True)
    for path in (root / "compare").iterdir():
        if path.suffix in (".csv", ".json"):
            shutil.copy2(path, portable / "compare" / path.name)
    plot_results(portable)
    assert (portable / "compare/torsion_comparison.png").is_file()

    # Non-overlapping debug jobs can be combined, but not different grids, YAMLs or failed jobs.
    split = []
    for index, names in enumerate((list(POLICIES)[:2], list(POLICIES)[2:])):
        job = root / f"split_{index}"
        job.mkdir()
        shutil.copy2(root / "case.yaml", job / "case.yaml")
        for name in names:
            (job / name).symlink_to(root / name, target_is_directory=True)
        manifest = {"history_mode": "per_qp", "mesh_command": ["python", "mesh.py", "mesh", "--cell_type=HEX8", "-x", "3"],
                    "executable_sha256": "same-executable", "requested_cases": names,
                    "runs": {name: {"storage": POLICIES[name][0], "scaling": POLICIES[name][1],
                                    "returncode": 0, "complete": True} for name in names}}
        (job / "manifest.json").write_text(json.dumps(manifest))
        split.append(job)
    compare_runs(root / "merged", split, make_plots=False)
    assert not (root / "merged/compare/torsion_comparison.png").exists()
    merged = pd.read_csv(root / "merged/compare/torsion_summary.csv")
    assert np.allclose(merged.peak_control_error_percent_of_ref_peak, summary.peak_control_error_percent_of_ref_peak)
    try:
        compare_runs(root / "merged", split, make_plots=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Preserve completed comparison tables")
    try:
        collect_runs([split[0], split[0]])
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("Reject duplicate case jobs")
    path_manifest = split[1] / "manifest.json"
    good_manifest = json.loads(path_manifest.read_text())
    for mutation, message in (("mesh", "mesh"), ("failed", "failed"), ("missing", "incomplete")):
        broken = json.loads(json.dumps(good_manifest))
        if mutation == "mesh":
            broken["mesh_command"][-1] = "4"
        elif mutation == "failed":
            broken["runs"]["fp16"]["returncode"] = 1
        else:
            del broken["runs"]["fp16"]
        path_manifest.write_text(json.dumps(broken))
        try:
            collect_runs(split)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"Reject {mutation}")
    path_manifest.write_text(json.dumps(good_manifest))
    element = root / "element_fixture"
    shutil.copytree(split[0], element, symlinks=True)
    element_manifest = json.loads((element / "manifest.json").read_text())
    element_manifest["history_mode"] = "per_elem"
    (element / "manifest.json").write_text(json.dumps(element_manifest))
    _, modes, reference = collect_runs([split[0], element])
    assert reference == "per_qp_fp64" and "per_elem_fp64" in modes
    _, _, reference = collect_runs([element])
    assert reference == "fp64"

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
