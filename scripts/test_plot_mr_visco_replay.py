"""Run with: python3 scripts/test_plot_mr_visco_replay.py."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from plot_mr_visco_replay import load_cases, plot_cases


def check(root):
    runs = [root / f"job_{i}" for i in range(3)]
    labels = ["float64_none", "float32_none", "float16_none",
              "float16_tensor", "float16_element_prony"]
    for i, label in enumerate(labels):
        path = runs[i // 2] / f"per_qp_{label}" / "test_mooney_rivlin_gravity"
        path.mkdir(parents=True)
        error = [0, i * 0.002, i * 0.001]
        pd.DataFrame({"time": [0, 1.5, 3], "max_abs_error": error,
                      "l2_error": error, "relative_l2_error": error}).to_csv(
            path / "history_replay.csv", index=False)
    cases = load_cases(runs, end_time=3)
    assert len(cases) == 5
    for log_scale in (False, True):
        summary = plot_cases(cases, root / "plots", log_scale)
        last = summary.iloc[-1]
        assert last["peak_relative_l2_percent"] == 0.8
        assert last["final_relative_l2_percent"] == 0.4
        assert last["peak_max_abs_time"] == 1.5
        stem = "replay_history_error_log" if log_scale else "replay_history_error"
        for extension in ("png", "pdf"):
            assert (root / "plots" / f"{stem}.{extension}").stat().st_size > 0
    for dirs, end_time in ((runs + [runs[0]], 3), (runs, 6)):
        try:
            load_cases(dirs, end_time)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate/incomplete runs must be rejected")
    path, data = next(iter(cases.values()))
    data.loc[1, "time"] = 1
    data.to_csv(path, index=False)
    try:
        load_cases(runs, 3)
    except ValueError as error:
        assert "Time grids differ" in str(error)
    else:
        raise AssertionError("Mismatched time grids must be rejected")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="sfem-replay-plot-") as directory:
        check(Path(directory))
    print("Replay plotting checks passed")
