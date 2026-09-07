"""Small coupled plotting check: python3 scripts/test_compare_mr_visco_history.py."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from compare_mr_visco_history import compare_displacement


def check(root):
    mesh = root / "mesh"
    mesh.mkdir()
    np.array([0, 5, 10, 10], dtype=np.float32).tofile(mesh / "x.raw")
    names = ["fp64", "fp32", "fp16", "fp16_tensor", "fp16_element_prony"]
    for i, name in enumerate(names):
        output = root / name / "test_mooney_rivlin_gravity"
        output.mkdir(parents=True)
        np.savetxt(output / "time.txt", [0, 0.025])
        for step in range(2):
            for c in range(3):
                values = np.array([0, -1, -2, -4], dtype=np.float64) if c == 1 else np.zeros(4)
                (values * step * (1 + i * 0.01)).tofile(output / f"disp.{c}.{step}.float64")
    command = [sys.executable, str(Path(__file__).with_name("compare_mr_visco_history.py")),
               "--reference", str(root / "fp64"), "--reference-label", "fp64",
               "--mesh", str(mesh), "--end-time", "0.025", "--out", str(root / "plots")]
    for name in names[1:]:
        command += ["--candidate", str(root / name)]
    subprocess.run(command, check=True)
    data = pd.read_csv(root / "plots/coupled_comparison.csv")
    assert len(data) == 8
    for i, name in enumerate(names[1:], 1):
        last = data[data.candidate == name].iloc[-1]
        assert np.isclose(last.reference_tip_uy, -3)
        assert np.isclose(last.candidate_tip_uy, -3 * (1 + i * 0.01))
        assert np.isclose(last.relative_l2_percent, i)
        assert np.isclose(last.max_abs_diff, 4 * i * 0.01)
    for extension in ("png", "pdf"):
        assert (root / "plots" / f"coupled_comparison.{extension}").stat().st_size > 0
    # Existing single-pair use still works without mesh input.
    single = compare_displacement(root / "fp64", root / "fp32", root / "single",
                                  "test_mooney_rivlin_gravity", "fp64", "fp32")
    assert "candidate_tip_uy" not in single
    np.savetxt(root / "fp32/test_mooney_rivlin_gravity/time.txt", [0, 0.05])
    try:
        compare_displacement(root / "fp64", root / "fp32", root / "bad",
                             "test_mooney_rivlin_gravity", "fp64", "fp32")
    except RuntimeError as error:
        assert "not aligned" in str(error)
    else:
        raise AssertionError("Mismatched output times must be rejected")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="sfem-coupled-plot-") as directory:
        check(Path(directory))
    print("Coupled plotting checks passed")
