#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys
from typing import List, Optional, Tuple

import numpy as np


def _find_component_files(folder: str, base: str, component: int) -> List[str]:
    pattern = os.path.join(folder, f"{base}.{component}.*.raw")
    files = glob.glob(pattern)
    if not files:
        return []

    def _step_idx(p: str):
        name = os.path.basename(p)
        parts = name.split(".")
        # Expected: base.component.step.raw
        if len(parts) >= 3:
            try:
                return int(parts[2])
            except Exception:
                return name
        return name

    return sorted(files, key=_step_idx)


def _read_times(folder: str) -> Optional[np.ndarray]:
    path = os.path.join(folder, "time.txt")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            ts = [float(line.strip()) for line in f if line.strip()]
        return np.array(ts, dtype=np.float64)
    except Exception:
        return None


def load_series(folder: str, base: str, block_size: int, dtype: np.dtype) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    comp_files = []
    for c in range(block_size):
        files = _find_component_files(folder, base, c)
        if not files:
            raise FileNotFoundError(f"no files found for component {c}: {os.path.join(folder, f'{base}.{c}.*.raw')}")
        comp_files.append(files)

    # Ensure equal number of timesteps across components
    num_steps = len(comp_files[0])
    for c in range(1, block_size):
        if len(comp_files[c]) != num_steps:
            raise RuntimeError(f"component {c} step count {len(comp_files[c])} != component 0 step count {num_steps}")

    series: List[np.ndarray] = []
    for i in range(num_steps):
        comps = []
        for c in range(block_size):
            arr = np.fromfile(comp_files[c][i], dtype=dtype)
            comps.append(arr)
        # Stack as (3N,) AoS-flattened order [x0,y0,z0,x1,y1,z1,...] for consistent L2
        U = np.stack(comps, axis=1).reshape(-1)
        series.append(U)

    times = _read_times(folder)
    if times is not None and len(times) != num_steps:
        # Length mismatch: ignore times
        times = None

    return series, times


def l2_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.dot(x, x)))


def compare_series(
    kv_folder: str,
    le_folder: str,
    base: str = "disp",
    block_size: int = 3,
    dtype: str = "float64",
    eps: float = 1e-30,
    out_csv: Optional[str] = None,
) -> int:
    np_dtype = np.float64 if dtype == "float64" else np.float32

    kv_series, kv_times = load_series(kv_folder, base, block_size, np_dtype)
    le_series, le_times = load_series(le_folder, base, block_size, np_dtype)

    n_steps = min(len(kv_series), len(le_series))
    if n_steps == 0:
        print("No common steps to compare.")
        return 1

    writer = None
    f_csv = None
    if out_csv:
        f_csv = open(out_csv, "w", newline="")
        writer = csv.writer(f_csv)
        writer.writerow(["step", "time", "abs_l2", "rel_l2", "kv_l2", "le_l2"])  # header

    max_rel = -1.0
    max_rel_step = -1

    for i in range(n_steps):
        u_kv = kv_series[i]
        u_le = le_series[i]
        if u_kv.shape != u_le.shape:
            print(f"[WARN] shape mismatch at step {i}: kv={u_kv.shape}, le={u_le.shape}")
            n = min(u_kv.size, u_le.size)
            u_kv = u_kv[:n]
            u_le = u_le[:n]

        diff = u_kv - u_le
        abs_l2 = l2_norm(diff)
        kv_l2 = l2_norm(u_kv)
        le_l2 = l2_norm(u_le)
        rel_l2 = abs_l2 / max(le_l2, eps)

        t = None
        if kv_times is not None and le_times is not None and i < len(kv_times) and i < len(le_times):
            # Prefer LE time if both exist and equal; else leave numeric index
            t = le_times[i] if abs(kv_times[i] - le_times[i]) < 1e-12 else kv_times[i]

        print(f"step {i+1:4d}: abs={abs_l2:.6e}, rel={rel_l2:.6e}, kv_l2={kv_l2:.6e}, le_l2={le_l2:.6e}" + (f", t={t:.6g}" if t is not None else ""))

        if writer is not None:
            writer.writerow([i + 1, (None if t is None else f"{t:.16g}"), f"{abs_l2:.16e}", f"{rel_l2:.16e}", f"{kv_l2:.16e}", f"{le_l2:.16e}"])

        if rel_l2 > max_rel:
            max_rel = rel_l2
            max_rel_step = i + 1

    if f_csv is not None:
        f_csv.close()

    print(f"max rel: {max_rel:.6e} at step {max_rel_step}")
    if out_csv:
        print(f"CSV written to: {out_csv}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare transient displacement series between two result folders (e.g., KV eta=0 vs Linear Elasticity)")
    p.add_argument("--kv-dir", required=True, help="Folder of KV results (contains disp.0.*.raw, disp.1.*.raw, disp.2.*.raw and time.txt)")
    p.add_argument("--le-dir", required=True, help="Folder of LE results")
    p.add_argument("--base", default="disp", help="Base name of point_data (default: disp)")
    p.add_argument("--block-size", type=int, default=3, help="Number of vector components (default: 3)")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64", help="Raw dtype (default: float64)")
    p.add_argument("--eps", type=float, default=1e-30, help="Stabilizer for relative error (default: 1e-30)")
    p.add_argument("--out-csv", default=None, help="Optional CSV output path")
    args = p.parse_args(argv)

    return compare_series(
        kv_folder=args.kv_dir,
        le_folder=args.le_dir,
        base=args.base,
        block_size=args.block_size,
        dtype=args.dtype,
        eps=args.eps,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    sys.exit(main())


