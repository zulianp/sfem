#!/usr/bin/env python3
"""Summarize paired accumulation timings without discarding negative differences."""

import argparse
import math
from pathlib import Path
import statistics as stats


def read_case(path):
    lines = path.read_text().splitlines()
    configs = [line for line in lines if line.startswith("scatter_config ")]
    if len(configs) != 1 or lines.count("scatter_gate PASS") != 1 or lines.count("scatter_study PASS") != 1:
        raise ValueError(f"{path}: incomplete or failed study")
    config = dict(item.split("=") for item in configs[0].split()[1:])
    expected = int(config["samples"])
    if expected < 6 or expected % 6:
        raise ValueError(f"{path}: unbalanced sample count")
    rows = []
    for line in lines:
        if line.startswith("scatter_sample "):
            _, index, normal, full, bypass = line.split()
            values = tuple(map(float, (normal, full, bypass)))
            if int(index) != len(rows) or not all(math.isfinite(x) and x > 0 for x in values):
                raise ValueError(f"{path}: invalid timing row")
            rows.append(values)
    if len(rows) != expected:
        raise ValueError(f"{path}: missing samples")
    return config, rows


def summarize(paths, expect_full=False):
    cases = {}
    for path in paths:
        config, rows = read_case(path)
        key = (int(config["threads"]), int(config["ho"]), config["layout"])
        if key in cases:
            raise ValueError(f"duplicate case {key}")
        cases[key] = (config, rows)
    if not cases:
        raise ValueError("no study logs")
    if expect_full:
        threads = {key[0] for key in cases}
        expected = {(t, h, l) for t in (1, 72) for h in (0, 1) for l in ("packed", "atomic")}
        if set(cases) != expected or threads != {1, 72}:
            raise ValueError("expected packed/atomic, first-order/Darwish at 1 and 72 threads")
        if any(int(c["limiter"]) != 3 for (t, h, l), (c, _) in cases.items() if h):
            raise ValueError("expected Darwish limiter=3")
    print("Times are ms per residual sweep; delta = full probe - bypass probe.")
    print("The range is sample variation, not a confidence interval. Buffer clearing remains in both arms.")
    print("\n| Threads | Scheme | Layout | Normal | Full probe | Bypass | Delta | Delta range | Delta/full | Probe overhead | Positive pairs |")
    print("|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|")
    means = {}
    for key, (config, rows) in sorted(cases.items()):
        normal, full, bypass = (stats.mean(row[i] for row in rows) for i in range(3))
        delta = [f - b for _, f, b in rows]
        overhead = 100 * (full / normal - 1)
        means[key] = (normal, full, bypass)
        t, h, layout = key
        scheme = "first-order" if not h else f"limiter {config['limiter']}"
        print(f"| {t} | {scheme} | {layout} | {normal*1000:.5f} | {full*1000:.5f} | "
              f"{bypass*1000:.5f} | {stats.mean(delta)*1000:.5f} | "
              f"[{min(delta)*1000:.5f}, {max(delta)*1000:.5f}] | "
              f"{100*stats.mean(delta)/full:.2f}% | {overhead:+.2f}% | "
              f"{sum(x > 0 for x in delta)}/{len(delta)} |")
    print("\nWithin the probe, atomic-minus-packed full time equals the bypass gap plus the difference in accumulation deltas.")
    print("| Threads | HO | Normal gap (ms) | Probe full gap | Bypass gap | Difference in deltas |")
    print("|---:|---:|---:|---:|---:|---:|")
    for t, h in sorted({(t, h) for t, h, _ in cases}):
        if (t, h, "atomic") not in means or (t, h, "packed") not in means:
            continue
        a, p = means[t, h, "atomic"], means[t, h, "packed"]
        ac, pc = cases[t, h, "atomic"][0], cases[t, h, "packed"][0]
        for field in ("elements", "nodes", "limiter", "repeat", "warmup", "samples"):
            if ac[field] != pc[field]:
                raise ValueError(f"incompatible cases at {t} threads: {field}")
        normal, full, bypass = (1000*(a[i]-p[i]) for i in range(3))
        print(f"| {t} | {h} | {normal:.5f} | {full:.5f} | {bypass:.5f} | {full-bypass:.5f} |")
    print("\nThese are marginal costs in the instrumented sweep, including memory and synchronization effects.")
    print("Inspect probe overhead and sample variation before relating them to normal execution. A small overhead alone does not prove identical scheduling.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--expect-full", action="store_true")
    args = parser.parse_args()
    try:
        summarize(args.logs, args.expect_full)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"scatter analysis failed: {error}\n")
