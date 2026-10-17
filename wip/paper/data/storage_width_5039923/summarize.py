#!/usr/bin/env python3
"""Read every paired phase sample without filtering or baseline subtraction."""
import json
import math
import statistics
from pathlib import Path


def summarize(root):
    result = {}
    for threads in (1, 72):
        rows = {"graph": [], "geometry": []}
        validations = []
        metadata = []
        complete = 0
        for line in (root / ("timed-T%d.log" % threads)).read_text().splitlines():
            fields = line.split(",")
            if fields[0] == "metadata":
                metadata.append(dict(item.split("=", 1) for item in fields[1:]))
            elif fields[0] == "validation":
                values = dict(item.split("=", 1) for item in fields[2:])
                error = float(values["geometry_max_relative"])
                assert values["graph_bit_equal"] == "1"
                assert math.isfinite(error) and 0 <= error <= 2e-7
                validations.append(fields[1])
            elif fields[0] == "probe_complete":
                assert fields == ["probe_complete", "1"]
                complete += 1
            elif fields[0].isdigit():
                sample, phase, first, wide, narrow, ratio = fields
                sample = int(sample)
                wide, narrow, ratio = map(float, (wide, narrow, ratio))
                assert all(math.isfinite(x) and x > 0 for x in (wide, narrow, ratio))
                assert math.isclose(wide/narrow, ratio, rel_tol=1e-12)
                assert first == ("narrow" if sample % 2 else "wide")
                rows[phase].append((sample, first, wide*1000, narrow*1000, ratio))
        assert complete == 1 and len(metadata) == 1
        assert metadata[0] == dict(n="128", pack="1024", threads=str(threads),
            elements="2097152", ghost_entries="673407", rows="561177", ordering="SFC")
        assert validations == ["cube", "sheared_affine_at_coordinate_precision", "timed_cube", "after_timing"]
        for phase, samples in rows.items():
            assert [x[0] for x in samples] == list(range(12))
            ratios = [x[4] for x in samples]
            entry = dict(samples=12, wide_ms_mean=statistics.mean(x[2] for x in samples),
                narrow_ms_mean=statistics.mean(x[3] for x in samples),
                ratio_mean=statistics.mean(ratios), ratio_median=statistics.median(ratios),
                ratio_range=[min(ratios), max(ratios)],
                narrow_faster_pairs=sum(x > 1 for x in ratios))
            for first in ("wide", "narrow"):
                subset = [x for x in samples if x[1] == first]
                entry[first + "_first"] = dict(samples=len(subset),
                    ratio_mean=statistics.mean(x[4] for x in subset),
                    ratio_median=statistics.median(x[4] for x in subset),
                    narrow_faster_pairs=sum(x[4] > 1 for x in subset))
            result["T%d_%s" % (threads, phase)] = entry
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(Path(__file__).resolve().parent), indent=2, sort_keys=True))
