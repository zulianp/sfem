#!/usr/bin/env python3
"""The pump chamber's pressure over time, from the FP70 FSI input, as a two-column table.

usage: pressure_table.py <FP70_MembranePressureInputFile.h5> <out.csv>

The HDF5 file (utopia's 2023 FP70b case) holds one dataset of shape (2, n): times (s) and the
pressure on the membrane (Pa, gauge). Its 0.18 s are four strokes of 45 ms: about +1.03 bar on
discharge and -0.13 bar on suction. Read with h5py when it is there, with h5dump otherwise.
The table is a `tabulated` load profile (time, value), comma separated.
"""

import shutil
import subprocess
import sys


def read_h5py(path):
    import h5py

    with h5py.File(path, "r") as f:
        data = f["dataset"][()]
    return [float(v) for v in data[0]], [float(v) for v in data[1]]


def read_h5dump(path):
    tool = shutil.which("h5dump")
    if not tool:
        sys.exit("pressure_table.py: neither h5py nor h5dump is available")
    text = subprocess.run([tool, "-d", "/dataset", "-y", "-w", "0", path], check=True, capture_output=True, text=True).stdout
    start = text.index("DATA {") + len("DATA {")
    body = text[start : text.index("}", start)]
    values = [float(v) for v in body.replace("\n", " ").split(",") if v.strip()]
    n = len(values) // 2
    if 2 * n != len(values):
        sys.exit("pressure_table.py: %s does not hold two rows" % path)
    return values[:n], values[n:]


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    try:
        times, pressures = read_h5py(argv[1])
    except ImportError:
        times, pressures = read_h5dump(argv[1])
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        sys.exit("pressure_table.py: the times must increase")
    with open(argv[2], "w") as out:
        out.write("# time (s), chamber pressure (Pa, gauge); from %s\n" % argv[1].rsplit("/", 1)[-1])
        for t, p in zip(times, pressures):
            out.write("%.9g, %.9g\n" % (t, p))
    print("%d rows, t in [%g, %g] s, p in [%g, %g] Pa" % (len(times), times[0], times[-1], min(pressures), max(pressures)))


if __name__ == "__main__":
    main(sys.argv)
