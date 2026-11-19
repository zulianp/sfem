#!/usr/bin/env python3
"""The pressure difference across each LDP24A valve over an ldp_fsi stroke, as the strokes' loads.

    valve_pressure.py <run> <out> [--strokes 4] [--period 0.0413793] [--outlet 1e5] [--ramp 0.003]

<run> is an ldp_fsi run folder with fluid frames (fluid/mesh, fluid/out/p.<k>.float32, time.txt;
MARS's raw format). Per frame, the pressure (Pa, gauge) is averaged over the fluid nodes of five
boxes (m), clear of the bodies:
  chamber   |x|, |z| < 6e-3,            -5.0e-3 < y < -3.5e-3  (diaphragm face -5.2, ceiling -3.3)
  pv_below  -18e-3 < x < -15.5e-3, |z| < 2e-3, -3.5e-3 < y < -1e-3  (the pressure valve's bore, chamber side)
  pv_above  -14e-3 < x < -10e-3,  |z| < 2e-3,  9e-3 < y < 12e-3    (its outlet plenum, y 6.9-16.1 mm)
  sv_below    8e-3 < x < 16e-3,   |z| < 2e-3, -4.6e-3 < y < -2.5e-3 (under the suction valve's dome)
  sv_above   10e-3 < x < 14e-3,   |z| < 2e-3,  9e-3 < y < 12e-3    (its suction channel)
Both valves' discs (PV_TOP, SV_TOP) take the difference below minus above, positive upward: it opens
the pressure valve and closes the suction valve. The run started from simulation 1, whose outlet
load on the pressure valve faded out over --ramp while the fluid's outlet pressure ramped in; the
pressure valve's load keeps that fade, -outlet (1 - t/ramp), so it starts where simulation 1 left
it. The stroke's load is repeated --strokes times, fade included (the fluid's outlet ramp and the
fading preload add up to the outlet's load at each stroke's start); the frame at rest, t = 0, only
opens the first.

Writes into <out> (a new folder; stops if it exists): fsi_stroke.csv (t, the five pressures) and
pv_load.csv, sv_load.csv (t, value: the tabulated profiles of sim2.yaml).
"""

import argparse
import os
import sys

import numpy as np

BOXES = {
    "chamber": (-6e-3, 6e-3, -5.0e-3, -3.5e-3, -6e-3, 6e-3),
    "pv_below": (-18e-3, -15.5e-3, -3.5e-3, -1e-3, -2e-3, 2e-3),
    "pv_above": (-14e-3, -10e-3, 9e-3, 12e-3, -2e-3, 2e-3),
    "sv_below": (8e-3, 16e-3, -4.6e-3, -2.5e-3, -2e-3, 2e-3),
    "sv_above": (10e-3, 14e-3, 9e-3, 12e-3, -2e-3, 2e-3),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("out")
    p.add_argument("--strokes", type=int, default=4)
    p.add_argument("--period", type=float, default=0.0413793)
    p.add_argument("--outlet", type=float, default=1e5)
    p.add_argument("--ramp", type=float, default=0.003)
    a = p.parse_args()
    if os.path.exists(a.out):
        sys.exit("valve_pressure: %s exists; give a new folder" % a.out)
    fluid = os.path.join(a.run, "fluid")
    x, y, z = (np.fromfile(os.path.join(fluid, "mesh", c + ".float32"), dtype=np.float32) for c in "xyz")
    sel = {k: (x >= b[0]) & (x <= b[1]) & (y >= b[2]) & (y <= b[3]) & (z >= b[4]) & (z <= b[5])
           for k, b in BOXES.items()}
    for k, s in sel.items():
        if not s.any():
            sys.exit("valve_pressure: no fluid node in box %s" % k)
    times = [float(t) for t in open(os.path.join(fluid, "out", "time.txt")) if t.strip()]
    rows = [(0.0, {k: 0.0 for k in BOXES})]
    for k, t in enumerate(times):
        pk = np.fromfile(os.path.join(fluid, "out", "p.%09d.float32" % k), dtype=np.float32)
        rows.append((t, {n: float(pk[s].mean()) for n, s in sel.items()}))
    os.makedirs(a.out)
    with open(os.path.join(a.out, "fsi_stroke.csv"), "w") as f:
        f.write("# The fluid pressure (Pa, gauge) of %s per box (valve_pressure.py), nodes: %s\n" % (
            os.path.abspath(a.run), ", ".join("%s %d" % (k, int(s.sum())) for k, s in sel.items())))
        f.write("t," + ",".join(BOXES) + "\n")
        for t, v in rows:
            f.write("%.9g," % t + ",".join("%.9g" % v[k] for k in BOXES) + "\n")
    stroke = [(t, v) for t, v in rows if t < a.period]
    for valve in ("pv", "sv"):
        with open(os.path.join(a.out, valve + "_load.csv"), "w") as f:
            f.write("# t [s], the pressure difference across the %s valve [Pa], below minus above, %d strokes of "
                    "%s (valve_pressure.py)%s\n" % ({"pv": "pressure", "sv": "suction"}[valve], a.strokes, a.run,
                                                   ", with simulation 1's outlet load fading out over each "
                                                   "stroke's first %g s" % a.ramp if valve == "pv" else ""))
            for n in range(a.strokes):
                for t, v in stroke[(1 if n else 0):]:
                    value = v[valve + "_below"] - v[valve + "_above"]
                    if valve == "pv":
                        value -= a.outlet * max(0.0, 1 - t / a.ramp)
                    f.write("%.9g, %.9g\n" % (n * a.period + t, value))
    print("valve_pressure: %d frames into %s" % (len(times), a.out))


if __name__ == "__main__":
    main()
