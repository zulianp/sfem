"""The stored history's error on the valves' surfaces, rendered off-screen by ParaView where the runs are.

usage: pvbatch render_history.py <case> <out folder> --reference=<run> --runs=<run>,... --times=t0,t1,...
                                  [--prefix=tp9_p2_] [--field=Hsum] [--range=lo,hi] [--size=W,H]

Reads, for each run <case>/<prefix><run>/sim2, the per-element error that `collect_prony.py snapshot`
wrote beside each history snapshot (history/<k>/<block>.iso.<field>.err.float64: |x - x_ref| over
max |x_ref|, the six components of the symmetric history) and shows its log10 as cell data on the
surfaces of the two valves, each seen from above at 40 degrees, deformed by the reference's
displacement at the snapshot's time
t_k (--times, in snapshot order; the reference's exported frame nearest to t_k). TET10 is drawn
through its corner nodes. One colour range for every image: --range in log10, or the largest
error over all panels and four decades below it.

Writes <out>/<run>_<k>_<valve>.png (valve: pressure_valve, suction_valve) without labels or colour
bar, and <out>/range.txt with the range (the figure's colour bar).
"""

import os
import sys

import numpy as np
import vtk
from paraview.simple import ColorBy, CreateView, ExtractSurface, GetColorTransferFunction, Render, SaveScreenshot, Show, TrivialProducer
from vtk.util import numpy_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_strokes import VALVES, plain, valve_grid  # noqa: E402


def blocks(mesh):
    """The valve blocks in the order valve_grid concatenates them, with their element counts."""
    out = []
    for block in sorted(os.listdir(os.path.join(mesh, "blocks"))):
        n = os.path.getsize(os.path.join(mesh, "blocks", block, "i0.int32")) // 4
        out.append((block, n))
    return out


def error(sim, k, field, order):
    parts = []
    for block, n in order:
        e = np.fromfile(os.path.join(sim, "history", str(k), "%s.iso.%s.err.float64" % (block, field)))
        if e.size != n:
            sys.exit("render_history.py: %s has %d elements in snapshot %d, the mesh %d" % (block, e.size, k, n))
        parts.append(e)
    return np.concatenate(parts)


def place(view, x):
    """The valve centred at x (m) from above and in front, 40 degrees down, its disc filling the image."""
    view.CameraFocalPoint = [x, -0.0012, 0]
    view.CameraPosition = [x, -0.0012 + 0.05 * 0.643, 0.05 * 0.766]
    view.CameraViewUp = [0, 0.766, -0.643]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 0.0042


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in argv[1:] if a.startswith("--"))
    if len(args) != 2 or not all(k in opts for k in ("reference", "runs", "times")):
        sys.exit(__doc__)
    case, out = args
    prefix = opts.get("prefix", "")
    field = opts.get("field", "Hsum")
    runs = opts["runs"].split(",")
    times = [float(t) for t in opts["times"].split(",")]
    width, height = (int(v) for v in opts.get("size", "1200,900").split(","))
    os.makedirs(out, exist_ok=True)

    ref = os.path.join(case, prefix + opts["reference"], "sim2")
    X, grid, points = valve_grid(os.path.join(ref, "mesh"))
    order = blocks(os.path.join(ref, "mesh"))
    errors = {(r, k): np.log10(np.maximum(error(os.path.join(case, prefix + r, "sim2"), k, field, order), 1e-300))
              for r in runs for k in range(len(times))}
    if "range" in opts:
        lo, hi = (float(v) for v in opts["range"].split(","))
    else:
        hi = float(np.ceil(max(e.max() for e in errors.values())))
        lo = hi - 4
    open(os.path.join(out, "range.txt"), "w").write("%g %g\n" % (lo, hi))

    valves = TrivialProducer()
    valves.GetClientSideObject().SetOutput(grid)
    section = ExtractSurface(Input=valves)
    # Viridis at 0, 1/4, 1/2, 3/4 and 1, the figure's colour bar (ldp_figures.py VIRIDIS).
    lut = GetColorTransferFunction("log_error")
    stops = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
    points_lut = [v for k, c in enumerate(stops) for v in [lo + (hi - lo) * k / 4] + [x / 255 for x in c]]
    lut.RGBPoints = points_lut
    view = CreateView("RenderView")
    plain(view, width, height)

    scratch = os.path.join(out, "panel.png")
    frame_times = np.loadtxt(os.path.join(ref, "out", "time.txt")).reshape(-1)
    shown = False
    for k, t in enumerate(times):
        step = int(np.argmin(np.abs(frame_times - t)))
        u = np.stack([np.fromfile(os.path.join(ref, "out", "disp.%d.%09d.float64" % (d, step))) for d in range(3)], axis=1)
        points.SetData(numpy_support.numpy_to_vtk(X + u, deep=True))
        for r in runs:
            values = numpy_support.numpy_to_vtk(np.clip(errors[(r, k)], lo, hi), deep=True)
            values.SetName("log_error")
            grid.GetCellData().AddArray(values)
            grid.Modified()
            valves.MarkModified(valves)
            if not shown:
                d = Show(section, view)
                ColorBy(d, ("CELLS", "log_error"))
                d.LookupTable = lut
                lut.AutomaticRescaleRangeMode = "Never"
                # Again after the colouring, which rescales the map to the data it first sees.
                lut.RGBPoints = points_lut
                shown = True
            for name, x in VALVES:
                place(view, x)
                # A first screenshot resets the camera to the data; the second keeps it.
                Render(view)
                SaveScreenshot(scratch, view, ImageResolution=[width, height])
                place(view, x)
                SaveScreenshot(os.path.join(out, "%s_%d_%s.png" % (r, k, name.replace(" ", "_"))), view,
                               ImageResolution=[width, height])
    if os.path.exists(scratch):
        os.remove(scratch)
    print("render_history.py: %d runs x %d times, log10 range [%g, %g], in %s" % (len(runs), len(times), lo, hi, out))


if __name__ == "__main__":
    main(sys.argv)
