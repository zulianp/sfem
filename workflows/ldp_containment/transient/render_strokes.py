"""Frames of simulation 2 for a movie, rendered off-screen by ParaView where the run is.

usage: pvbatch render_strokes.py <sim2 folder> <frames folder> [--every=N] [--size=W,H]
                                  [--first=K] [--count=M]

Reads the run's raw output -- the valve mesh (<sim2>/mesh: x, y, z and blocks/<name>/i0..i3, TET4),
every exported step (<sim2>/out/disp.<d>.<step>.float64, time.txt) and the obstacle's grid
(<sim2>/sdf: meta.yaml, sdf.float32) -- so no Exodus or XDMF file is written first.

Each frame is a 3 x 2 grid. Four views of the valves at X + u, coloured by the vertical
displacement u_y in mm (fixed range), with the cavity wall, the obstacle's zero level set,
translucent:
  top left       overview from the front, a little from above, the wall's front half cut away
  top middle     from above
  bottom left    the pressure valve's section at z = 0, the wall's section as a line
  bottom middle  the suction valve's section at z = 0
and two charts of the run's history (<sim2>/history.csv), a cursor at the frame's time:
  top right      the chamber pressure (<sim2>/../fp70_membrane_pressure.csv) and the valves' centre
                 lift (the probes pv_centre, sv_centre)
  bottom right   the Prony series (the prony probe): each term's RMS overstress |H_k|, labelled by
                 its tau (from <sim2>/run.log), and the step's relaxation factor gamma
The overview also shows the time and the chamber pressure. Frames are
<frames>/%05d.png, numbered over the whole run; --first and --count render a range of them, one
process per range (the EGL driver of the Alps nodes, 595.91.07, aborts a process after 25 frames).
Encode them with ffmpeg, e.g.
ffmpeg -framerate 30 -i %05d.png -c:v libx264 -pix_fmt yuv420p strokes.mp4.
"""

import bisect
import glob
import os
import sys

import csv
import io
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from PIL import Image
from paraview.simple import (
    Clip,
    ColorBy,
    Contour,
    CreateView,
    GetColorTransferFunction,
    GetScalarBar,
    Render,
    SaveScreenshot,
    Show,
    Slice,
    Text,
    TrivialProducer,
)
from vtk.util import numpy_support

# The valves and their centres in x (m), where each section is centred.
VALVES = (("pressure valve", -0.011), ("suction valve", 0.011))


def read(path_glob, dtype):
    paths = glob.glob(path_glob)
    if len(paths) != 1:
        sys.exit("render_strokes.py: expected one file for %s, found %d" % (path_glob, len(paths)))
    return np.fromfile(paths[0], dtype=dtype)


def read_meta(path):
    meta = {}
    for line in open(path):
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def valve_grid(mesh):
    X = np.stack([read(os.path.join(mesh, "%s.float*" % c), np.float32).astype(np.float64) for c in "xyz"], axis=1)
    cells = []
    for block in sorted(os.listdir(os.path.join(mesh, "blocks"))):
        conn = [read(os.path.join(mesh, "blocks", block, "i%d.int32" % k), np.int32) for k in range(4)]
        cells.append(np.stack(conn, axis=1))
    cells = np.concatenate(cells)
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(X.copy(), deep=True))
    grid.SetPoints(points)
    flat = np.hstack([np.full((len(cells), 1), 4, dtype=np.int64), cells.astype(np.int64)]).ravel()
    ids = vtk.vtkCellArray()
    ids.ImportLegacyFormat(numpy_support.numpy_to_vtkIdTypeArray(flat, deep=True))
    grid.SetCells(vtk.VTK_TETRA, ids)
    return X, grid, points


def wall_image(sdf, stride):
    """The obstacle's signed distance on every `stride`-th grid point, as vtkImageData."""
    meta = read_meta(os.path.join(sdf, "meta.yaml"))
    n = [int(meta[k]) for k in ("nx", "ny", "nz")]
    o = [float(meta[k]) for k in ("ox", "oy", "oz")]
    d = [float(meta[k]) for k in ("dx", "dy", "dz")]
    values = np.fromfile(os.path.join(sdf, meta.get("path", "sdf.float32")), dtype=np.float32).reshape(n[2], n[1], n[0])
    values = np.ascontiguousarray(values[::stride, ::stride, ::stride])
    image = vtk.vtkImageData()
    image.SetDimensions(values.shape[2], values.shape[1], values.shape[0])
    image.SetOrigin(*o)
    image.SetSpacing(*(stride * s for s in d))
    array = numpy_support.numpy_to_vtk(values.ravel(), deep=True)
    array.SetName("sdf")
    image.GetPointData().SetScalars(array)
    return image


def plain(view, width, height):
    view.ViewSize = [width, height]
    view.OrientationAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.Background = [1, 1, 1]


def grey(display, rgb):
    ColorBy(display, None)
    display.AmbientColor = display.DiffuseColor = rgb


def label(view, text, size):
    source = Text(Text=text)
    display = Show(source, view)
    display.WindowLocation = "Upper Left Corner"
    display.Color = [0, 0, 0]
    display.FontSize = size
    return source


def history(sim):
    """The run's history as columns of floats, the converged steps only."""
    rows = [r for r in csv.DictReader(open(os.path.join(sim, "history.csv"))) if r["status"] != "initial"]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "status"} if rows else {}


def prony_legend(sim):
    """{column: tau} from the probe's lines in the run's log."""
    legend = {}
    log = os.path.join(sim, "run.log")
    if os.path.isfile(log):
        for m in re.finditer(r"Prony probe: (\w+) is the term g = ([0-9.eE+-]+), tau = ([0-9.eE+-]+)", open(log).read()):
            legend[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return legend


class Charts:
    """The two history charts, drawn once; each frame moves their cursors."""

    def __init__(self, sim, t_p, size):
        h = history(sim)
        legend = prony_legend(sim)
        dpi = 100
        self.size = size
        self.figures, self.cursors = [], []
        t = h.get("t", np.zeros(0)) * 1e3

        fig, ax = plt.subplots(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        if t_p is not None and len(t):
            mask = t_p[:, 0] <= t[-1] / 1e3
            ax.plot(t_p[mask, 0] * 1e3, t_p[mask, 1] / 1e3, color="0.3", lw=1.2, label="chamber pressure")
        ax.axhline(90.543, color="0.3", lw=0.8, ls=":", label="outlet pressure")
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("pressure (kPa)")
        lift = ax.twinx()
        for name, color in (("pv_centre", "tab:red"), ("sv_centre", "tab:blue")):
            if name + "_uy" in h:
                label = ("pressure" if name.startswith("pv") else "suction") + " valve centre lift"
                lift.plot(t, h[name + "_uy"] * 1e3, color=color, lw=1.4, label=label)
        lift.set_ylabel("lift (mm)")
        lines = ax.get_legend_handles_labels()[0] + lift.get_legend_handles_labels()[0]
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower left")
        ax.set_title("load and lift", fontsize=10)
        self._add(fig, [ax])

        fig, ax = plt.subplots(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        terms = sorted((k for k in h if re.fullmatch(r"\w+_H\d+", k) and np.any(h[k] > 0)), key=lambda k: -legend.get(k, (0, 0))[1])
        cmap = plt.get_cmap("viridis", max(len(terms), 1))
        for i, k in enumerate(terms):
            g, tau = legend.get(k, (0, 0))
            ax.semilogy(t, np.maximum(h[k], 1e-30) / 1e3, color=cmap(i), lw=1.4, label="tau = %.3g s (g = %.2g)" % (tau, g))
        ax.set_xlabel("t (ms)")
        ax.set_ylabel("RMS overstress |H_k| (kPa)")
        if terms:
            low = min(np.max(h[k]) for k in terms) / 1e3
            ax.set_ylim(bottom=max(low * 1e-3, 1e-12))
            ax.legend(fontsize=8, loc="lower right")
            gammas = [k for k in h if k.endswith("_gamma")]
            if gammas:
                ax.set_title("Prony series: each term's overstress; gamma = %.4g" % np.median(h[gammas[0]]), fontsize=10)
        else:
            ax.text(0.5, 0.5, "no prony probe in this run", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Prony series", fontsize=10)
        self._add(fig, [ax])

    def _add(self, fig, axes):
        fig.tight_layout()
        self.figures.append(fig)
        self.cursors.append([a.axvline(0, color="k", lw=0.8) for a in axes])

    def images(self, t_ms):
        out = []
        for fig, cursors in zip(self.figures, self.cursors):
            for c in cursors:
                c.set_xdata([t_ms, t_ms])
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            out.append(Image.open(buffer).convert("RGB").resize(self.size))
        return out


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in argv[1:] if a.startswith("--"))
    if len(args) != 2:
        sys.exit(__doc__)
    sim, frames = args
    every = int(opts.get("every", "1"))
    first = int(opts.get("first", "0"))
    width, height = (int(v) for v in opts.get("size", "1920,1080").split(","))
    os.makedirs(frames, exist_ok=True)

    X, grid, points = valve_grid(os.path.join(sim, "mesh"))
    uy = numpy_support.numpy_to_vtk(np.zeros(len(X)), deep=True)
    uy.SetName("uy_mm")
    grid.GetPointData().AddArray(uy)
    valves = TrivialProducer()
    valves.GetClientSideObject().SetOutput(grid)

    # The cavity wall: the zero level set, on every second grid point (60 um), once.
    image = TrivialProducer()
    image.GetClientSideObject().SetOutput(wall_image(os.path.join(sim, "sdf"), 2))
    wall = Contour(Input=image, ContourBy=["POINTS", "sdf"], Isosurfaces=[0.0])
    wall_back = Clip(Input=wall, ClipType="Plane")
    wall_back.ClipType.Origin = [0, 0, 0]
    wall_back.ClipType.Normal = [0, 0, 1]
    wall_back.Invert = 1
    wall_section = Slice(Input=wall, SliceType="Plane")
    wall_section.SliceType.Origin = [0, 0, 0]
    wall_section.SliceType.Normal = [0, 0, 1]
    valve_section = Slice(Input=valves, SliceType="Plane")
    valve_section.SliceType.Origin = [0, 0, 0]
    valve_section.SliceType.Normal = [0, 0, 1]

    lut = GetColorTransferFunction("uy_mm")
    lut.ApplyPreset("Cool to Warm", True)

    panel = (width // 3, height // 2)
    views = [CreateView("RenderView") for _ in range(4)]
    for v in views:
        plain(v, *panel)
    overview, top, sections = views[0], views[1], views[2:]

    for v, surface, opacity in ((overview, wall_back, 0.3), (top, wall, 0.15)):
        d = Show(surface, v)
        grey(d, [0.55, 0.55, 0.6])
        d.Opacity = opacity
    for v in sections:
        d = Show(wall_section, v)
        grey(d, [0.2, 0.2, 0.2])
        d.LineWidth = 2
    for v, source in ((overview, valves), (top, valves), (sections[0], valve_section), (sections[1], valve_section)):
        d = Show(source, v)
        ColorBy(d, ("POINTS", "uy_mm"))
        d.LookupTable = lut
    # The range fixed after the colouring, which rescales it to the data it first sees.
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(-1.4, 1.1)
    bar = GetScalarBar(lut, overview)
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Lower Center"
    bar.Title = "vertical displacement (mm)"
    bar.ComponentTitle = ""
    bar.TitleColor = bar.LabelColor = [0, 0, 0]
    bar.Visibility = 1

    clock = label(overview, "", 22)
    label(top, "from above", 18)
    for v, (name, _) in zip(sections, VALVES):
        label(v, "%s, section z = 0" % name, 18)

    # The cameras after a first render, which resets them to the data.
    for v in views:
        Render(v)
    centre = X.mean(axis=0)
    overview.CameraFocalPoint = list(centre)
    overview.CameraPosition = [centre[0], centre[1] + 0.014, centre[2] + 0.065]
    overview.CameraViewUp = [0, 1, 0]
    overview.CameraViewAngle = 30
    top.CameraFocalPoint = list(centre)
    top.CameraPosition = [centre[0], centre[1] + 0.08, centre[2]]
    top.CameraViewUp = [0, 0, -1]
    top.CameraParallelProjection = 1
    top.CameraParallelScale = 0.0145
    for v, (_, x) in zip(sections, VALVES):
        v.CameraFocalPoint = [x, -0.0008, 0]
        v.CameraPosition = [x, -0.0008, 0.05]
        v.CameraViewUp = [0, 1, 0]
        v.CameraParallelProjection = 1
        v.CameraParallelScale = 0.0055

    out = os.path.join(sim, "out")
    times = np.loadtxt(os.path.join(out, "time.txt")).reshape(-1)
    steps = sorted(int(p.split(".")[-2]) for p in glob.glob(os.path.join(out, "disp.0.*.float64")))
    table = os.path.join(sim, "..", "fp70_membrane_pressure.csv")
    t_p = np.loadtxt(table, delimiter=",", comments="#") if os.path.isfile(table) else None
    charts = Charts(sim, t_p, panel)
    scratch = os.path.join(frames, "panel.png")

    frames_all = steps[::every]
    count = int(opts.get("count", str(len(frames_all))))
    for n in range(first, min(first + count, len(frames_all))):
        step = frames_all[n]
        u = np.stack([np.fromfile(os.path.join(out, "disp.%d.%09d.float64" % (d, step))) for d in range(3)], axis=1)
        points.SetData(numpy_support.numpy_to_vtk(X + u, deep=True))
        uy_mm = numpy_support.numpy_to_vtk(u[:, 1] * 1e3, deep=True)
        uy_mm.SetName("uy_mm")
        grid.GetPointData().AddArray(uy_mm)
        grid.Modified()
        valves.MarkModified(valves)
        t = times[step]
        text = "t = %6.2f ms" % (t * 1e3)
        if t_p is not None:
            i = min(max(bisect.bisect_left(t_p[:, 0], t), 1), len(t_p) - 1)
            (t0, p0), (t1, p1) = t_p[i - 1], t_p[i]
            text += "    chamber %+6.1f kPa" % ((p0 + (p1 - p0) * (t - t0) / (t1 - t0)) / 1e3)
        clock.Text = text
        # The four views and the two charts, placed in the frame's 3 x 2 grid.
        frame = Image.new("RGB", (3 * panel[0], 2 * panel[1]), "white")
        for v, (col, row) in zip(views, ((0, 0), (1, 0), (0, 1), (1, 1))):
            SaveScreenshot(scratch, v, ImageResolution=list(panel))
            frame.paste(Image.open(scratch).convert("RGB"), (col * panel[0], row * panel[1]))
        for row, image in enumerate(charts.images(t * 1e3)):
            frame.paste(image, (2 * panel[0], row * panel[1]))
        frame.save(os.path.join(frames, "%05d.png" % n))
    os.remove(scratch)
    print("render_strokes.py: frames %d to %d of %d in %s" % (first, min(first + count, len(frames_all)) - 1, len(frames_all), frames))


if __name__ == "__main__":
    main(sys.argv)
