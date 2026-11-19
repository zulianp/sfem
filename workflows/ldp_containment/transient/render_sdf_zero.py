"""The obstacle's zero level set before and after sdf_round, rendered off-screen by ParaView.

usage: pvbatch render_sdf_zero.py <out> <label>=<grid> [<label>=<grid> ...] [--pump=LDP24A]

Each <grid> is an SDF grid folder (meta.yaml, sdf.float32), read memory-mapped; only boxes about the
valves' seats are contoured, at the grid's own resolution. Writes into <out> (a new folder):
  <valve>_seat.png    the seat from the valve's side, tilted towards +z, one panel per grid, the
                      same camera: the zero level set
  <valve>_spot.png    the same, close to the seat's sharpest spot (the smallest radius of the
                      original level set where the valve rests, contact_features.py)
  sections.png        sections through the spots, normal to z: the zero contour of every grid on
                      one plot (the first grid's obstacle, sdf > 0, shaded), the whole seat and a
                      1 mm window about the spot
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from PIL import Image
from paraview.simple import (
    ColorBy,
    Contour,
    CreateView,
    Render,
    SaveScreenshot,
    Show,
    Text,
    TrivialProducer,
)
from vtk.util import numpy_support

MM = 1e-3
# Per pump: per valve, the seat's box (mm: x0, x1, y0, y1, z0, z1), its sharpest spot (mm) and the
# side of the seat the valve is on, where the views look from (+1: above, the pressure valve; -1:
# below, the suction valve, which closes upwards against the plate's underside).
PUMPS = {
    "LDP24A": {
        "suction valve": {"box": (4.0, 20.0, 0.5, 3.0, -8.5, 8.5), "spot": (7.690, 2.493, -1.910),
                          "side": -1},
        "pressure valve": {"box": (-20.0, -4.0, 0.5, 3.0, -8.5, 8.5), "spot": (-16.779, 2.508, -4.245),
                           "side": 1},
    },
}
COLOURS = ["black", "tab:red", "tab:blue", "tab:green"]


class Grid:
    def __init__(self, folder):
        meta = {}
        for line in open(os.path.join(folder, "meta.yaml")):
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        self.n = [int(meta[k]) for k in ("nx", "ny", "nz")]
        self.o = [float(meta[k]) for k in ("ox", "oy", "oz")]
        self.h = [float(meta[k]) for k in ("dx", "dy", "dz")]
        self.v = np.memmap(os.path.join(folder, meta.get("path", "sdf.float32")), dtype=np.float32, mode="r",
                           shape=(self.n[2], self.n[1], self.n[0]))

    def index(self, d, x):
        return int(np.clip(round((x * MM - self.o[d]) / self.h[d]), 0, self.n[d] - 1))

    def box(self, b):
        """The samples in the box b (mm), as vtkImageData."""
        i0, i1 = self.index(0, b[0]), self.index(0, b[1])
        j0, j1 = self.index(1, b[2]), self.index(1, b[3])
        k0, k1 = self.index(2, b[4]), self.index(2, b[5])
        values = np.ascontiguousarray(self.v[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1])
        image = vtk.vtkImageData()
        image.SetDimensions(values.shape[2], values.shape[1], values.shape[0])
        image.SetOrigin(self.o[0] + i0 * self.h[0], self.o[1] + j0 * self.h[1], self.o[2] + k0 * self.h[2])
        image.SetSpacing(*self.h)
        array = numpy_support.numpy_to_vtk(values.ravel(), deep=True)
        array.SetName("sdf")
        image.GetPointData().SetScalars(array)
        return image

    def section(self, z, b):
        """The samples of the plane nearest z (mm) in the box b's x and y: x, y (mm) and values."""
        k = self.index(2, z)
        i0, i1 = self.index(0, b[0]), self.index(0, b[1])
        j0, j1 = self.index(1, b[2]), self.index(1, b[3])
        x = (self.o[0] + self.h[0] * np.arange(i0, i1 + 1)) / MM
        y = (self.o[1] + self.h[1] * np.arange(j0, j1 + 1)) / MM
        return x, y, np.array(self.v[k, j0:j1 + 1, i0:i1 + 1]), (self.o[2] + k * self.h[2]) / MM


def render(grids, box, focus, position, scale, size, path):
    """One panel per grid, the same parallel camera, side by side into path."""
    panels = []
    scratch = path + ".panel.png"
    for label, grid in grids:
        image = TrivialProducer()
        image.GetClientSideObject().SetOutput(grid.box(box))
        surface = Contour(Input=image, ContourBy=["POINTS", "sdf"], Isosurfaces=[0.0], ComputeNormals=1)
        view = CreateView("RenderView")
        view.ViewSize = list(size)
        view.OrientationAxesVisibility = 0
        view.UseColorPaletteForBackground = 0
        view.Background = [1, 1, 1]
        d = Show(surface, view)
        ColorBy(d, None)
        d.AmbientColor = d.DiffuseColor = [0.75, 0.72, 0.68]
        d.Specular = 0.3
        t = Show(Text(Text=label), view)
        t.WindowLocation = "Upper Left Corner"
        t.Color = [0, 0, 0]
        t.FontSize = 20
        Render(view)
        SaveScreenshot(scratch, view, ImageResolution=list(size))  # resets the camera; placed after
        view.CameraFocalPoint = [c * MM for c in focus]
        view.CameraPosition = [(f + p) * MM for f, p in zip(focus, position)]
        view.CameraViewUp = [0, 1, 0] if abs(position[1]) < 0.9 * np.linalg.norm(position) else [0, 0, -1]
        view.CameraParallelProjection = 1
        view.CameraParallelScale = scale * MM
        SaveScreenshot(scratch, view, ImageResolution=list(size))
        panels.append(Image.open(scratch).convert("RGB"))
    frame = Image.new("RGB", (size[0] * len(panels), size[1]), "white")
    for i, p in enumerate(panels):
        frame.paste(p, (i * size[0], 0))
    frame.save(path)
    os.remove(scratch)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in argv[1:] if a.startswith("--"))
    if len(args) < 2:
        sys.exit(__doc__)
    out = args[0]
    if os.path.exists(out):
        sys.exit("render_sdf_zero.py: %s exists; give a new folder" % out)
    os.makedirs(out)
    grids = [(a.split("=", 1)[0], Grid(a.split("=", 1)[1])) for a in args[1:]]
    valves = PUMPS[opts.get("pump", "LDP24A")]

    for name, v in valves.items():
        box, spot, side = v["box"], v["spot"], v["side"]
        centre = (0.5 * (box[0] + box[1]), 2.5, 0.0)
        tag = name.replace(" ", "_")
        # The seat from above, tilted towards +z; then close to the spot, from above and the side.
        render(grids, box, centre, (0.0, side * 30.0, 18.0), 6.0, (900, 700), os.path.join(out, tag + "_seat.png"))
        spot_box = (spot[0] - 0.8, spot[0] + 0.8, spot[1] - 0.8, spot[1] + 0.8, spot[2] - 0.8, spot[2] + 0.8)
        render(grids, spot_box, spot, (6.0, side * 20.0, 12.0), 0.45, (700, 700), os.path.join(out, tag + "_spot.png"))

    fig, axes = plt.subplots(len(valves), 2, figsize=(14, 4.2 * len(valves)), gridspec_kw={"width_ratios": [2.2, 1]})
    for row, (name, v) in zip(np.atleast_2d(axes), valves.items()):
        box, spot = v["box"], v["spot"]
        for g, ((label, grid), colour) in enumerate(zip(grids, COLOURS)):
            for ax, b in ((row[0], box), (row[1], (spot[0] - 0.5, spot[0] + 0.5, spot[1] - 0.5, spot[1] + 0.5))):
                x, y, values, z = grid.section(spot[2], b)
                if g == 0:
                    ax.contourf(x, y, values, levels=[0.0, np.inf], colors=["0.88"])
                ax.contour(x, y, values, levels=[0.0], colors=[colour], linewidths=1.2)
                ax.plot([], [], color=colour, label=label)
        for ax in row:
            ax.set_aspect("equal")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
        row[0].set_title("%s seat, section z = %.3f mm (obstacle shaded)" % (name, z), fontsize=10)
        row[1].set_title("1 mm about (%.2f, %.2f)" % spot[:2], fontsize=10)
        row[0].plot([spot[0]], [spot[1]], "o", mfc="none", mec="0.4", ms=14)
        row[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sections.png"), dpi=130)
    print("render_sdf_zero.py: %s" % ", ".join(sorted(os.listdir(out))))


if __name__ == "__main__":
    main(sys.argv)
