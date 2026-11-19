"""The wound strip (cases/epdm_strip_turns.yaml) rendered off-screen by ParaView where the runs are.

usage: pvbatch render_strip.py <run> <out folder> (--steps=k0,k1,... | --times=t0,t1,...)
                                [--range=0,umax] [--size=W,H] [--stamp] [--prefix=frame] [--crop]

Reads <run>/results/mesh and <run>/results/out/disp.<c>.<step>.float64 and writes, per step k (or
the exported frame nearest to each time, since cut-backs shift the steps),
<out>/<prefix>_<k>.png: the strip's surface deformed by u and coloured by |u| (viridis, the range
--range in m, default 0 to the largest |u| over the steps drawn), seen obliquely from above and the
side. The quadratic elements are drawn through their corner nodes. --stamp writes the time in the
upper left corner (the movies); the paper's frames carry no text. --crop cuts the images of the call
to the union of their non-white pixels, so the panels keep one scale without the empty margin. The Alps EGL driver aborts a process
after about 25 images, so callers pass at most 20 steps per call (render_strip.sbatch).
"""

import os
import sys

import numpy as np
import vtk
from paraview.simple import (ColorBy, CreateView, ExtractSurface, GetColorTransferFunction, Render, SaveScreenshot, Show,
                             Text, TrivialProducer)
from vtk.util import numpy_support

# The corner nodes of each element type and the VTK cell drawn through them.
CORNERS = {"HEX8": (8, vtk.VTK_HEXAHEDRON), "HEX27": (8, vtk.VTK_HEXAHEDRON), "TET4": (4, vtk.VTK_TETRA),
           "TET10": (4, vtk.VTK_TETRA)}
# Viridis at 0, 1/4, 1/2, 3/4 and 1 (ldp_figures.py VIRIDIS).
VIRIDIS = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]


def meta(mesh):
    out = {}
    for line in open(os.path.join(mesh, "meta.yaml")):
        if ":" in line and not line.lstrip().startswith("-"):
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def raw(mesh, stem, dtype):
    for name in os.listdir(mesh):
        if name.startswith(stem + "."):
            return np.fromfile(os.path.join(mesh, name), dtype=dtype)
    sys.exit("render_strip.py: no %s in %s" % (stem, mesh))


def strip_grid(mesh):
    m = meta(mesh)
    element = m["element_type"]
    corners, cell = CORNERS[element]
    X = np.stack([raw(mesh, c, np.float32).astype(np.float64) for c in "xyz"], axis=1)
    conn = np.stack([raw(mesh, "i%d" % k, np.int32) for k in range(corners)], axis=1)
    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(X.copy(), deep=True))
    grid.SetPoints(points)
    flat = np.hstack([np.full((len(conn), 1), corners, dtype=np.int64), conn.astype(np.int64)]).ravel()
    ids = vtk.vtkCellArray()
    ids.ImportLegacyFormat(numpy_support.numpy_to_vtkIdTypeArray(flat, deep=True))
    grid.SetCells(cell, ids)
    return X, grid, points


def displacement(run, step, n):
    out = os.path.join(run, "results", "out")
    u = np.stack([np.fromfile(os.path.join(out, "disp.%d.%09d.float64" % (d, step))) for d in range(3)], axis=1)
    if u.shape[0] != n:
        sys.exit("render_strip.py: step %d has %d nodes, the mesh %d" % (step, u.shape[0], n))
    return u


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in argv[1:] if a.startswith("--"))
    if len(args) != 2 or ("steps" not in opts and "times" not in opts):
        sys.exit(__doc__)
    run, out = args
    times = np.loadtxt(os.path.join(run, "results", "out", "time.txt")).reshape(-1)
    if "steps" in opts:
        steps = [int(k) for k in opts["steps"].split(",")]
    else:
        steps = [int(np.argmin(np.abs(times - float(t)))) for t in opts["times"].split(",")]
    width, height = (int(v) for v in opts.get("size", "1600,700").split(","))
    prefix = opts.get("prefix", "frame")
    os.makedirs(out, exist_ok=True)

    X, grid, points = strip_grid(os.path.join(run, "results", "mesh"))
    us = {k: displacement(run, k, len(X)) for k in steps}
    if "range" in opts:
        lo, hi = (float(v) for v in opts["range"].split(","))
    else:
        lo, hi = 0.0, max(float(np.linalg.norm(u, axis=1).max()) for u in us.values())

    def set_step(k):
        u = us[k]
        points.SetData(numpy_support.numpy_to_vtk((X + u).copy(), deep=True))
        norm = numpy_support.numpy_to_vtk(np.linalg.norm(u, axis=1).copy(), deep=True)
        norm.SetName("u_norm")
        grid.GetPointData().AddArray(norm)
        grid.Modified()
        source.MarkModified(source)

    source = TrivialProducer()
    source.GetClientSideObject().SetOutput(grid)
    surface = ExtractSurface(Input=source)
    view = CreateView("RenderView")
    view.ViewSize = [width, height]
    view.OrientationAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.Background = [1, 1, 1]
    rgb = [v for k, c in enumerate(VIRIDIS) for v in [lo + (hi - lo) * k / 4] + [x / 255 for x in c]]
    lut = GetColorTransferFunction("u_norm")
    set_step(steps[0])
    display = Show(surface, view)
    ColorBy(display, ("POINTS", "u_norm"))
    display.LookupTable = lut
    lut.AutomaticRescaleRangeMode = "Never"
    # Again after the colouring, which rescales the map to the data it first sees.
    lut.RGBPoints = rgb
    display.Ambient, display.Diffuse = 0.25, 0.85
    stamp = None
    if "stamp" in opts:
        stamp = Text(Text="")
        sd = Show(stamp, view)
        sd.WindowLocation = "Upper Left Corner"
        sd.Color = [0, 0, 0]
        sd.FontSize = 22
    # A first render resets the camera to the data; the view below is set after it.
    Render(view)

    # One view for the whole run, whichever steps a call draws: the box the deformed strip sweeps,
    # sampled at every 20th exported step, seen from above, the front and the clamped end's side.
    lo_box, hi_box = X.min(axis=0), X.max(axis=0)
    for k in range(0, len(times), 20):
        Y = X + displacement(run, k, len(X))
        lo_box, hi_box = np.minimum(lo_box, Y.min(axis=0)), np.maximum(hi_box, Y.max(axis=0))
    centre = (lo_box + hi_box) / 2
    direction = np.array([-0.35, -0.75, 0.56])
    direction /= np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0])
    up -= up.dot(direction) * direction
    up /= np.linalg.norm(up)
    right = np.cross(direction, up)
    corners = np.array([[a, b, c] for a in (lo_box[0], hi_box[0]) for b in (lo_box[1], hi_box[1]) for c in (lo_box[2], hi_box[2])]) - centre
    half_h = max(np.abs(corners @ up).max(), np.abs(corners @ right).max() * height / width)
    view.CameraFocalPoint = list(centre)
    view.CameraPosition = list(centre + 0.5 * direction)
    view.CameraViewUp = list(up)
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 1.04 * half_h

    for k in steps:
        set_step(k)
        if stamp is not None:
            stamp.Text = "t = %.3f s" % times[k]
        Render(view)
        SaveScreenshot(os.path.join(out, "%s_%05d.png" % (prefix, k)), view, ImageResolution=[width, height])
    if "crop" in opts:
        crop([os.path.join(out, "%s_%05d.png" % (prefix, k)) for k in steps])
    open(os.path.join(out, "range.txt"), "w").write("%g %g\n" % (lo, hi))


def crop(paths, margin=8):
    """Cut every image to the union of the images' non-white pixels, plus a margin."""
    images = []
    for path in paths:
        reader = vtk.vtkPNGReader()
        reader.SetFileName(path)
        reader.Update()
        data = reader.GetOutput()
        w, h, _ = data.GetDimensions()
        c = data.GetPointData().GetScalars().GetNumberOfComponents()
        images.append(numpy_support.vtk_to_numpy(data.GetPointData().GetScalars()).reshape(h, w, c))
    ink = np.zeros(images[0].shape[:2], dtype=bool)
    for a in images:
        ink |= (a[:, :, :3] < 250).any(axis=2)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, ink.shape[0])
    c0, c1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, ink.shape[1])
    for path, a in zip(paths, images):
        cut = np.ascontiguousarray(a[r0:r1, c0:c1])
        image = vtk.vtkImageData()
        image.SetDimensions(cut.shape[1], cut.shape[0], 1)
        array = numpy_support.numpy_to_vtk(cut.reshape(-1, cut.shape[2]), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        image.GetPointData().SetScalars(array)
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputData(image)
        writer.Write()


if __name__ == "__main__":
    main(sys.argv)
