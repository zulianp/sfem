# Part of qp/viz/twist_movies.sh.
"""Frames of the twist for one case: pvbatch render.py CASE_DIR LABEL OUT_DIR [--macro]."""

import os
import sys

from paraview.simple import (Calculator, ColorBy, ExtractSurface, Outline, Tube, GetActiveViewOrCreate, GetAnimationScene,
                             GetColorTransferFunction, GetScalarBar, Render, SaveScreenshot, Show, Text,
                             WarpByVector, XDMFReader)

case, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
edges = os.path.exists(os.path.join(case, "edges.xdmf"))
os.makedirs(outdir, exist_ok=True)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1280, 800]
view.Background = [1, 1, 1]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
view.OrientationAxesVisibility = 1

reader = XDMFReader(FileNames=[os.path.join(case, "case.xdmf")])
warp = WarpByVector(Input=reader, Vectors=["POINTS", "disp"])
mag = Calculator(Input=warp, ResultArrayName="|u|", Function="mag(disp)")
d = Show(mag, view)
d.SetRepresentationType("Surface" if edges else "Surface With Edges")
d.NonlinearSubdivisionLevel = 2
d.EdgeColor = [0.15, 0.15, 0.15]
d.LineWidth = 0.6
ColorBy(d, ("POINTS", "|u|"))
lut = GetColorTransferFunction("|u|")
lut.ApplyPreset("Viridis", True)
lut.RescaleTransferFunction(0.0, 1.0)
lut.AutomaticRescaleRangeMode = "Never"
bar = GetScalarBar(lut, view)
bar.Title = "|u|"
bar.ComponentTitle = ""
bar.TitleColor = [0, 0, 0]
bar.LabelColor = [0, 0, 0]
bar.TitleFontSize = 18
bar.LabelFontSize = 16
bar.WindowLocation = "Any Location"
bar.Position = [0.88, 0.25]
bar.ScalarBarLength = 0.5
bar.AutomaticLabelFormat = 0
bar.LabelFormat = "{:.1f}"
bar.RangeLabelFormat = "{:.1f}"
d.SetScalarBarVisibility(view, True)

# The undeformed cube, for reference.
od = Show(Outline(Input=reader), view)
od.ColorArrayName = ["POINTS", ""]
od.AmbientColor = [0.55, 0.55, 0.55]
od.DiffuseColor = [0.55, 0.55, 0.55]
od.LineWidth = 1.5

if edges:
    # The quadratic elements' true edges (3-node polylines through the mid-edge nodes) as thin tubes.
    ereader = XDMFReader(FileNames=[os.path.join(case, "edges.xdmf")])
    # Curved edges come already deformed (no disp array); straight 3-node ones are warped here.
    ereader.UpdatePipeline()
    curved = "disp" not in ereader.PointData.keys()
    ewarp = ereader if curved else WarpByVector(Input=ereader, Vectors=["POINTS", "disp"])
    tubes = Tube(Input=ExtractSurface(Input=ewarp))
    tubes.Radius = 0.0028
    tubes.NumberofSides = 8
    ed = Show(tubes, view)
    ed.ColorArrayName = ["POINTS", ""]
    ed.AmbientColor = [0.12, 0.12, 0.12]
    ed.DiffuseColor = [0.12, 0.12, 0.12]

title = Text(Text=label)
td = Show(title, view)
td.WindowLocation = "Upper Left Corner"
td.FontSize = 22
td.Color = [0, 0, 0]
angle = Text(Text="")
ad = Show(angle, view)
ad.WindowLocation = "Lower Left Corner"
ad.FontSize = 18
ad.Color = [0.1, 0.1, 0.1]

# Twist axis horizontal, the clamped face on the left, seen from above and in front.
view.CameraPosition = [2.6, -2.0, 2.0]
view.CameraFocalPoint = [0.5, 0.5, 0.5]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelProjection = 0
view.CameraViewAngle = 28

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(reader.TimestepValues)
for k, t in enumerate(times):
    scene.AnimationTime = t
    angle.Text = f"neo-Hookean, E = 1, nu = 0.3    twist {90 * t:.1f} deg    load step {k} of {len(times) - 1}"
    Render(view)
    SaveScreenshot(os.path.join(outdir, f"frame_{k:03d}.png"), view, ImageResolution=[1280, 800])
print("frames", len(times), "->", outdir)
