"""pvbatch script: render the last frame of a gravity-test XDMF, deformed and coloured by the
displacement magnitude: the cantilever over its undeformed outline, the contact block on the
obstacle (the zero level of its signed distance field).

usage: pvbatch render_deformed.py cantilever|contact <output.xdmf> <image.png> [sdf.xdmf]
"""

import os
import sys

from paraview.simple import (
    Calculator,
    ColorBy,
    Contour,
    CreateView,
    GetAnimationScene,
    GetColorTransferFunction,
    GetScalarBar,
    Outline,
    ResetCamera,
    SaveScreenshot,
    Show,
    WarpByVector,
    XDMFReader,
)

case, xdmf, image = sys.argv[1], sys.argv[2], sys.argv[3]
sdf = sys.argv[4] if len(sys.argv) > 4 else None

reader = XDMFReader(FileNames=[xdmf])
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.AnimationTime = reader.TimestepValues[-1]

disp = Calculator(Input=reader, ResultArrayName="displacement")
disp.Function = "disp_0*iHat + disp_1*jHat + disp_2*kHat"
warp = WarpByVector(Input=disp, Vectors=["POINTS", "displacement"])

view = CreateView("RenderView")
view.ViewSize = [1600, 900]
view.OrientationAxesVisibility = 1
view.Background = [1.0, 1.0, 1.0]
view.UseColorPaletteForBackground = 0

shown = Show(warp, view)
shown.SetRepresentationType("Surface With Edges")
ColorBy(shown, ("POINTS", "displacement", "Magnitude"))
shown.RescaleTransferFunctionToDataRange(True, False)
shown.SetScalarBarVisibility(view, True)
lut = GetColorTransferFunction("displacement")
bar = GetScalarBar(lut, view)
bar.Title = "|u|"
bar.ComponentTitle = ""
bar.TitleColor = [0.1, 0.1, 0.1]
bar.LabelColor = [0.1, 0.1, 0.1]

if case == "cantilever":
    outline = Show(Outline(Input=reader), view)
    outline.DiffuseColor = [0.4, 0.4, 0.4]
    outline.AmbientColor = [0.4, 0.4, 0.4]
    view.ViewSize = [1600, 700]
    view.CameraPosition = [5.0, -6.0, 18.0]
    view.CameraFocalPoint = [5.0, -0.5, 0.5]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    ResetCamera(view)
    view.GetActiveCamera().Dolly(1.6)
else:
    # See-through, so the obstacle under it shows.
    shown.Opacity = 0.55
    if sdf and os.path.exists(sdf):
        surface = Contour(Input=XDMFReader(FileNames=[sdf]))
        surface.ContourBy = ["POINTS", surface.Input.PointData.keys()[0]]
        surface.Isosurfaces = [0.0]
        obstacle = Show(surface, view)
        ColorBy(obstacle, None)
        obstacle.DiffuseColor = [0.75, 0.75, 0.72]
        obstacle.Opacity = 0.6
    # Follow the body, which is free and may have slid away from where it started.
    warp.UpdatePipeline(reader.TimestepValues[-1])
    b = warp.GetDataInformation().GetBounds()
    centre = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2]
    view.CameraFocalPoint = centre
    view.CameraPosition = [centre[0] + 3.2, centre[1] - 3.7, centre[2] + 2.4]
    view.CameraViewUp = [0.0, 0.0, 1.0]

SaveScreenshot(image, view, ImageResolution=list(view.ViewSize))
print(f"Wrote {image} (t = {reader.TimestepValues[-1]:g})")
