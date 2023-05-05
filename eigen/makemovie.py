# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Xdmf3ReaderT'
xxmf = Xdmf3ReaderT(registrationName='x.xmf', FileName=['./x.xmf'])
xxmf.PointArrays = ['real']

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
xxmfDisplay = Show(xxmf, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'real'
realTF2D = GetTransferFunction2D('real')
realTF2D.ScalarRangeInitialized = 1
realTF2D.Range = [-0.04421231896179885, -0.02329834094024706, 0.0, 1.0]

# get color transfer function/color map for 'real'
realLUT = GetColorTransferFunction('real')
realLUT.AutomaticRescaleRangeMode = 'Never'
realLUT.TransferFunction2D = realTF2D
realLUT.RGBPoints = [-0.1733855153172917, 0.231373, 0.298039, 0.752941, -1.3877787807814457e-16, 0.865003, 0.865003, 0.865003, 0.17338551531729143, 0.705882, 0.0156863, 0.14902]
realLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'real'
realPWF = GetOpacityTransferFunction('real')
realPWF.Points = [-0.1733855153172917, 0.0, 0.5, 0.0, 0.17338551531729143, 1.0, 0.5, 0.0]
realPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
xxmfDisplay.Representation = 'Surface'
xxmfDisplay.ColorArrayName = ['POINTS', 'real']
xxmfDisplay.LookupTable = realLUT
xxmfDisplay.SelectTCoordArray = 'None'
xxmfDisplay.SelectNormalArray = 'None'
xxmfDisplay.SelectTangentArray = 'None'
xxmfDisplay.OSPRayScaleArray = 'real'
xxmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
xxmfDisplay.SelectOrientationVectors = 'None'
xxmfDisplay.ScaleFactor = 0.1
xxmfDisplay.SelectScaleArray = 'real'
xxmfDisplay.GlyphType = 'Arrow'
xxmfDisplay.GlyphTableIndexArray = 'real'
xxmfDisplay.GaussianRadius = 0.005
xxmfDisplay.SetScaleArray = ['POINTS', 'real']
xxmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
xxmfDisplay.OpacityArray = ['POINTS', 'real']
xxmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
xxmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
xxmfDisplay.PolarAxes = 'PolarAxesRepresentation'
xxmfDisplay.ScalarOpacityFunction = realPWF
xxmfDisplay.ScalarOpacityUnitDistance = 0.14324046167853577
xxmfDisplay.OpacityArrayName = ['POINTS', 'real']
xxmfDisplay.SelectInputVectors = [None, '']
xxmfDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
xxmfDisplay.ScaleTransferFunction.Points = [-0.04421231896179885, 0.0, 0.5, 0.0, -0.02329834094024706, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
xxmfDisplay.OpacityTransferFunction.Points = [-0.04421231896179885, 0.0, 0.5, 0.0, -0.02329834094024706, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
xxmfDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
# realLUT.RescaleTransferFunction(-0.1733855153172917, 0.17338551531729146)
realLUT.RescaleTransferFunction(-0.01, 0.01)

# Rescale transfer function
# realPWF.RescaleTransferFunction(-0.1733855153172917, 0.17338551531729146)
realPWF.RescaleTransferFunction(-0.01, 0.01)

# get color legend/bar for realLUT in view renderView1
realLUTColorBar = GetScalarBar(realLUT, renderView1)
realLUTColorBar.Title = 'real'
realLUTColorBar.ComponentTitle = ''

# change scalar bar placement
realLUTColorBar.WindowLocation = 'Any Location'
realLUTColorBar.Position = [0.0919023136246787, 0.3665943600867679]
realLUTColorBar.ScalarBarLength = 0.3300000000000001

# reset view to fit data
renderView1.ResetCamera(True)

# reset view to fit data bounds
renderView1.ResetCamera(-0.5, 0.5, -0.49999427795410156, 0.5, -0.5, 0.5, False)

# get layout
layout1 = GetLayout()

# layout/tab size in pixels
layout1.SetSize(1556, 922)

# current camera placement for renderView1
renderView1.CameraPosition = [1.5434827324629512, 2.3140706713534973, 1.859801316452238]
renderView1.CameraFocalPoint = [0.0, 2.86102294921875e-06, 0.0]
renderView1.CameraViewUp = [-0.41703161329094246, 0.7219362062938995, -0.5521709405228838]
renderView1.CameraParallelScale = 0.8660237519752193

# save animation
SaveAnimation('./movie.avi', renderView1, ImageResolution=[1556, 920],
    FrameRate=4,
    FrameWindow=[0, 799])

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1556, 922)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [1.5434827324629512, 2.3140706713534973, 1.859801316452238]
renderView1.CameraFocalPoint = [0.0, 2.86102294921875e-06, 0.0]
renderView1.CameraViewUp = [-0.41703161329094246, 0.7219362062938995, -0.5521709405228838]
renderView1.CameraParallelScale = 0.8660237519752193

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).