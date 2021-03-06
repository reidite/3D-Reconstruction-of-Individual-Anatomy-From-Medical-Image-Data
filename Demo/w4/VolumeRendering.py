# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import numpy
import vtk
import csv


# %%
from IPython.display import Image
def vtk_show(renderer, width=400, height=300):
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data   = str(buffer(writer.GetResult()))

    return Image(data)


# %%
def createDummyRenderer():
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)

    camera = renderer.MakeCamera()
    camera.SetPosition(-256, -256, 512)
    camera.SetFocalPoint(0.0, 0.0, 255.0)
    camera.SetViewAngle(30.0)
    camera.SetViewUp(0.46, -0.80, -0.38)
    renderer.SetActiveCamera(camera)

    return renderer


# %%
l2n = lambda l: numpy.array(1)
n2l = lambda n: list(n)


# %%
# Path to the .mha file
filenameSegmentation = "./nac_brain_atlas/brain_segmentation.mha"

# Path to colorfile.txt 
filenameColorfile = "./nac_brain_atlas/colorfile.txt"

# Opacity of the different volumes (between 0.0 and 1.0)
volOpacityDef = 0.25


# %%
reader = vtk.vtkMetaImageReader()
reader.SetFileName(filenameSegmentation)

castFilter = vtk.vtkImageCast()
castFilter.SetInputConnection(reader.GetOutputPort())
castFilter.SetOutputScalarTypeToUnsignedShort()
castFilter.Update()

imdataBrainSeg = castFilter.GetOutputPort()


# %%
fid = open(filenameColorfile, "r")
reader = csv.reader(fid)

dictRGB = {}
for line in reader:
    dictRGB[int(line[0])] = [float(line[2])/255.0,
                             float(line[3])/255.0,
                             float(line[4])/255.0]
fid.close()


# %%
funcColor = vtk.vtkColorTransferFunction()

for idx in dictRGB.keys():
    funcColor.AddRGBPoint(idx,
                          dictRGB[idx][0],
                          dictRGB[idx][1],
                          dictRGB[idx][2])


# %%
funcOpacityScalar = vtk.vtkPiecewiseFunction()

for idx in dictRGB.keys():
    funcOpacityScalar.AddPoint(idx, volOpacityDef if idx != 0 else 0.0)


# %%
funcOpacityGradient = vtk.vtkPiecewiseFunction()

funcOpacityGradient.AddPoint(1, 0.0)
funcOpacityGradient.AddPoint(5, 0.1)
funcOpacityGradient.AddPoint(100, 1.0)


# %%
propVolume = vtk.vtkVolumeProperty()
propVolume.ShadeOff()
propVolume.SetColor(funcColor)
propVolume.SetScalarOpacity(funcOpacityScalar)
propVolume.SetGradientOpacity(funcOpacityGradient)
propVolume.SetInterpolationTypeToLinear()


# %%
# ## Volume Rendering


# %%
mapperVolume = vtk.vtkFixedPointVolumeRayCastMapper()
mapperVolume.SetInputConnection(imdataBrainSeg)

actorVolume = vtk.vtkVolume()
actorVolume.SetMapper(mapperVolume)
actorVolume.SetProperty(propVolume)

renderer = vtk.vtkRenderer()
renderWin = vtk.vtkRenderWindow()
renderWin.AddRenderer(renderer)
renderInteractor = vtk.vtkRenderWindowInteractor()
renderInteractor.SetRenderWindow(renderWin)

renderer.AddVolume(actorVolume)
renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("Black"))
renderWin.SetSize(800, 800)

renderWin.Render()
renderInteractor.Start()


# %%



