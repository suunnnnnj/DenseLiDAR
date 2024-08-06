import vtk
import numpy as np
import cv2

# Define intrinsic parameters (replace with actual values if known)
fx = 1000  # Focal length in x (in pixels)
fy = 1000  # Focal length in y (in pixels)
cx = 320   # Principal point x (in pixels)
cy = 240   # Principal point y (in pixels)

# Define rotation matrix R and translation vector T from calibration
R = np.array([
    [7.533745e-03, -9.999714e-01, -6.166020e-04],
    [1.480249e-02, 7.280733e-04, -9.998902e-01],
    [9.998621e-01, 7.523790e-03, 1.480755e-02]
])

T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

# Paths to depth and color images
depth_image_path = 'post_process/post_processing_depth.png'
color_image_path = '/home/mobiltech/Desktop/DenseLiDAR/demo/demo_image.png'

# Load the depth and color images
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread(color_image_path)

# Ensure color image is in RGB format
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# Image dimensions
height, width = depth_image.shape

# VTK data structures
points = vtk.vtkPoints()
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
vertices = vtk.vtkCellArray()

# Background removal thresholds
background_threshold_low = 1
background_threshold_high = 254

# Get valid pixels
valid_pixels = np.where((depth_image >= background_threshold_low) & 
                        (depth_image <= background_threshold_high))

# Iterate over valid pixels
for y, x in zip(valid_pixels[0], valid_pixels[1]):
    z = depth_image[y, x]
    z_scaled = z / 255.0  # Example scaling factor (adjust as needed)
    
    # Convert depth image coordinates to 3D coordinates
    X = (x - cx) * z_scaled / fx
    Y = (y - cy) * z_scaled / fy
    Z = z_scaled  # Depth value as Z coordinate
    
    # Apply rotation and translation
    point = np.array([X, Y, Z])
    point_transformed = R @ point + T
    
    # Color mapping
    color = color_image[y, x]  # BGR format; convert to RGB as needed
    colors.InsertNextTuple3(color[2], color[1], color[0])  # Insert RGB color
    
    # Insert point into VTK structure
    point_id = points.InsertNextPoint(point_transformed[0], point_transformed[1], point_transformed[2])
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(point_id)

# Create PolyData object and add Points, Colors, Vertices
poly_data = vtk.vtkPolyData()
poly_data.SetPoints(points)
poly_data.GetPointData().SetScalars(colors)  # Set color data
poly_data.SetVerts(vertices)

# Create a mapper and set the input
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(poly_data)

# Create an actor and set the mapper
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Set up the renderer, render window, and interactor
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Add the actor to the renderer and set the background color
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.3)  # Background color

# Render and interact
render_window.Render()
render_window_interactor.Start()
