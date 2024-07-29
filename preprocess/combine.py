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

# PATH
point_cloud_image_path = 'gt_lidar.png'
depth_image = cv2.imread("dense_depth_output.png", cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread("demo_image.png")

height, width = depth_image.shape

# Create VTK data structures
points = vtk.vtkPoints()
vertices = vtk.vtkCellArray()
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName("Colors")

# Remove background
background_threshold_low = 1
background_threshold_high = 254
valid_pixels = np.where((depth_image >= background_threshold_low) & 
                        (depth_image <= background_threshold_high))

# Add points to VTK
for y, x in zip(valid_pixels[0], valid_pixels[1]):
    z = depth_image[y, x]
    # Scale depth value to real-world units if needed
    z_scaled = z / 255.0  # Example scaling factor (adjust as needed)
    
    # Convert depth image coordinates to 3D coordinates
    X = (x - cx) * z_scaled / fx
    Y = (y - cy) * z_scaled / fy
    Z = z_scaled  # Depth value as Z coordinate

    # Apply rotation and translation
    point = np.array([X, Y, Z])
    point_transformed = R @ point + T
    
    # Insert point into VTK structure
    point_id = points.InsertNextPoint(point_transformed[0], point_transformed[1], point_transformed[2])
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(point_id)

    # Get the color from the color image
    color = color_image[y, x]
    colors.InsertNextTuple3(color[2], color[1], color[0])  # BGR to RGB

# Create PolyData object and add Points, Vertices, Colors
poly_data = vtk.vtkPolyData()
poly_data.SetPoints(points)
poly_data.SetVerts(vertices)
poly_data.GetPointData().SetScalars(colors)

# Create Mapper and set PolyData
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(poly_data)

# Create Actor and set Mapper
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create Renderer, Render Window, and Interactor
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Add Actor to Renderer
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.3)

# Start rendering and interaction
render_window.Render()
render_window_interactor.Start()
