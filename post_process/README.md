# Post Processing & 3D visualization

## Post Processing

### Usage

**Specify file path**

- Specify file paths in post_processing.py
  
```
post_processing_depth_path = 'post_processing_depth.png'

predicted_depth_path = 'dense_depth_output.png'
pseudo_depth_path = 'demo_pseudo_depth.png'

```

**Running**

```
python post_process.py
```

## 3D visualization

### Installation

```
pip install vtk
```

### Usage

**Specify file path**

- Specify file paths in 3d_visualization.py
  
```
depth_image_path = '/home/mobiltech/SSDC/post_process/post_processing_depth.png'
color_image_path = '/home/mobiltech/SSDC/demo/demo_image.png'
```

**Running**

```
python 3d_visualization.py
```
### Result
![Screenshot from 2024-08-06 14-15-53](https://github.com/user-attachments/assets/d5c01af9-9c5e-4a82-82d1-b8281175b733)
![Screenshot from 2024-08-06 14-16-11](https://github.com/user-attachments/assets/3d3cc7f7-d0c0-4d68-b6e4-8478b73ec1ee)



