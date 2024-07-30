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

![Screenshot from 2024-07-30 15-58-49](https://github.com/user-attachments/assets/7c1151c1-2bb8-4cc7-a348-06a86f318f38)

