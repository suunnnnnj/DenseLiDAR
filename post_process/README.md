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

### Result

**no enhance**
![image](https://github.com/user-attachments/assets/6d3d0fc2-cbe5-4fba-a63d-1f74753b4373)

**enhance**
![post_processing_depth_enhance](https://github.com/user-attachments/assets/cc0ad1ee-fff9-4844-8424-3eaf1864a8de)

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

**Before Post Processing**
![image](https://github.com/user-attachments/assets/ad0534a8-1085-414a-ae58-a75c536077e8)

**After Post Processing**
![image](https://github.com/user-attachments/assets/edf059da-9b69-4e9d-a166-4666a08aa867)
