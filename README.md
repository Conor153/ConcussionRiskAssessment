# Using Computer Vision to Assess the Risk of Concussion in American Football Collisions from Video Footage

This final year capstone project uses computer vision to detect and assess the risk of concussion in American football collisions. A custom trained YOLO11 model detects players, helmets and jerseys in video footage, with a homography transformation applied to calculate real-world speed, acceleration and G-Force from helmet movement. YOLO11 Pose estimation is then used to track the angular orientation of the head across frames, with left and right ear keypoint positions used to calculate angular velocity and acceleration. G-Force and angular acceleration are combined to generate a traffic light concussion risk classification of green, yellow or red.

## Requirments
- Python 3.12.3
- TypeScript
- Windows 11
- Linux Ubuntu 24.04 *(Recommended for AMD GPUs)*
- ROCm 6.4.2
- Pytorch 2.6
- RoboFlow
- AMD Radeon RX 7700 XT GPU


## Project Set Up
### 1. Create Virtual Environment
In Command Prompt or Terminal
**Windows:**  
```python -m venv .venv```

**Linux:**   
```python -m venv .venv```

### 2. Activate Environment on Command Prompt with
**Windows:**   
```.venv\Scripts\activate.bat```

**Linux:**   
```source .venv/bin/activate```

### 3. Deactivate Virtual Environment
```deactivate```

## Package Installation

### 1. Install for ROCm 6.4.2 on Linux
- ```wget https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb```
- ```sudo apt install ./amdgpu-install_6.4.60402-1_all.deb```
- ```sudo apt update```
- ```sudo apt install rocm```

### 2. Install for PyTorch 2.6 for ROCm 6.4.2
```
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torch-2.6.0%2Brocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl
```
```
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchvision-0.21.0%2Brocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl
```
```
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/pytorch_triton_rocm-3.2.0%2Brocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl
```
```
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchaudio-2.6.0%2Brocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl
```
```
pip3 uninstall torch torchvision pytorch-triton-rocm
```
```
pip3 install torch-2.6.0+rocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl torchvision-0.21.0+rocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl torchaudio-2.6.0+rocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl pytorch_triton_rocm-3.2.0+rocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl
```

### 3. Install Python dependencies
```
pip install -r requirements.txt
```

--- 

## Model Training 

### 1. Downlaod American Football Dataset
- Option 1

Download zipped folder from RoboFlow   
[https://app.roboflow.com/concussion-risk-analysis/nfl_collisions-kxmoy/3](https://app.roboflow.com/concussion-risk-analysis/nfl_collisions-kxmoy/3)

- Option 2 

Download on CoLab Notebook
```
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_OWN_API_KEY")
project = rf.workspace("concussion-risk-analysis").project("nfl_collisions-kxmoy")
version = project.version(3)
dataset = version.download("yolov11")
```

### 2. Initiate Training
**AMD GPU YOLO11n**
```
yolo detect train data=data.yaml model=yolo11n.pt epochs=50 imgsz=640 device=0 plots=True project=AMD name=AMDGPUv1TrainYOLON
```

**AMD GPU YOLO11s**
```
yolo detect train data=data.yaml model=yolo11s.pt epochs=50 imgsz=640 device=0 plots=True project=AMD name=AMDGPUv1TrainYOLOS
```

**AMD CPU**
```
yolo detect train data=data.yaml model=yolo11n.pt epochs=10 imgsz=640 device=cpu plots=True project=AMD name=AMDCPUv1TrainYOLON
```

**Google CoLab T4 GPU**
```
!yolo task=detect mode=train model=yolo11n.pt data=/content/datasets/NFL_Collisions-3/data.yaml epochs=50 imgsz=640
```


# Risk Classification

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Linear Acceleration (G-Force) | Below 49G | 49G – 79G | 80G or greater |
| Angular Acceleration ($`Radians/S^2`$) | Below 3512 | 3512 – 5874 | 5875 or greater |


## Testing and Evaluation

### 1. Unit Tests
To run unit tests navigate to backend directory
```
python testing.py
```

### 2. Model Results
| Metric        | AMD GPU YOLO11S | AMD GPU YOLO11N | Google CoLab YOLO11N | AMD Ryzen 5 7600 Series CPU YOLO11N|
|--------------|-----------------|-----------------|----------------------|---------|
| Precision     | 0.835           | 0.816           | 0.804                | 0.772   |
| Recall        | 0.75            | 0.699           | 0.709                | 0.627   |
| mAP 50        | 0.811           | 0.771           | 0.767                | 0.700   |
| mAP 50-95%    | 0.531           | 0.484           | 0.478                | 0.414   |
| Epochs        | 50              | 50              | 50                   | 10      |
| Time          | 0.234           | 0.143           | 0.459                | 0.296   |

