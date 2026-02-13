# Using Computer Vision to Assess the Risk of Concussion in American Football Collisions from Video Footage
Final Year Capstone using computer vision  to analyse the risk of concussion from American football clips

# Python Version
Python 3.11.7

# Set up Virtual Environment
Windows:  python -m venv .venv
Linux:    source venv/bin/activate

# Activate Environment on Command Prompt with
.venv\Scripts\activate.bat

# Packages to Install: 
- pip3 install opencv-python
- pip install ultralytics
- pip install scikit-learn
- pip install scikit-learn


# Deactivate Virtual Environment
deactivate

# Model Training 

# AMD Radeon RX 7700 XT
Requires

- Linux Ubuntu 24.04
- ROCm 6.4.2
- Pytorch 2.6
- Yolo

# Install for ROCm 6.4.2 on Linux
- wget https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
- sudo apt install ./amdgpu-install_6.4.60402-1_all.deb
- sudo apt update
- sudo apt install rocm

# Install for PyTorch 2.6 on Linux
- wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torch-2.6.0%2Brocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl
- wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchvision-0.21.0%2Brocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl
- wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/pytorch_triton_rocm-3.2.0%2Brocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl
- wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchaudio-2.6.0%2Brocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl
- pip3 uninstall torch torchvision pytorch-triton-rocm
- pip3 install torch-2.6.0+rocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl torchvision-0.21.0+rocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl torchaudio-2.6.0+rocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl pytorch_triton_rocm-3.2.0+rocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl
# Google CoLab Training


# Initiate Training
AMD GPU
- yolo detect train data=data.yaml model=yolo11n.pt epochs=50 imgsz=640 device=0 plots=True project=AMD name=AMDGPUv1TrainYOLON

AMD CPU
- yolo detect train data=data.yaml model=yolo11n.pt epochs=10 imgsz=640 device=cpu plots=True project=AMD name=AMDCPUv1TrainYOLON

Google CoLab T4 GPU
- 

### Model Results
| Metric        | AMD GPU YOLO11S | AMD GPU YOLO11N | Google CoLab YOLO11N | AMD Ryzen 5 7600 Series CPU YOLO11N|
|--------------|-----------------|-----------------|----------------------|---------|
| Precision     | 0.835           | 0.816           | 0.804                | 0.772   |
| Recall        | 0.75            | 0.699           | 0.709                | 0.627   |
| mAP 50        | 0.811           | 0.771           | 0.767                | 0.700   |
| mAP 50-95%    | 0.531           | 0.484           | 0.478                | 0.414   |
| Epochs        | 50              | 50              | 50                   | 10      |
| Time          | 0.234           | 0.143           | 0.459                | 0.296   |

# How is Concussion calculated
Concussion is calculated using 

| Metric    | Red | Yellow | Green |
| -------- | ------- | ------- | ------- |
| Linear Acceleration (G-Force)  |  70G or Greater   | Between 55G and 70G | |
| Angular Acceleration (Angular Rotation) | 4600 $`Radians^2`$ or Greater | Between 3000 $`Radians^2`$ and 4600 $`Radians^2`$ | |

# How is Concussion calculated
Python utilities.py