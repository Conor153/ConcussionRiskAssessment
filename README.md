# ConcussionRiskAssesment
Final Year Capstone computer vision project to analyse the risk of concussion from NFL clips


# Environment Setup  Windows
Python 3.11.7

# Create Environment
python -m venv .venv

# Activate Environment on Command Prompt with
.\.venv\Scripts\activate.bat


# Packages : Note Deveplment device GPU AMD Radeon RX 7700 XT
- pip3 install opencv-python
- pip install ultralytics
- pip install scikit-learn

# AMD Device
Requires
driver 25.11.1
RX 7700 XT
ROCM 7.10.0 preview
- python -m pip install --index-url https://repo.amd.com/rocm/whl gfx110X-dgpu/ "rocm[libraries,devel]"
python -m pip install --index-url https://repo.amd.com/rocm/whl/gfx110X-dgpu/ torch torchvision torchaudio

- # Deactivate on Command Prompt with with
deactivate


# Environment Setup Linux Ubuntu 24.04

AMD Radeon RX 7700 XT
ROCm 6.4.2
Pytorch 2.6


# Set up Virtual Environment
source venv/bin/activate

# Install for ROCm 6.4.2
wget https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
sudo apt install ./amdgpu-install_6.4.60402-1_all.deb
sudo apt update
sudo apt install rocm

# Install for PyTorch 2.6
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torch-2.6.0%2Brocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchvision-0.21.0%2Brocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/pytorch_triton_rocm-3.2.0%2Brocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/torchaudio-2.6.0%2Brocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl
pip3 uninstall torch torchvision pytorch-triton-rocm
pip3 install torch-2.6.0+rocm6.4.2.git76481f7c-cp312-cp312-linux_x86_64.whl torchvision-0.21.0+rocm6.4.2.git4040d51f-cp312-cp312-linux_x86_64.whl torchaudio-2.6.0+rocm6.4.2.gitd8831425-cp312-cp312-linux_x86_64.whl pytorch_triton_rocm-3.2.0+rocm6.4.2.git7e948ebf-cp312-cp312-linux_x86_64.whl

#Results
AMD CPU
10 epochs completed in 0.296 hours.
Ultralytics 8.4.8 🚀 Python-3.12.3 torch-2.6.0+rocm6.4.2.git76481f7c CPU (AMD Ryzen 5 7600X 6-Core Processor)
YOLO11n summary (fused): 101 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 1.1it/s 3.5s
                   all        109       3293      0.772      0.627        0.7      0.414
                Helmet        109        993      0.735       0.51      0.575      0.298
                Jersey        109       1112      0.769      0.638      0.723       0.41
                Player        109       1188      0.813      0.733      0.802      0.534



# AMD GPU
# YOLO11S
50 epochs completed in 0.234 hours.
Ultralytics 8.4.8 🚀 Python-3.12.3 torch-2.6.0+rocm6.4.2.git76481f7c CUDA:0 (AMD Radeon RX 7700 XT, 12272MiB)
YOLO11s summary (fused): 101 layers, 9,413,961 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 2                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 5                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 7                 Class     Images  Instances      Box(                                  P          R       mAP50    mAP50-95): 100% ━━━━━━━━━━━━ 4/4 3.2it/s 1.3s
                   all        109       3293      0.835       0.75      0.811      0.531
                Helmet        109        993      0.798      0.679      0.722      0.425
                Jersey        109       1112      0.839      0.769      0.834      0.541
                Player        109       1188      0.868      0.801      0.878      0.628

#YOLO11N
50 epochs completed in 0.143 hours.
Optimizer stripped from /home/conor/Desktop/ConcussionRiskAssesment/runs/detect/AMD/AMDGPUv1Train7/weights/best.pt, 5.4MB
Ultralytics 8.4.8 🚀 Python-3.12.3 torch-2.6.0+rocm6.4.2.git76481f7c CUDA:0 (AMD Radeon RX 7700 XT, 12272MiB)
YOLO11n summary (fused): 101 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 3.4it/s 1.2s
                   all        109       3293      0.816      0.699      0.771      0.484
                Helmet        109        993      0.786      0.592      0.666      0.376
                Jersey        109       1112      0.822      0.726      0.803       0.49
                Player        109       1188       0.84      0.778      0.845      0.584

Google CoLab



