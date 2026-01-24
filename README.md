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
