#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
ENV_NAME="end2race"
PYTHON_VERSION="3.10"

echo "=================================================="
echo "Starting Environment Setup for End2Race"
echo "Environment name: ${ENV_NAME}"
echo "=================================================="

# 1. Create a new conda environment
echo "\n>>> Step 1: Creating conda environment..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
echo "Conda environment '${ENV_NAME}' created successfully."

# 2. Install PyTorch with CUDA 12.1 and Intel OpenMP to fix MKL linking issues
echo "\n>>> Step 2: Installing PyTorch with CUDA 12.1 and Intel OpenMP..."
conda run -n ${ENV_NAME} conda install pytorch torchvision torchaudio pytorch-cuda=12.1 intel-openmp -c pytorch -c nvidia -c conda-forge -y
echo "PyTorch and dependencies installed successfully."

# 3. Install other Python dependencies using pip
echo "\n>>> Step 3: Installing other Python dependencies..."
conda run -n ${ENV_NAME} pip install pandas scikit-learn "numpy<1.27" open3d numba tensorboard
echo "Python dependencies installed successfully."

# 4. Compile C++/CUDA extensions
echo "\n>>> Step 4: Compiling C++/CUDA extensions for PointPillars..."
conda run -n ${ENV_NAME} python setup.py build_ext --inplace
echo "Custom extensions compiled successfully."

echo "\n=================================================="
echo "Setup complete!"
echo "To activate the environment, run: conda activate ${ENV_NAME}"
echo "=================================================="
