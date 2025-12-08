#!/bin/bash
# Setup script for VideoPose3D and PoseFormerV2 training environment

set -e

echo "Creating conda environment 'pose3d'..."
conda create -n pose3d python=3.8 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pose3d

echo "Installing PyTorch and basic dependencies..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

echo "Installing common dependencies..."
pip install numpy matplotlib scipy tqdm

echo "Installing VideoPose3D dependencies..."
# VideoPose3D doesn't have a requirements.txt, but needs basic packages
# Additional packages may be needed based on usage

echo "Installing PoseFormerV2 dependencies..."
cd ../PoseFormerV2
# Install essential dependencies (some packages in requirements.txt require Python 3.9+)
pip install einops torch-dct yacs timm || true
cd ..

echo "Setting up data directories..."
# Create data directories in both repositories
mkdir -p VideoPose3D/data
mkdir -p PoseFormerV2/data

# Link existing data files
if [ -d "../human3.6" ]; then
    echo "Linking Human3.6M data files..."
    ln -sf ../../human3.6/data_2d_h36m_cpn_ft_h36m_dbb.npz VideoPose3D/data/
    ln -sf ../../human3.6/data_2d_h36m_gt.npz VideoPose3D/data/
    ln -sf ../../human3.6/data_3d_h36m.npz VideoPose3D/data/
    
    ln -sf ../../human3.6/data_2d_h36m_cpn_ft_h36m_dbb.npz PoseFormerV2/data/
    ln -sf ../../human3.6/data_2d_h36m_gt.npz PoseFormerV2/data/
    ln -sf ../../human3.6/data_3d_h36m.npz PoseFormerV2/data/
else
    echo "Warning: human3.6 directory not found. Please ensure data files are in the correct location."
fi

echo "Creating checkpoint directories..."
mkdir -p VideoPose3D/checkpoint
mkdir -p PoseFormerV2/checkpoint

echo "Setup complete!"
echo "To activate the environment, run: conda activate pose3d"

