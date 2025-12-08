#!/bin/bash
# Download pretrained models for VideoPose3D and PoseFormerV2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "Downloading pretrained models..."
echo ""

# Create checkpoint directories
mkdir -p "$REPO_DIR/VideoPose3D/checkpoint"
mkdir -p "$REPO_DIR/PoseFormerV2/checkpoint"

# Download VideoPose3D pretrained models
echo "=========================================="
echo "VideoPose3D Models"
echo "=========================================="
cd "$REPO_DIR/VideoPose3D/checkpoint"

# CPN model (for Human3.6M evaluation)
if [ ! -f "pretrained_h36m_cpn.bin" ]; then
    echo "Downloading VideoPose3D CPN model..."
    wget -q --show-progress https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin
    echo "✓ VideoPose3D CPN model downloaded: pretrained_h36m_cpn.bin"
    echo "  Architecture: 3,3,3,3,3 (243 frames)"
    echo "  Expected MPJPE: 46.8 mm"
else
    echo "✓ VideoPose3D CPN model already exists: pretrained_h36m_cpn.bin"
fi

# Detectron COCO model (for video comparison with COCO keypoints)
if [ ! -f "pretrained_h36m_detectron_coco.bin" ]; then
    echo "Downloading VideoPose3D Detectron COCO model..."
    wget -q --show-progress https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin
    echo "✓ VideoPose3D Detectron COCO model downloaded: pretrained_h36m_detectron_coco.bin"
    echo "  Architecture: 3,3,3,3,3 (243 frames)"
    echo "  For use with COCO-format keypoints (e.g., from HRNet)"
else
    echo "✓ VideoPose3D Detectron COCO model already exists: pretrained_h36m_detectron_coco.bin"
fi

# Download PoseFormerV2 best pretrained model
echo ""
echo "=========================================="
echo "PoseFormerV2 Model"
echo "=========================================="

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip install gdown
fi

CHECKPOINT_DIR="$REPO_DIR/PoseFormerV2/checkpoint"
mkdir -p "$CHECKPOINT_DIR"
cd "$CHECKPOINT_DIR"

# Best model: 243 frames, 27/27, 45.2 mm MPJPE
FILE_ID="14SpqPyq9yiblCzTH5CorymKCUsXapmkg"
OUTPUT_NAME="243_27_27_45.2.bin"

if [ ! -f "$OUTPUT_NAME" ]; then
    echo "Downloading PoseFormerV2 best pretrained model..."
    echo "Model: Best performance (243 frames, 27/27)"
    echo "Expected MPJPE: 45.2 mm"
    echo ""
    gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUTPUT_NAME"
    echo "✓ Downloaded: $OUTPUT_NAME"
else
    echo "✓ PoseFormerV2 model already exists: $OUTPUT_NAME"
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo "All pretrained models are ready:"
echo "  - VideoPose3D CPN: VideoPose3D/checkpoint/pretrained_h36m_cpn.bin"
echo "  - VideoPose3D Detectron COCO: VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin"
echo "  - PoseFormerV2: PoseFormerV2/checkpoint/243_27_27_45.2.bin"
echo ""
echo "To evaluate models, run:"
echo "  ./eval_videopose3d.sh"
echo "  ./eval_poseformerv2.sh 243_27_27_45.2.bin 243 27 27"
