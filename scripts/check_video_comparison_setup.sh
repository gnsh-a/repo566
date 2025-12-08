#!/bin/bash
# Quick script to check if all prerequisites are available for video comparison

echo "Checking prerequisites for video comparison script..."
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MISSING=0

# Check video
VIDEO="$REPO_ROOT/PoseFormerV2/demo/video/sample_video.mp4"
if [ -f "$VIDEO" ]; then
    echo "✓ Sample video found: $VIDEO"
else
    echo "✗ Sample video not found: $VIDEO"
    MISSING=$((MISSING + 1))
fi

# Check VideoPose3D checkpoint
VP3D_CHECKPOINTS=(
    "$REPO_ROOT/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin"
    "$REPO_ROOT/VideoPose3D/checkpoint/pretrained_h36m_cpn.bin"
)
VP3D_FOUND=0
for ckpt in "${VP3D_CHECKPOINTS[@]}"; do
    if [ -f "$ckpt" ]; then
        echo "✓ VideoPose3D checkpoint found: $ckpt"
        VP3D_FOUND=1
        break
    fi
done
if [ $VP3D_FOUND -eq 0 ]; then
    echo "✗ VideoPose3D checkpoint not found"
    echo "  Download: https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
    echo "  Save to: VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin"
    MISSING=$((MISSING + 1))
fi

# Check PoseFormerV2 checkpoint
PF2_CHECKPOINTS=(
    "$REPO_ROOT/PoseFormerV2/checkpoint/243_27_27_45.2.bin"
    "$REPO_ROOT/PoseFormerV2/checkpoint/27_243_45.2.bin"
)
PF2_FOUND=0
for ckpt in "${PF2_CHECKPOINTS[@]}"; do
    if [ -f "$ckpt" ]; then
        echo "✓ PoseFormerV2 checkpoint found: $ckpt"
        PF2_FOUND=1
        break
    fi
done
if [ $PF2_FOUND -eq 0 ]; then
    echo "✗ PoseFormerV2 checkpoint not found"
    echo "  Download from: https://drive.google.com/file/d/14SpqPyq9yiblCzTH5CorymKCUsXapmkg/view?usp=share_link"
    echo "  Save to: PoseFormerV2/checkpoint/243_27_27_45.2.bin"
    MISSING=$((MISSING + 1))
fi

# Check HRNet/YOLOv3 weights
PF2_DEMO_CHECKPOINT="$REPO_ROOT/PoseFormerV2/demo/lib/checkpoint"
YOLO_WEIGHT="$PF2_DEMO_CHECKPOINT/yolov3.weights"
HRNET_WEIGHT="$PF2_DEMO_CHECKPOINT/pose_hrnet_w48_384x288.pth"

if [ -f "$YOLO_WEIGHT" ]; then
    echo "✓ YOLOv3 weight found: $YOLO_WEIGHT"
else
    echo "✗ YOLOv3 weight not found: $YOLO_WEIGHT"
    echo "  Download: https://drive.google.com/file/d/1YgA9riqm0xG2j72qhONi5oyiAxc98Y1N/view?usp=sharing"
    MISSING=$((MISSING + 1))
fi

if [ -f "$HRNET_WEIGHT" ]; then
    echo "✓ HRNet weight found: $HRNET_WEIGHT"
else
    echo "✗ HRNet weight not found: $HRNET_WEIGHT"
    echo "  Download: https://drive.google.com/file/d/1YLShFgDJt2Cs9goDw9BmR-UzFVgX3lc8/view?usp=sharing"
    MISSING=$((MISSING + 1))
fi

echo ""
if [ $MISSING -eq 0 ]; then
    echo "✓ All prerequisites are available!"
    echo "You can run: python scripts/compare_video_models.py --video PoseFormerV2/demo/video/sample_video.mp4"
else
    echo "✗ Missing $MISSING prerequisite(s). Please download the missing files above."
    exit 1
fi

