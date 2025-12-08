# CS 566 Group Project: VideoPose3D vs PoseFormerV2 Comparison

This repository contains evaluation and comparison scripts for comparing VideoPose3D and PoseFormerV2 on the Human3.6M dataset.

**Train subjects:** S1, S5, S6, S7, S8  
**Test subjects:** S9, S11

## Quick Start

```bash
# 1. Setup environment
cd scripts
./setup_env.sh

# 2. Download pretrained models
./download_pretrained.sh

# 3. Evaluate both models
conda activate pose3d
./eval_videopose3d.sh
./eval_poseformerv2.sh 243_27_27_45.2.bin 243 27 27

# 4. Compare results
python compare_metrics.py
```

## Setup

### Environment Setup

```bash
cd scripts
./setup_env.sh
```

This creates a conda environment `pose3d` with Python 3.8 and installs all dependencies.

### Activate Environment

```bash
conda activate pose3d
```

## Evaluation

### Using Pretrained Models

```bash
cd scripts
./download_pretrained.sh  # Downloads both VideoPose3D and PoseFormerV2 models

# Evaluate VideoPose3D
./eval_videopose3d.sh

# Evaluate PoseFormerV2
./eval_poseformerv2.sh 243_27_27_45.2.bin 243 27 27
```

### Using Your Own Models

```bash
# VideoPose3D
./eval_videopose3d.sh [checkpoint_file.bin]

# PoseFormerV2
./eval_poseformerv2.sh [checkpoint_file.bin] [frame_length] [frame_kept] [coeff_kept]
```

## Compare Results

```bash
cd scripts
python compare_metrics.py
```

Generates `results/metrics.json` and `results/comparison_table.csv` with MPJPE, P-MPJPE, and N-MPJPE metrics.

## Video Comparison

Compare both models on custom videos:

```bash
cd scripts
conda activate pose3d

# Check prerequisites
./check_video_comparison_setup.sh

# Run comparison
python compare_video_models.py \
    --video path/to/video.mp4 \
    --videopose3d-checkpoint VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin \
    --poseformerv2-checkpoint PoseFormerV2/checkpoint/243_27_27_45.2.bin \
    --output-dir ../results/video_comparison \
    --gpu 0
```

**Prerequisites:**
- VideoPose3D Detectron COCO checkpoint (for COCO keypoints)
- PoseFormerV2 checkpoint
- HRNet and YOLOv3 weights (see `check_video_comparison_setup.sh`)

**Output:** Comparison video and 3D predictions saved to `results/video_comparison/<video_name>/`

## Training

### VideoPose3D

```bash
cd VideoPose3D
python run.py \
    -e 60 \
    -k cpn_ft_h36m_dbb \
    -arc 3,3,3 \
    -str S1,S5,S6,S7,S8 \
    -ste S9,S11 \
    -c checkpoint
```

### PoseFormerV2

```bash
cd PoseFormerV2
python run_poseformer.py \
    -e 200 \
    -k cpn_ft_h36m_dbb \
    -str S1,S5,S6,S7,S8 \
    -ste S9,S11 \
    -c checkpoint \
    -frame 81 \
    -g 0
```

## Pretrained Models

- **VideoPose3D**: `pretrained_h36m_cpn.bin` (46.8 mm MPJPE)
- **VideoPose3D Detectron COCO**: `pretrained_h36m_detectron_coco.bin` (for video comparison)
- **PoseFormerV2**: `243_27_27_45.2.bin` (45.2 mm MPJPE)

All models are downloaded automatically via `./download_pretrained.sh`.

## Troubleshooting

- **Data not found**: Run `./setup_env.sh` to set up data directories
- **Checkpoint not found**: Run `./download_pretrained.sh`
- **Import errors**: Ensure `conda activate pose3d` and re-run `./setup_env.sh` if needed
- **GPU errors**: Check GPU availability with `nvidia-smi`

## References

- VideoPose3D: https://github.com/facebookresearch/VideoPose3D
- PoseFormerV2: https://github.com/QitaoZhao/PoseFormerV2
