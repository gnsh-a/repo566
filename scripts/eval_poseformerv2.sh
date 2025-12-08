#!/bin/bash
# Evaluation script for PoseFormerV2 on Human3.6M test set

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pose3d

# Navigate to PoseFormerV2 directory
cd "$(dirname "$0")/../PoseFormerV2" || exit

# Parameters
KEYPOINTS="cpn_ft_h36m_dbb"
SUBJECTS_TRAIN="S1,S5,S6,S7,S8"
SUBJECTS_TEST="S9,S11"
CHECKPOINT_DIR="checkpoint"
RESULTS_DIR="../results"

# Set GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get the latest checkpoint or use specified one, or use pretrained
if [ -z "$1" ]; then
    # Check for best pretrained model first
    if [ -f "$CHECKPOINT_DIR/243_27_27_45.2.bin" ]; then
        CHECKPOINT="243_27_27_45.2.bin"
        NUMBER_OF_FRAMES=243
        FRAME_KEPT=27
        COEFF_KEPT=27
        echo "Using best pretrained model: $CHECKPOINT"
    else
        # Check for any other checkpoint
        CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*.bin 2>/dev/null | head -1)
        if [ -z "$CHECKPOINT" ]; then
            echo "Error: No checkpoint found in $CHECKPOINT_DIR"
            echo "Usage: $0 [checkpoint_file.bin] [frame_length] [frame_kept] [coeff_kept]"
            echo "Or download pretrained model: ./download_pretrained.sh"
            exit 1
        fi
        CHECKPOINT=$(basename "$CHECKPOINT")
        
        # Try to infer parameters from checkpoint name
        # Format: [frames]_[frame_kept]_[coeff_kept]_[mpjpe].bin or similar
        if [[ "$CHECKPOINT" =~ ([0-9]+)_([0-9]+)_([0-9]+) ]]; then
            NUMBER_OF_FRAMES="${BASH_REMATCH[1]}"
            FRAME_KEPT="${BASH_REMATCH[2]}"
            COEFF_KEPT="${BASH_REMATCH[3]}"
        else
            # Default values (best pretrained config)
            NUMBER_OF_FRAMES=243
            FRAME_KEPT=27
            COEFF_KEPT=27
        fi
    fi
else
    CHECKPOINT="$1"
    # Use provided parameters or defaults (best model defaults)
    NUMBER_OF_FRAMES="${2:-243}"
    FRAME_KEPT="${3:-27}"
    COEFF_KEPT="${4:-27}"
fi

echo "Evaluating PoseFormerV2 with checkpoint: $CHECKPOINT"
echo "Sequence length: $NUMBER_OF_FRAMES frames"
echo "Frames kept: $FRAME_KEPT"
echo "Coeffs kept: $COEFF_KEPT"
echo "Test subjects: $SUBJECTS_TEST"

# Evaluate the model with reduced batch size to avoid OOM
python run_poseformer.py \
    -k $KEYPOINTS \
    -str $SUBJECTS_TRAIN \
    -ste $SUBJECTS_TEST \
    -c $CHECKPOINT_DIR \
    -frame $NUMBER_OF_FRAMES \
    -frame-kept $FRAME_KEPT \
    -coeff-kept $COEFF_KEPT \
    --evaluate "$CHECKPOINT" \
    -b 32 \
    -g 0 \
    > "$RESULTS_DIR/poseformerv2_eval_output.txt" 2>&1

echo "Evaluation completed!"
echo "Results saved to: $RESULTS_DIR/poseformerv2_eval_output.txt"

