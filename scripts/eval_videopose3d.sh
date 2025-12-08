#!/bin/bash
# Evaluation script for VideoPose3D on Human3.6M test set

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pose3d

# Navigate to VideoPose3D directory
cd "$(dirname "$0")/../VideoPose3D" || exit

# Parameters
KEYPOINTS="cpn_ft_h36m_dbb"
SUBJECTS_TRAIN="S1,S5,S6,S7,S8"
SUBJECTS_TEST="S9,S11"
CHECKPOINT_DIR="checkpoint"
RESULTS_DIR="../results"

# Get the latest checkpoint or use specified one, or use pretrained
if [ -z "$1" ]; then
    # Check for pretrained model first
    if [ -f "$CHECKPOINT_DIR/pretrained_h36m_cpn.bin" ]; then
        CHECKPOINT="pretrained_h36m_cpn.bin"
        ARCHITECTURE="3,3,3,3,3"  # Pretrained model uses 5 blocks (243 frames)
        echo "Using pretrained model: $CHECKPOINT"
    else
        CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*.bin 2>/dev/null | head -1)
        if [ -z "$CHECKPOINT" ]; then
            echo "Error: No checkpoint found in $CHECKPOINT_DIR"
            echo "Usage: $0 [checkpoint_file.bin]"
            echo "Or download pretrained model: ./download_pretrained.sh"
            exit 1
        fi
        CHECKPOINT=$(basename "$CHECKPOINT")
        ARCHITECTURE="3,3,3"  # Default for custom trained models
    fi
else
    CHECKPOINT="$1"
    # Determine architecture based on checkpoint name
    if [[ "$CHECKPOINT" == *"pretrained"* ]]; then
        ARCHITECTURE="3,3,3,3,3"
    else
        ARCHITECTURE="3,3,3"
    fi
fi

echo "Evaluating VideoPose3D with checkpoint: $CHECKPOINT"
echo "Architecture: $ARCHITECTURE"
echo "Test subjects: $SUBJECTS_TEST"

# Evaluate the model
python run.py \
    -k $KEYPOINTS \
    -arc $ARCHITECTURE \
    -str $SUBJECTS_TRAIN \
    -ste $SUBJECTS_TEST \
    -c $CHECKPOINT_DIR \
    --evaluate "$CHECKPOINT" \
    > "$RESULTS_DIR/videopose3d_eval_output.txt" 2>&1

echo "Evaluation completed!"
echo "Results saved to: $RESULTS_DIR/videopose3d_eval_output.txt"

