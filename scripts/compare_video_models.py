#!/usr/bin/env python3
"""
Unified script to compare VideoPose3D and PoseFormerV2 on the same video.
Uses HRNet for 2D keypoint detection and runs both models for fair comparison.
"""

import sys
import os
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import copy
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, writers
from multiprocessing import Pool, cpu_count
from functools import partial

# Add paths for both repositories
# Get absolute path to script, then go up two levels to repo root
script_path = Path(__file__).resolve()
repo_root = script_path.parent.parent
original_cwd = os.getcwd()

# Import from PoseFormerV2 (need to change directory)
pf2_dir = repo_root / "PoseFormerV2"
if not pf2_dir.exists():
    raise FileNotFoundError(f"PoseFormerV2 directory not found: {pf2_dir}")
os.chdir(str(pf2_dir))
sys.path.insert(0, str(pf2_dir))
sys.path.insert(0, str(pf2_dir / "demo"))  # Add demo directory for lib imports

from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from lib.preprocess import h36m_coco_format, revise_kpts
from common.model_poseformer import PoseTransformerV2 as PoseFormerV2Model
from common.camera import normalize_screen_coordinates, camera_to_world
from common.h36m_dataset import h36m_skeleton

os.chdir(original_cwd)

# Import from VideoPose3D (need to change directory)
vp3d_dir = repo_root / "VideoPose3D"
if not vp3d_dir.exists():
    raise FileNotFoundError(f"VideoPose3D directory not found: {vp3d_dir}")
os.chdir(str(vp3d_dir))
sys.path.insert(0, str(vp3d_dir))

from common.skeleton import Skeleton
from common.camera import normalize_screen_coordinates as vp3d_normalize
from common.camera import image_coordinates
from common.camera import camera_to_world as vp3d_camera_to_world
from common.custom_dataset import CustomDataset
from common.generators import UnchunkedGenerator
from common.model import TemporalModel
from data.data_utils import suggest_metadata

os.chdir(original_cwd)

# Human3.6M joint names (17 joints)
H36M_JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck/Nose', 'Head', 'LShoulder', 'LElbow',
    'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

def get_video_info(video_path):
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count
    }

def extract_2d_keypoints_hrnet(video_path, output_dir):
    """
    Extract 2D keypoints using HRNet (from PoseFormerV2 demo).
    Returns both COCO format (for VideoPose3D) and H36M format (for PoseFormerV2).
    """
    print("\n" + "="*60)
    print("Step 1: Extracting 2D keypoints using HRNet")
    print("="*60)
    
    # Change to PoseFormerV2 directory for HRNet to work properly
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    pf2_dir = repo_root / "PoseFormerV2"
    original_cwd = os.getcwd()
    
    # Check for required HRNet/YOLOv3 weights
    pf2_demo_checkpoint = pf2_dir / "demo/lib/checkpoint"
    yolo_weight = pf2_demo_checkpoint / "yolov3.weights"
    hrnet_weight = pf2_demo_checkpoint / "pose_hrnet_w48_384x288.pth"
    
    if not yolo_weight.exists() or not hrnet_weight.exists():
        print("\nWARNING: HRNet/YOLOv3 weights not found!")
        print(f"Expected location: {pf2_demo_checkpoint}")
        print("Please download:")
        print("  - YOLOv3: https://drive.google.com/file/d/1YgA9riqm0xG2j72qhONi5oyiAxc98Y1N/view?usp=sharing")
        print("  - HRNet: https://drive.google.com/file/d/1YLShFgDJt2Cs9goDw9BmR-UzFVgX3lc8/view?usp=sharing")
        print("\nContinuing anyway (may fail if weights are missing)...\n")
    
    # Change to PoseFormerV2 directory for HRNet
    os.chdir(str(pf2_dir))
    
    try:
        # Generate keypoints using HRNet (COCO format)
        # HRNet's gen_video_kpts and YOLOv3's load_model call parse_args() internally, 
        # which conflicts with our args. We need to patch both argument parsers.
        original_argv = sys.argv.copy()
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        
        # Import YOLOv3 modules to patch them
        from lib.yolov3.human_detector import arg_parse as yolo_arg_parse
        from types import SimpleNamespace
        
        # Create a mock args object for YOLOv3
        pf2_demo_checkpoint = pf2_dir / "demo/lib/checkpoint"
        yolo_args = SimpleNamespace(
            confidence=0.70,
            nms_thresh=0.4,
            reso=416,
            weight_file=str(pf2_demo_checkpoint / 'yolov3.weights'),
            cfg_file=str(pf2_dir / 'demo/lib/yolov3/cfg/yolov3.cfg'),
            animation=False,
            video=str(video_path),
            image=str(pf2_dir / 'demo/lib/yolov3/data/dog-cycle-car.png'),
            num_person=1,
            gpu=gpu_id
        )
        
        # Patch YOLOv3's arg_parse to return our mock args
        import lib.yolov3.human_detector as yolo_module
        original_yolo_arg_parse = yolo_module.arg_parse
        yolo_module.arg_parse = lambda: yolo_args
        
        # Set sys.argv with only HRNet-compatible arguments
        sys.argv = [
            sys.argv[0], 
            '--video', str(video_path),
            '--det-dim', '416',
            '--num-person', '1',
            '--gpu', gpu_id,
            '--thred-score', '0.30',
        ]
        
        print("Running HRNet detection...")
        keypoints_coco, scores = hrnet_pose(str(video_path), det_dim=416, num_peroson=1, gen_output=True)
    finally:
        # Restore original sys.argv, YOLOv3 arg_parse, and directory
        sys.argv = original_argv
        if 'yolo_module' in locals():
            yolo_module.arg_parse = original_yolo_arg_parse
        os.chdir(original_cwd)
    # keypoints_coco shape: (num_persons, num_frames, 17, 2) in COCO format, pixel coordinates
    
    # Convert to H36M format for PoseFormerV2
    print("Converting to H36M format...")
    keypoints_h36m, scores_h36m, valid_frames = h36m_coco_format(keypoints_coco, scores)
    keypoints_h36m = revise_kpts(keypoints_h36m, scores_h36m, valid_frames)
    # keypoints_h36m shape: (num_persons, num_frames, 17, 2) in H36M format, pixel coordinates
    
    # Save both formats
    os.makedirs(output_dir / "2d_keypoints", exist_ok=True)
    
    # COCO format for VideoPose3D (pixel coordinates, first person only)
    coco_kpts = keypoints_coco[0]  # (num_frames, 17, 2)
    np.savez_compressed(output_dir / "2d_keypoints" / "coco_format.npz", 
                       keypoints=coco_kpts, scores=scores[0])
    
    # H36M format for PoseFormerV2 (pixel coordinates, will be normalized later)
    np.savez_compressed(output_dir / "2d_keypoints" / "h36m_format.npz",
                       reconstruction=keypoints_h36m)
    
    print(f"✓ Saved COCO format: {output_dir / '2d_keypoints' / 'coco_format.npz'}")
    print(f"✓ Saved H36M format: {output_dir / '2d_keypoints' / 'h36m_format.npz'}")
    
    return coco_kpts, keypoints_h36m, get_video_info(video_path)

def prepare_videopose3d_data(coco_keypoints, video_info, output_dir):
    """
    Prepare data for VideoPose3D inference.
    Creates custom dataset format expected by VideoPose3D.
    """
    print("\n" + "="*60)
    print("Step 2: Preparing VideoPose3D data")
    print("="*60)
    
    # Create custom dataset format
    video_name = "sample_video"
    metadata = suggest_metadata('coco')
    metadata['video_metadata'] = {
        video_name: {
            'w': video_info['width'],
            'h': video_info['height']
        }
    }
    
    # Format: positions_2d[video_name]['custom'] = [keypoints]
    # keypoints shape: (num_frames, 17, 2) in pixel coordinates
    output_data = {
        video_name: {
            'custom': [coco_keypoints.astype('float32')]
        }
    }
    
    # Save as VideoPose3D expects
    output_file = output_dir / "2d_keypoints" / "data_2d_custom_sample_video.npz"
    np.savez_compressed(output_file, 
                       positions_2d=output_data, 
                       metadata=metadata)
    
    print(f"✓ Prepared VideoPose3D dataset: {output_file}")
    return output_file, video_name

def run_videopose3d_inference(data_file, video_name, checkpoint_path, output_dir, video_info):
    """
    Run VideoPose3D inference on the prepared data.
    """
    print("\n" + "="*60)
    print("Step 3: Running VideoPose3D inference")
    print("="*60)
    
    # Load custom dataset
    dataset = CustomDataset(str(data_file), remove_static_joints=True)
    
    # Load model
    print(f"Loading VideoPose3D model from {checkpoint_path}...")
    model_pos = TemporalModel(
        num_joints_in=17,
        in_features=2,
        num_joints_out=17,
        filter_widths=[3, 3, 3, 3, 3],  # 243 frame receptive field
        causal=False,
        dropout=0.25,
        channels=1024,
        dense=False
    )
    
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_pos.load_state_dict(checkpoint['model_pos'])
    model_pos.eval()
    
    # Get keypoints
    keypoints = np.load(str(data_file), allow_pickle=True)['positions_2d'].item()
    input_keypoints = keypoints[video_name]['custom'][0].copy()
    
    # Ensure keypoints are in correct format: (num_frames, 17, 2) with (x, y) coordinates
    # HRNet outputs pixel coordinates with origin at top-left: (x, y) where x in [0, width], y in [0, height]
    assert input_keypoints.shape[-1] >= 2, f"Expected at least 2 coordinates, got {input_keypoints.shape[-1]}"
    input_keypoints = input_keypoints[..., :2].copy()  # Take only x, y coordinates
    
    # Get camera parameters
    cam = dataset.cameras()[video_name][0]
    w, h = cam['res_w'], cam['res_h']
    
    # Debug: Print coordinate ranges before normalization
    x_min, x_max = input_keypoints[:, :, 0].min(), input_keypoints[:, :, 0].max()
    y_min, y_max = input_keypoints[:, :, 1].min(), input_keypoints[:, :, 1].max()
    print(f"Keypoint coordinate ranges before normalization:")
    print(f"  x: [{x_min:.2f}, {x_max:.2f}] (expected: [0, {w}])")
    print(f"  y: [{y_min:.2f}, {y_max:.2f}] (expected: [0, {h}])")
    
    # Note: VideoPose3D expects standard image coordinates (origin at top-left)
    # The normalization function handles the mapping to [-1, 1] range correctly.
    # No Y-flipping is needed.
    
    # Ensure coordinates are within valid pixel range (clip to image bounds)
    input_keypoints[:, :, 0] = np.clip(input_keypoints[:, :, 0], 0, w)
    input_keypoints[:, :, 1] = np.clip(input_keypoints[:, :, 1], 0, h)
    
    # Normalize keypoints using VideoPose3D's normalization function
    # Formula: X/w*2 - [1, h/w]
    # This maps: x: [0, w] -> [-1, 1], y: [0, h] -> [-h/w, h/w] (preserves aspect ratio)
    input_keypoints_norm = vp3d_normalize(input_keypoints, w=w, h=h)
    
    # Debug: Print coordinate ranges after normalization
    x_norm_min, x_norm_max = input_keypoints_norm[:, :, 0].min(), input_keypoints_norm[:, :, 0].max()
    y_norm_min, y_norm_max = input_keypoints_norm[:, :, 1].min(), input_keypoints_norm[:, :, 1].max()
    print(f"Keypoint coordinate ranges after normalization:")
    print(f"  x: [{x_norm_min:.4f}, {x_norm_max:.4f}] (expected: [-1, 1])")
    print(f"  y: [{y_norm_min:.4f}, {y_norm_max:.4f}] (expected: [-{h/w:.4f}, {h/w:.4f}])")
    
    # Create generator
    pad = (243 - 1) // 2  # 243 frame receptive field
    kps_left = list(dataset.skeleton().joints_left())
    kps_right = list(dataset.skeleton().joints_right())
    
    gen = UnchunkedGenerator(None, None, [input_keypoints_norm],
                            pad=pad, causal_shift=0, augment=False,
                            kps_left=kps_left, kps_right=kps_right,
                            joints_left=kps_left, joints_right=kps_right)
    
    # Run inference
    print("Running inference...")
    predictions_3d = []
    with torch.no_grad():
        for batch_cam, batch_3d, batch_2d in gen.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
            
            predicted_3d = model_pos(inputs_2d)
            predictions_3d.append(predicted_3d.cpu().numpy())
    
    # Concatenate predictions
    prediction = np.concatenate(predictions_3d, axis=0)
    prediction = prediction.squeeze()  # (num_frames, 17, 3)
    
    # FIX: Apply coordinate transformation to fix coordinate system mismatch
    # VideoPose3D outputs are in camera space, need to transform to world space
    # Use the same camera_to_world transformation that PoseFormerV2 uses
    # This matches the transformation used in VideoPose3D's visualization code
    print("Applying coordinate transformation: camera space -> world space")
    print("  - Using camera_to_world transformation with same rotation as PoseFormerV2")
    # Use the same rotation quaternion that PoseFormerV2 uses (from custom_camera_params)
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], 
                   dtype='float32')
    # Transform from camera space to world space (handles all frames at once)
    prediction = vp3d_camera_to_world(prediction, R=rot, t=0)
    # Rebase height (same as VideoPose3D visualization)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    
    # Save predictions (transformed to match PoseFormerV2 coordinate system)
    output_file = output_dir / "3d_predictions" / "videopose3d.npy"
    os.makedirs(output_dir / "3d_predictions", exist_ok=True)
    np.save(output_file, prediction)
    
    print(f"✓ VideoPose3D inference complete: {output_file}")
    print(f"  Prediction shape: {prediction.shape}")
    
    return prediction

def run_poseformerv2_inference(h36m_keypoints, checkpoint_path, output_dir, video_path, video_info):
    """
    Run PoseFormerV2 inference on the prepared data.
    """
    print("\n" + "="*60)
    print("Step 4: Running PoseFormerV2 inference")
    print("="*60)
    
    # Setup model args
    class Args:
        embed_dim_ratio = 32
        depth = 4
        frames = 243
        number_of_kept_frames = 27
        number_of_kept_coeffs = 27
        pad = (243 - 1) // 2
        n_joints = 17
        out_joints = 17
    
    args = Args()
    
    # Load model
    print(f"Loading PoseFormerV2 model from {checkpoint_path}...")
    model = nn.DataParallel(PoseFormerV2Model(args=args))
    if torch.cuda.is_available():
        model = model.cuda()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_pos' in checkpoint:
        model.load_state_dict(checkpoint['model_pos'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    # Get keypoints (first person)
    keypoints = h36m_keypoints[0]  # (num_frames, 17, 2) in pixel coordinates
    video_length = len(keypoints)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    predictions_3d = []
    
    print("Running inference...")
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape
        
        # Get input frames (243 frame window)
        start = max(0, i - args.pad)
        end = min(i + args.pad, len(keypoints) - 1)
        input_2D_no = keypoints[start:end+1]
        
        # Pad if needed
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints) - 1)
            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        # Normalize
        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])
        
        # Test-time augmentation (flip)
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        
        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), 
                                   np.expand_dims(input_2D_aug, axis=0)), 0)
        input_2D = input_2D[np.newaxis, :, :, :, :]
        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        
        # Inference
        with torch.no_grad():
            output_3D_non_flip = model(input_2D[:, 0])
            output_3D_flip = model(input_2D[:, 1])
        
        # Average flip and non-flip
        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = \
            output_3D_flip[:, :, joints_right + joints_left, :]
        output_3D = (output_3D_non_flip + output_3D_flip) / 2
        
        # Extract prediction for center frame
        output_3D[:, :, 0, :] = 0  # Root joint at origin
        post_out = output_3D[0, 0].cpu().detach().numpy()
        
        # Transform to world space (same as vis.py)
        rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])
        
        predictions_3d.append(post_out)
    
    cap.release()
    
    # Convert to numpy array
    prediction = np.array(predictions_3d)  # (num_frames, 17, 3)
    
    # Save predictions
    output_file = output_dir / "3d_predictions" / "poseformerv2.npy"
    os.makedirs(output_dir / "3d_predictions", exist_ok=True)
    np.save(output_file, prediction)
    
    print(f"✓ PoseFormerV2 inference complete: {output_file}")
    print(f"  Prediction shape: {prediction.shape}")
    
    return prediction

def compute_per_joint_differences(pred_vp3d, pred_pf2, output_dir):
    """
    Compute per-joint differences between the two predictions.
    Note: Coordinate systems may differ, so we compute relative differences.
    """
    print("\n" + "="*60)
    print("Step 5: Computing per-joint differences")
    print("="*60)
    
    # Align by root joint (subtract root from all joints)
    pred_vp3d_aligned = pred_vp3d - pred_vp3d[:, 0:1, :]  # Root is joint 0
    pred_pf2_aligned = pred_pf2 - pred_pf2[:, 0:1, :]
    
    # Compute per-joint Euclidean distances
    differences = np.linalg.norm(pred_vp3d_aligned - pred_pf2_aligned, axis=2)  # (num_frames, 17)
    
    # Statistics per joint
    joint_stats = {}
    for j, joint_name in enumerate(H36M_JOINT_NAMES):
        joint_diffs = differences[:, j]
        joint_stats[joint_name] = {
            'mean': float(np.mean(joint_diffs)),
            'std': float(np.std(joint_diffs)),
            'min': float(np.min(joint_diffs)),
            'max': float(np.max(joint_diffs)),
            'median': float(np.median(joint_diffs))
        }
    
    # Overall statistics
    overall_stats = {
        'mean': float(np.mean(differences)),
        'std': float(np.std(differences)),
        'min': float(np.min(differences)),
        'max': float(np.max(differences)),
        'median': float(np.median(differences))
    }
    
    # Save metrics
    metrics = {
        'per_joint': joint_stats,
        'overall': overall_stats,
        'note': 'Differences computed after root alignment. Coordinate systems may differ (VideoPose3D: camera space, PoseFormerV2: world space).'
    }
    
    metrics_file = output_dir / "3d_predictions" / "comparison_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics: {metrics_file}")
    print(f"\nOverall difference statistics:")
    print(f"  Mean: {overall_stats['mean']:.4f}")
    print(f"  Std:  {overall_stats['std']:.4f}")
    print(f"  Min:  {overall_stats['min']:.4f}")
    print(f"  Max:  {overall_stats['max']:.4f}")
    
    return metrics, differences

def draw_coordinate_axes(ax, scale, corner='bottom-left-front'):
    """
    Draw XYZ coordinate axes on a 3D plot in an outside corner.
    
    Args:
        ax: 3D matplotlib axis
        scale: Length of axes (used to determine corner position)
        corner: Which corner to place axes ('bottom-left-front', 'bottom-right-back', etc.)
    """
    # Calculate corner position (bottom-left-front corner: negative x, negative y, positive z)
    # Position at 80% of the scale from center to corner
    corner_offset = scale * 0.8
    origin = [-corner_offset, -corner_offset, corner_offset]
    
    # Define axis colors
    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axis_length = scale * 0.2  # Length of each axis line
    
    for i, (color, label, direction) in enumerate(zip(colors, labels, directions)):
        # Draw axis line
        end_point = [origin[j] + direction[j] * axis_length for j in range(3)]
        ax.plot([origin[0], end_point[0]], 
                [origin[1], end_point[1]], 
                [origin[2], end_point[2]], 
                color=color, linewidth=2.5, alpha=0.9)
        
        # Add axis label
        label_pos = [origin[j] + direction[j] * axis_length * 1.2 for j in range(3)]
        ax.text(label_pos[0], label_pos[1], label_pos[2], label, 
                color=color, fontsize=11, fontweight='bold')

def generate_single_frame(args_tuple):
    """
    Generate a single comparison frame. Designed for parallel processing.
    
    Args:
        args_tuple: Tuple of (i, frame_rgb, kpts, pred_vp3d_aligned, pred_pf2_aligned, 
                             fixed_scale, parents, joints_left, joints_right, 
                             frames_dir, num_frames, frame_diff)
    
    Returns:
        (frame_idx, success): Frame index and whether generation succeeded
    """
    (i, frame_rgb, kpts, pred_vp3d_aligned, pred_pf2_aligned, fixed_scale, 
     parents, joints_left, joints_right, frames_dir, num_frames, frame_diff) = args_tuple
    
    try:
        # Use a smaller figure size and lower DPI for faster rendering
        fig = plt.figure(figsize=(20, 6), dpi=80)  # Reduced from 100 to 80
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        
        # Original video with 2D overlay
        ax1 = plt.subplot(gs[0])
        ax1.imshow(frame_rgb)
        ax1.set_title('Original + 2D Keypoints', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Draw 2D skeleton (pre-computed connections)
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                      [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                      [8, 14], [14, 15], [15, 16]]
        for conn in connections:
            if kpts[conn[0], 0] > 0 and kpts[conn[1], 0] > 0:  # Valid keypoints
                ax1.plot([kpts[conn[0], 0], kpts[conn[1], 0]],
                        [kpts[conn[0], 1], kpts[conn[1], 1]], 'r-', linewidth=2)
        ax1.scatter(kpts[:, 0], kpts[:, 1], c='yellow', s=30, zorder=10)
        
        # VideoPose3D 3D visualization
        ax2 = plt.subplot(gs[1], projection='3d')
        ax2.view_init(elev=15., azim=70)
        pos = pred_vp3d_aligned.copy()
        
        for j, j_parent in enumerate(parents):
            if j_parent == -1:
                continue
            col = 'red' if j in joints_right else 'black'
            ax2.plot([pos[j, 0], pos[j_parent, 0]],
                    [pos[j, 1], pos[j_parent, 1]],
                    [pos[j, 2], pos[j_parent, 2]], c=col, linewidth=2)
        ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=50)
        ax2.set_title('VideoPose3D', fontsize=12, fontweight='bold')
        ax2.set_xlim([-fixed_scale, fixed_scale])
        ax2.set_ylim([-fixed_scale, fixed_scale])
        ax2.set_zlim([-fixed_scale, fixed_scale])
        draw_coordinate_axes(ax2, fixed_scale)
        ax2.set_xlabel('X', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax2.set_zlabel('Z', fontsize=10, fontweight='bold')
        
        # PoseFormerV2 3D visualization
        ax3 = plt.subplot(gs[2], projection='3d')
        ax3.view_init(elev=15., azim=70)
        pos = pred_pf2_aligned.copy()
        
        for j, j_parent in enumerate(parents):
            if j_parent == -1:
                continue
            col = 'red' if j in joints_right else 'black'
            ax3.plot([pos[j, 0], pos[j_parent, 0]],
                    [pos[j, 1], pos[j_parent, 1]],
                    [pos[j, 2], pos[j_parent, 2]], c=col, linewidth=2)
        ax3.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='green', s=50)
        ax3.set_title('PoseFormerV2', fontsize=12, fontweight='bold')
        ax3.set_xlim([-fixed_scale, fixed_scale])
        ax3.set_ylim([-fixed_scale, fixed_scale])
        ax3.set_zlim([-fixed_scale, fixed_scale])
        draw_coordinate_axes(ax3, fixed_scale)
        ax3.set_xlabel('X', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax3.set_zlabel('Z', fontsize=10, fontweight='bold')
        
        # Add per-joint difference text
        mean_diff = np.mean(frame_diff)
        diff_text = f"Mean diff: {mean_diff:.4f}"
        fig.suptitle(f"Frame {i+1}/{num_frames} - {diff_text}", 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(frames_dir / f"frame_{i:04d}.png", dpi=80, bbox_inches='tight')
        plt.close(fig)
        return (i, True)
    except Exception as e:
        print(f"Error generating frame {i}: {e}")
        return (i, False)

def create_comparison_visualization(video_path, pred_vp3d, pred_pf2, coco_keypoints, 
                                   differences, output_dir, video_info, num_workers=None):
    """
    Create side-by-side comparison video: Original | VideoPose3D | PoseFormerV2
    with per-joint difference overlay.
    """
    print("\n" + "="*60)
    print("Step 6: Creating comparison visualization")
    print("="*60)
    
    # Load skeleton
    skeleton = h36m_skeleton
    parents = skeleton.parents()
    joints_left = skeleton.joints_left()
    joints_right = skeleton.joints_right()
    
    # Read video and pre-convert frames to RGB (faster than converting per frame)
    print("Loading video frames...")
    cap = cv2.VideoCapture(video_path)
    frames_bgr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()
    
    num_frames = min(len(frames_bgr), len(pred_vp3d), len(pred_pf2))
    
    # Pre-convert all frames to RGB once (much faster than converting per frame)
    print("Converting frames to RGB...")
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    del frames_bgr  # Free memory
    
    # Calculate fixed scale based on all predictions (all frames)
    # Align all predictions to root (center on root) - pre-compute once
    print("Pre-computing aligned poses...")
    pred_vp3d_aligned = pred_vp3d - pred_vp3d[:, 0:1, :]  # (num_frames, 17, 3)
    pred_pf2_aligned = pred_pf2 - pred_pf2[:, 0:1, :]  # (num_frames, 17, 3)
    
    # Find maximum absolute value across all frames and both models
    max_vp3d = np.max(np.abs(pred_vp3d_aligned))
    max_pf2 = np.max(np.abs(pred_pf2_aligned))
    fixed_range = max(max_vp3d, max_pf2)
    
    # Add padding
    padding = 0.1
    fixed_scale = fixed_range * (1 + padding)
    
    print(f"Fixed scale determined from all predictions: {fixed_scale:.4f}")
    print(f"  VideoPose3D max range: {max_vp3d:.4f}")
    print(f"  PoseFormerV2 max range: {max_pf2:.4f}")
    
    # Create output directory for frames
    frames_dir = output_dir / "visualizations" / "comparison" / "frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    if num_workers is None:
        num_workers = min(cpu_count(), num_frames, 16)  # Use up to 16 workers by default
    else:
        num_workers = min(num_workers, num_frames, cpu_count())
    print(f"Generating {num_frames} comparison frames using {num_workers} parallel workers...")
    
    # Prepare all frame generation arguments
    frame_args = []
    for i in range(num_frames):
        frame_args.append((
            i,
            frames_rgb[i],
            coco_keypoints[i],
            pred_vp3d_aligned[i],
            pred_pf2_aligned[i],
            fixed_scale,
            parents,
            joints_left,
            joints_right,
            frames_dir,
            num_frames,
            differences[i]
        ))
    
    # Generate frames in parallel
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(generate_single_frame, frame_args),
                total=num_frames,
                desc="Generating frames"
            ))
    else:
        # Fallback to sequential if only 1 worker
        results = [generate_single_frame(args) for args in tqdm(frame_args, desc="Generating frames")]
    
    # Check for failures
    failed_frames = [i for i, success in results if not success]
    if failed_frames:
        print(f"⚠ Warning: {len(failed_frames)} frames failed to generate: {failed_frames}")
    else:
        print(f"✓ Successfully generated all {num_frames} frames")
    
    # Create video from frames using GPU-accelerated FFmpeg
    print("Creating comparison video with GPU acceleration...")
    output_video = output_dir / "visualizations" / "comparison" / "comparison_video.mp4"
    
    # Get first frame to determine size
    first_frame = cv2.imread(str(frames_dir / "frame_0000.png"))
    height, width = first_frame.shape[:2]
    
    # Determine which ffmpeg to use (prefer conda environment's GPU-enabled ffmpeg)
    ffmpeg_binary = 'ffmpeg'
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env:
        # Try to use conda environment's ffmpeg (which may have GPU support)
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            conda_ffmpeg = Path(conda_prefix) / 'bin' / 'ffmpeg'
            if conda_ffmpeg.exists():
                ffmpeg_binary = str(conda_ffmpeg)
                print(f"Using ffmpeg from conda environment: {ffmpeg_binary}")
    
    # Try GPU-accelerated encoding first (NVENC for NVIDIA GPUs)
    # Check if FFmpeg is available and supports NVENC
    use_gpu = False
    gpu_method = None
    try:
        # Check for hardware acceleration methods
        hwaccel_result = subprocess.run([ffmpeg_binary, '-hide_banner', '-hwaccels'], 
                                       capture_output=True, text=True, timeout=5)
        has_cuda = 'cuda' in hwaccel_result.stdout.lower()
        
        # Check for NVENC encoders
        encoder_result = subprocess.run([ffmpeg_binary, '-hide_banner', '-encoders'], 
                                       capture_output=True, text=True, timeout=5)
        has_nvenc = 'h264_nvenc' in encoder_result.stdout or 'nvenc' in encoder_result.stdout.lower()
        
        if has_cuda and has_nvenc:
            use_gpu = True
            gpu_method = 'NVENC'
            print(f"✓ GPU acceleration detected: CUDA + NVENC available")
            print("  Using GPU-accelerated encoding (NVENC)")
        elif has_cuda:
            print("  CUDA available but NVENC encoders not found, using CPU encoding")
        else:
            print("  No GPU acceleration detected, using CPU encoding")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        print(f"  Could not detect GPU support: {e}")
    
    if use_gpu:
        # Use FFmpeg with NVENC GPU encoding
        input_pattern = str(frames_dir / "frame_%04d.png")
        ffmpeg_cmd = [
            ffmpeg_binary,
            '-y',  # Overwrite output file
            '-framerate', str(video_info['fps']),
            '-i', input_pattern,
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',  # Fast preset for NVENC (p1=fastest, p7=slowest/highest quality)
            '-rc', 'vbr',  # Variable bitrate for better quality
            '-b:v', '10M',  # Target bitrate
            '-maxrate', '12M',  # Maximum bitrate
            '-bufsize', '20M',  # Buffer size
            '-pix_fmt', 'yuv420p',
            '-vf', f'scale={width}:{height}',
            str(output_video)
        ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"✓ Comparison video saved (GPU-accelerated {gpu_method}): {output_video}")
        except subprocess.CalledProcessError as e:
            print(f"⚠ GPU encoding failed: {e.stderr if e.stderr else 'Unknown error'}")
            print("  Falling back to CPU encoding...")
            use_gpu = False
    
    if not use_gpu:
        # Fall back to CPU encoding with FFmpeg (still faster than OpenCV)
        input_pattern = str(frames_dir / "frame_%04d.png")
        ffmpeg_cmd = [
            ffmpeg_binary,
            '-y',  # Overwrite output file
            '-framerate', str(video_info['fps']),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',  # Quality setting (lower = better quality, 18-28 is typical range)
            '-pix_fmt', 'yuv420p',
            '-vf', f'scale={width}:{height}',
            str(output_video)
        ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"✓ Comparison video saved (CPU encoding): {output_video}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Final fallback to OpenCV if FFmpeg is not available
            error_msg = e.stderr if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
            print(f"⚠ FFmpeg not available, using OpenCV (slower): {error_msg}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, video_info['fps'], (width, height))
            
            for i in range(num_frames):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                if frame_path.exists():
                    frame = cv2.imread(str(frame_path))
                    out.write(frame)
            
            out.release()
            print(f"✓ Comparison video saved (OpenCV fallback): {output_video}")
    
    return output_video

def main():
    parser = argparse.ArgumentParser(description='Compare VideoPose3D and PoseFormerV2 on a video')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--videopose3d-checkpoint', type=str,
                       default='VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin',
                       help='Path to VideoPose3D checkpoint (for COCO input, use detectron_coco model)')
    parser.add_argument('--poseformerv2-checkpoint', type=str,
                       default='PoseFormerV2/checkpoint/243_27_27_45.2.bin',
                       help='Path to PoseFormerV2 checkpoint')
    parser.add_argument('--output-dir', type=str, default='results/video_comparison',
                       help='Output directory for results')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers for frame generation (default: min(CPU count, 8))')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Setup paths
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        video_path = repo_root / args.video
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {args.video}")
    
    video_name = video_path.stem
    output_dir = repo_root / args.output_dir / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoints - resolve relative to repo_root
    vp3d_checkpoint = Path(args.videopose3d_checkpoint)
    if not vp3d_checkpoint.is_absolute():
        vp3d_checkpoint = repo_root / args.videopose3d_checkpoint
    
    pf2_checkpoint = Path(args.poseformerv2_checkpoint)
    if not pf2_checkpoint.is_absolute():
        pf2_checkpoint = repo_root / args.poseformerv2_checkpoint
    
    if not vp3d_checkpoint.exists():
        # Try alternative checkpoint names
        alt_checkpoints = [
            repo_root / "VideoPose3D/checkpoint/pretrained_h36m_cpn.bin",
            repo_root / "VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin"
        ]
        found = False
        for alt in alt_checkpoints:
            if alt.exists():
                vp3d_checkpoint = alt
                print(f"Using alternative checkpoint: {vp3d_checkpoint}")
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"VideoPose3D checkpoint not found: {vp3d_checkpoint}\n"
                f"Please download pretrained_h36m_detectron_coco.bin from:\n"
                f"https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
            )
    
    if not pf2_checkpoint.exists():
        # Try alternative checkpoint names
        alt_checkpoints = [
            repo_root / "PoseFormerV2/checkpoint/243_27_27_45.2.bin",
            repo_root / "PoseFormerV2/checkpoint/27_243_45.2.bin"
        ]
        found = False
        for alt in alt_checkpoints:
            if alt.exists():
                pf2_checkpoint = alt
                print(f"Using alternative checkpoint: {pf2_checkpoint}")
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"PoseFormerV2 checkpoint not found: {pf2_checkpoint}\n"
                f"Please download 243_27_27_45.2.bin from Google Drive:\n"
                f"https://drive.google.com/file/d/14SpqPyq9yiblCzTH5CorymKCUsXapmkg/view?usp=share_link"
            )
    
    print("="*60)
    print("Video Model Comparison Pipeline")
    print("="*60)
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"VideoPose3D checkpoint: {vp3d_checkpoint}")
    print(f"PoseFormerV2 checkpoint: {pf2_checkpoint}")
    print("="*60)
    
    try:
        # Step 1: Extract 2D keypoints
        coco_kpts, h36m_kpts, video_info = extract_2d_keypoints_hrnet(
            str(video_path), output_dir)
        
        # Step 2: Prepare VideoPose3D data
        vp3d_data_file, vp3d_video_name = prepare_videopose3d_data(
            coco_kpts, video_info, output_dir)
        
        # Step 3: Run VideoPose3D
        pred_vp3d = run_videopose3d_inference(
            vp3d_data_file, vp3d_video_name, str(vp3d_checkpoint),
            output_dir, video_info)
        
        # Step 4: Run PoseFormerV2
        pred_pf2 = run_poseformerv2_inference(
            h36m_kpts, str(pf2_checkpoint), output_dir,
            str(video_path), video_info)
        
        # Step 5: Compute differences
        metrics, differences = compute_per_joint_differences(
            pred_vp3d, pred_pf2, output_dir)
        
        # Step 6: Create visualization
        comparison_video = create_comparison_visualization(
            str(video_path), pred_vp3d, pred_pf2, coco_kpts,
            differences, output_dir, video_info, num_workers=args.num_workers)
        
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        print(f"Comparison video: {comparison_video}")
        print(f"Metrics: {output_dir / '3d_predictions' / 'comparison_metrics.json'}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

