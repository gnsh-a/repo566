#!/usr/bin/env python3
"""
Comparison script for VideoPose3D and PoseFormerV2 metrics.
Extracts metrics from evaluation outputs and generates a comparison table.
"""

import re
import json
import csv
import sys
from pathlib import Path

def extract_videopose3d_metrics(output_file):
    """Extract metrics from VideoPose3D evaluation output."""
    metrics = {
        'MPJPE': None,
        'P-MPJPE': None,
        'N-MPJPE': None
    }
    
    if not Path(output_file).exists():
        print(f"Warning: {output_file} not found")
        return metrics
    
    with open(output_file, 'r') as f:
        content = f.read()
    
    # VideoPose3D outputs metrics like:
    # Protocol #1   (MPJPE) action-wise average: XX.XX mm
    # Protocol #2 (P-MPJPE) action-wise average: XX.XX mm
    # Protocol #3 (N-MPJPE) action-wise average: XX.XX mm
    # Or: Protocol #1 Error (MPJPE): XX.XX mm
    mpjpe_match = re.search(r'Protocol\s+#1.*?\(MPJPE\).*?(?:average|Error):\s*([\d.]+)\s*mm', content)
    p_mpjpe_match = re.search(r'Protocol\s+#2.*?\(P-MPJPE\).*?(?:average|Error):\s*([\d.]+)\s*mm', content)
    n_mpjpe_match = re.search(r'Protocol\s+#3.*?\(N-MPJPE\).*?(?:average|Error):\s*([\d.]+)\s*mm', content)
    
    if mpjpe_match:
        metrics['MPJPE'] = float(mpjpe_match.group(1))
    if p_mpjpe_match:
        metrics['P-MPJPE'] = float(p_mpjpe_match.group(1))
    if n_mpjpe_match:
        metrics['N-MPJPE'] = float(n_mpjpe_match.group(1))
    
    return metrics

def extract_poseformerv2_metrics(output_file):
    """Extract metrics from PoseFormerV2 evaluation output."""
    metrics = {
        'MPJPE': None,
        'P-MPJPE': None,
        'N-MPJPE': None
    }
    
    if not Path(output_file).exists():
        print(f"Warning: {output_file} not found")
        return metrics
    
    with open(output_file, 'r') as f:
        content = f.read()
    
    # PoseFormerV2 outputs metrics in similar format:
    # Protocol #1   (MPJPE) action-wise average: XX.XX mm
    # Protocol #2 (P-MPJPE) action-wise average: XX.XX mm
    # Protocol #3 (N-MPJPE) action-wise average: XX.XX mm
    # Or: Protocol #1 Error (MPJPE): XX.XX mm
    mpjpe_match = re.search(r'Protocol\s+#1.*?\(MPJPE\).*?(?:average|Error):\s*([\d.]+)\s*mm', content)
    p_mpjpe_match = re.search(r'Protocol\s+#2.*?\(P-MPJPE\).*?(?:average|Error):\s*([\d.]+)\s*mm', content)
    n_mpjpe_match = re.search(r'Protocol\s+#3.*?\(N-MPJPE\).*?(?:average|Error):\s*([\d.]+)\s*mm', content)
    
    if mpjpe_match:
        metrics['MPJPE'] = float(mpjpe_match.group(1))
    if p_mpjpe_match:
        metrics['P-MPJPE'] = float(p_mpjpe_match.group(1))
    if n_mpjpe_match:
        metrics['N-MPJPE'] = float(n_mpjpe_match.group(1))
    
    return metrics

def format_metric(value):
    """Format metric value for display."""
    if value is None:
        return "N/A"
    return f"{value:.2f}"

def main():
    # Get the script directory and navigate to results
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Extract metrics from both models
    videopose3d_file = results_dir / "videopose3d_eval_output.txt"
    poseformerv2_file = results_dir / "poseformerv2_eval_output.txt"
    
    videopose3d_metrics = extract_videopose3d_metrics(videopose3d_file)
    poseformerv2_metrics = extract_poseformerv2_metrics(poseformerv2_file)
    
    # Save metrics to JSON
    metrics_dict = {
        'VideoPose3D': videopose3d_metrics,
        'PoseFormerV2': poseformerv2_metrics
    }
    
    json_file = results_dir / "metrics.json"
    with open(json_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {json_file}")
    
    # Create comparison table
    comparison_data = [
        {
            'Model': 'VideoPose3D',
            'Input Type': '2D → 3D lifting',
            'Seq Len': '243',
            'MPJPE': format_metric(videopose3d_metrics['MPJPE']),
            'P-MPJPE': format_metric(videopose3d_metrics['P-MPJPE']),
            'N-MPJPE': format_metric(videopose3d_metrics['N-MPJPE'])
        },
        {
            'Model': 'PoseFormerV2',
            'Input Type': 'ViT + temporal fusion',
            'Seq Len': '243',
            'MPJPE': format_metric(poseformerv2_metrics['MPJPE']),
            'P-MPJPE': format_metric(poseformerv2_metrics['P-MPJPE']),
            'N-MPJPE': format_metric(poseformerv2_metrics['N-MPJPE'])
        }
    ]
    
    # Save to CSV
    csv_file = results_dir / "comparison_table.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'Input Type', 'Seq Len', 'MPJPE', 'P-MPJPE', 'N-MPJPE'])
        writer.writeheader()
        writer.writerows(comparison_data)
    
    print(f"\nComparison table saved to: {csv_file}\n")
    
    # Print formatted table
    print("=" * 80)
    print("METRICS COMPARISON (Human3.6M Test Set: S9, S11)")
    print("=" * 80)
    print(f"{'Model':<15} {'Input Type':<25} {'Seq Len':<10} {'MPJPE':<10} {'P-MPJPE':<10} {'N-MPJPE':<10}")
    print("-" * 80)
    for row in comparison_data:
        print(f"{row['Model']:<15} {row['Input Type']:<25} {row['Seq Len']:<10} {row['MPJPE']:<10} {row['P-MPJPE']:<10} {row['N-MPJPE']:<10}")
    print("=" * 80)
    print("\nNote: All metrics are in millimeters (mm)")
    print("MPJPE: Mean Per-Joint Position Error")
    print("P-MPJPE: Procrustes-aligned MPJPE")
    print("N-MPJPE: Normalized MPJPE")

if __name__ == "__main__":
    main()

