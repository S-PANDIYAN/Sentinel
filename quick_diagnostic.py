"""
Quick Diagnostic Tool - Fast sampling-based analysis
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
from pathlib import Path
import json

def quick_detection_test(video_path, model_path="data/models/yolov8m.pt", conf=0.3, imgsz=1280, sample_frames=30):
    """
    Quick detection test - sample every Nth frame instead of all frames
    
    Args:
        sample_frames: Number of frames to sample (default: 30)
    """
    from ultralytics import YOLO
    
    print("\n" + "="*70)
    print("DIAGNOSTIC TEST: Detection Analysis (Quick)")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Confidence: {conf}")
    print(f"Image Size: {imgsz}px")
    print(f"Sampling: Every Nth frame (target: {sample_frames} frames)")
    print("="*70 + "\n")
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    
    # Calculate sampling interval
    sample_interval = max(1, total_frames // sample_frames)
    print(f"Sampling interval: Every {sample_interval} frames\n")
    
    detections_per_frame = []
    sampled_frames = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % sample_interval == 0:
            results = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                classes=[0],
                verbose=False
            )
            
            num_detections = len(results[0].boxes)
            detections_per_frame.append(num_detections)
            sampled_frames.append(frame_idx)
            
            print(f"Frame {frame_idx}/{total_frames}: {num_detections} people detected")
        
        frame_idx += 1
    
    cap.release()
    
    # Analysis
    avg = np.mean(detections_per_frame)
    min_det = np.min(detections_per_frame)
    max_det = np.max(detections_per_frame)
    std_det = np.std(detections_per_frame)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Frames sampled: {len(detections_per_frame)}")
    print(f"Average detections: {avg:.1f}")
    print(f"Min detections: {min_det}")
    print(f"Max detections: {max_det}")
    print(f"Std deviation: {std_det:.2f}")
    print("="*70)
    
    # Diagnosis
    print("\nDIAGNOSIS:")
    if avg < 15:
        print("WARNING: LOW DETECTION COUNT")
        print("-> This is a MODEL LIMITATION problem")
        print("-> Solutions:")
        print("  1. Use larger model (yolov8m or yolo11m)")
        print("  2. Increase image_size (1280 or 1920)")
        print("  3. Lower confidence (0.25-0.30)")
    elif avg < 25:
        print("MODERATE: Detection count acceptable but could be better")
        print("-> Consider:")
        print("  1. Lower confidence threshold slightly")
        print("  2. Increase image_size if GPU allows")
    else:
        print("GOOD: Detection count looks reasonable")
        print("-> If count still wrong in full pipeline:")
        print("-> Check COUNTING LOGIC in code")
    
    print("="*70 + "\n")
    
    # Save results
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'video': str(video_path),
        'model': model_path,
        'confidence': conf,
        'image_size': imgsz,
        'frames_sampled': len(detections_per_frame),
        'total_frames': total_frames,
        'avg_detections': float(avg),
        'min_detections': int(min_det),
        'max_detections': int(max_det),
        'std_detections': float(std_det),
        'detections': [int(d) for d in detections_per_frame],
        'sampled_frame_indices': sampled_frames
    }
    
    output_file = output_dir / "quick_diagnostic.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved: {output_file}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_diagnostic.py <video_path> [conf] [imgsz] [sample_frames]")
        print("\nExample:")
        print("  python quick_diagnostic.py data/input/test.mp4")
        print("  python quick_diagnostic.py data/input/test.mp4 0.25 1920 50")
        sys.exit(1)
    
    video_path = sys.argv[1]
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
    sample_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    
    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    
    quick_detection_test(video_path, conf=conf, imgsz=imgsz, sample_frames=sample_frames)
