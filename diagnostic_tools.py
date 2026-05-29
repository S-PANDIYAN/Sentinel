"""
PRAVAHA Diagnostic Tools
Systematic verification of detection and tracking issues
"""

import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

class DetectionDiagnostics:
    """Test #1: Missing People (Undercount)"""
    
    def __init__(self):
        self.detection_history = []
        self.frame_count = 0
        
    def test_detection_only(self, video_path, model_path="data/models/yolov8m.pt", conf=0.3, imgsz=1280):
        """
        Run DETECTION ONLY (no tracking) to verify model capability
        
        Returns:
            dict: {
                'avg_detections': float,
                'min_detections': int,
                'max_detections': int,
                'frames_analyzed': int
            }
        """
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        detections_per_frame = []
        
        print("\n" + "="*70)
        print("🔍 DIAGNOSTIC TEST #1: Detection Only (No Tracking)")
        print("="*70)
        print(f"Model: {model_path}")
        print(f"Confidence: {conf}")
        print(f"Image Size: {imgsz}px")
        print("="*70 + "\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            
            # Run detection only (track=False)
            results = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                classes=[0],  # person class only
                verbose=False,
                stream=False
            )
            
            # Count detections
            num_detections = len(results[0].boxes)
            detections_per_frame.append(num_detections)
            
            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count}: {num_detections} people detected")
        
        cap.release()
        
        # Analysis
        avg = np.mean(detections_per_frame)
        min_det = np.min(detections_per_frame)
        max_det = np.max(detections_per_frame)
        
        print("\n" + "="*70)
        print("📊 DETECTION RESULTS")
        print("="*70)
        print(f"Average detections per frame: {avg:.1f}")
        print(f"Minimum detections: {min_det}")
        print(f"Maximum detections: {max_det}")
        print(f"Frames analyzed: {self.frame_count}")
        print("="*70)
        
        # Diagnosis
        print("\n💡 DIAGNOSIS:")
        if avg < 15:
            print("⚠️  LOW DETECTION COUNT")
            print("   → This is a MODEL LIMITATION problem")
            print("   → Solutions:")
            print("     1. Use larger model (yolov8m or yolo11m)")
            print("     2. Increase image_size (1280 or 1920)")
            print("     3. Lower confidence (0.25-0.30)")
        else:
            print("✅ Detection count looks reasonable")
            print("   → If count is still wrong in full pipeline:")
            print("   → Check COUNTING LOGIC in code")
        
        return {
            'avg_detections': avg,
            'min_detections': int(min_det),
            'max_detections': int(max_det),
            'frames_analyzed': self.frame_count
        }


class TrackingDiagnostics:
    """Test #2: ID Switching"""
    
    def __init__(self):
        self.id_history = defaultdict(list)  # track_id -> [(frame, bbox)]
        self.id_switches = []
        self.frame_count = 0
        
    def test_tracking_stability(self, video_path, model_path="data/models/yolov8m.pt", conf=0.3):
        """
        Run TRACKING and monitor ID stability
        
        Returns:
            dict: {
                'total_tracks': int,
                'id_switches': int,
                'avg_track_length': float,
                'unstable_tracks': list
            }
        """
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        track_lifespans = defaultdict(int)
        previous_positions = {}  # track_id -> bbox_center
        
        print("\n" + "="*70)
        print("🔍 DIAGNOSTIC TEST #2: Tracking Stability (ID Switching)")
        print("="*70)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            
            # Run tracking
            results = model.track(
                source=frame,
                conf=conf,
                classes=[0],
                persist=True,
                verbose=False
            )
            
            # Analyze IDs
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xywh.cpu().numpy()
                
                for track_id, box in zip(track_ids, boxes):
                    track_lifespans[track_id] += 1
                    center = (box[0], box[1])
                    
                    # Check for ID switch (same position, different ID)
                    if track_id in previous_positions:
                        prev_center = previous_positions[track_id]
                        distance = np.sqrt((center[0]-prev_center[0])**2 + (center[1]-prev_center[1])**2)
                        
                        if distance > 100:  # Sudden jump
                            self.id_switches.append({
                                'frame': self.frame_count,
                                'track_id': track_id,
                                'distance': distance
                            })
                    
                    previous_positions[track_id] = center
            
            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count}: {len(track_lifespans)} unique tracks so far")
        
        cap.release()
        
        # Analysis
        total_tracks = len(track_lifespans)
        avg_lifespan = np.mean(list(track_lifespans.values()))
        short_tracks = [tid for tid, span in track_lifespans.items() if span < 10]
        
        print("\n" + "="*70)
        print("📊 TRACKING RESULTS")
        print("="*70)
        print(f"Total unique tracks: {total_tracks}")
        print(f"Average track lifespan: {avg_lifespan:.1f} frames")
        print(f"Short tracks (<10 frames): {len(short_tracks)}")
        print(f"Detected ID switches: {len(self.id_switches)}")
        print("="*70)
        
        # Diagnosis
        print("\n💡 DIAGNOSIS:")
        if len(short_tracks) > total_tracks * 0.3:
            print("⚠️  HIGH ID SWITCHING DETECTED")
            print("   → This is a TRACKER CONFIGURATION problem")
            print("   → Solutions:")
            print("     1. Enable DeepSORT with ReID features")
            print("     2. Increase track_buffer (30-60 frames)")
            print("     3. Adjust match_thresh (0.7-0.8)")
            print("     4. Use appearance-based tracker")
        else:
            print("✅ Tracking stability looks good")
            print("   → ID switching rate is acceptable")
        
        return {
            'total_tracks': total_tracks,
            'id_switches': len(self.id_switches),
            'avg_track_length': avg_lifespan,
            'short_tracks': len(short_tracks)
        }


class ChokeDiagnostics:
    """Test #3: Wrong Choke Detection (Secondary Issue)"""
    
    def __init__(self):
        self.choke_events = []
        
    def analyze_choke_accuracy(self, detection_results, tracking_results):
        """
        Analyze if choke detection errors are primary or secondary
        
        Args:
            detection_results: Output from DetectionDiagnostics
            tracking_results: Output from TrackingDiagnostics
        """
        print("\n" + "="*70)
        print("🔍 DIAGNOSTIC TEST #3: Choke Detection Analysis")
        print("="*70)
        
        # Check if upstream issues exist
        detection_incomplete = detection_results['avg_detections'] < 15
        tracking_unstable = tracking_results['short_tracks'] > tracking_results['total_tracks'] * 0.3
        
        print("\n📊 DEPENDENCY CHECK:")
        print(f"Detection incomplete: {'YES ❌' if detection_incomplete else 'NO ✅'}")
        print(f"Tracking unstable: {'YES ❌' if tracking_unstable else 'NO ✅'}")
        
        print("\n💡 DIAGNOSIS:")
        if detection_incomplete or tracking_unstable:
            print("🔴 CHOKE DETECTION ISSUES ARE SECONDARY")
            print("   → Caused by upstream problems:")
            if detection_incomplete:
                print("     ❌ Incomplete detection → Speed/Density wrong")
            if tracking_unstable:
                print("     ❌ Unstable tracking → Direction vectors noisy")
            print("\n   → FIX ORDER:")
            print("     1️⃣  First: Fix detection (model/resolution)")
            print("     2️⃣  Second: Fix tracking (ReID/tracker config)")
            print("     3️⃣  Finally: Verify choke detection")
        else:
            print("✅ UPSTREAM SYSTEMS HEALTHY")
            print("   → If choke detection still wrong:")
            print("   → Check choke detection ALGORITHM:")
            print("     - Speed thresholds")
            print("     - Density calculation")
            print("     - Direction entropy logic")
            print("     - Flow opposition detection")
        
        print("="*70)


def run_full_diagnostics(video_path):
    """
    Run complete diagnostic suite
    
    Usage:
        python diagnostic_tools.py path/to/video.mp4
    """
    print("\n" + "="*70)
    print("🌊 PRAVAHA - Full System Diagnostics")
    print("="*70)
    print(f"Video: {video_path}")
    print("="*70)
    
    # Test 1: Detection
    print("\n\n")
    det = DetectionDiagnostics()
    detection_results = det.test_detection_only(video_path)
    
    # Test 2: Tracking
    print("\n\n")
    track = TrackingDiagnostics()
    tracking_results = track.test_tracking_stability(video_path)
    
    # Test 3: Choke Analysis
    print("\n\n")
    choke = ChokeDiagnostics()
    choke.analyze_choke_accuracy(detection_results, tracking_results)
    
    # Save results
    results = {
        'video': video_path,
        'detection': detection_results,
        'tracking': tracking_results,
        'timestamp': str(Path(video_path).stem)
    }
    
    output_file = Path("data/output") / "diagnostic_report.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n📁 Full report saved: {output_file}")
    print("\n" + "="*70)
    print("✅ DIAGNOSTICS COMPLETE")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnostic_tools.py <video_path>")
        print("\nExample:")
        print("  python diagnostic_tools.py data/input/test.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found: {video_path}")
        sys.exit(1)
    
    run_full_diagnostics(video_path)
