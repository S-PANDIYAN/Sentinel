"""
PRAVAHA Layer 2 Pipeline - Flow Intelligence
Runs complete perception + flow intelligence pipeline (Phases 1-6)

Usage:
    python run_flow_intelligence.py <input_video> [options]

Example:
    python run_flow_intelligence.py ../data/input/crowd_video.mp4 --model yolov8s.pt
"""

import cv2
import numpy as np
import sys
import os
import argparse
import time
from pathlib import Path

# Fix OpenMP library conflict (common with PyTorch + other libs)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase1_data_acquisition import VideoDataLoader
from phase2_preprocessing import FramePreprocessor
from phase3_yolo_detection import YOLODetector
from phase3_enhanced_detection import EnhancedYOLODetector  # Enhanced detection for dense crowds
from phase4_enhanced_tracking import EnhancedDeepSORTTracker  # Enhanced tracker with ReID
from phase5_flow_intelligence import FlowIntelligence
from phase6_flow_visualization import FlowVisualizer


def run_flow_intelligence_pipeline(
    input_video: str,
    output_video: str = None,
    confidence_threshold: float = 0.35,
    max_tracks: int = 100,
    model_path: str = "yolo11m.pt",
    image_size: int = 1280,
    grid_size: int = 50,
    show_preview: bool = False,
    save_stats: bool = True,
    use_enhanced_detection: bool = True,  # NEW: Use enhanced detection
    use_enhanced_tracking: bool = True
):
    """
    Run complete flow intelligence pipeline

    Args:
        input_video: Path to input video
        output_video: Path to output video
        confidence_threshold: YOLO confidence threshold
        max_tracks: Maximum simultaneous tracks
        model_path: YOLOv8 model to use
        grid_size: Grid cell size for flow field (pixels)
        show_preview: Show live preview
        save_stats: Save statistics file
        use_enhanced_tracking: Use enhanced tracker with appearance ReID (default: True)
    """

    print("="*70)
    print("🌊 PRAVAHA - Layer 2: Flow Intelligence Pipeline")
    print("="*70)
    print(f"\n📹 Input:  {input_video}")

    if output_video:
        print(f"📼 Output: {output_video}")

    # Check input
    if not os.path.exists(input_video):
        print(f"\n❌ Error: Input video not found: {input_video}")
        return

    try:
        # Initialize all phases
        print("\n🔧 Initializing pipeline...")

        loader = VideoDataLoader(input_video)
        
        # Initialize detector (enhanced or basic)
        if use_enhanced_detection:
            detector = EnhancedYOLODetector(
                model_path=model_path,       # yolo11m.pt
                confidence_threshold=confidence_threshold,  # 0.35
                iou_threshold=0.5,
                image_size=image_size,       # 1280x1280
                device="cuda",
                verbose=False
            )
            preprocessor = None  # Enhanced detector handles preprocessing internally
        else:
            preprocessor = FramePreprocessor(target_size=(640, 640), normalize=True)
            detector = YOLODetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device="cuda",
                verbose=False
            )

        # Initialize tracker (enhanced or standard)
        if use_enhanced_tracking:
            tracker = EnhancedDeepSORTTracker(
                max_age=90,              # Keep tracks for 3 seconds
                n_init=3,                # Confirm after 3 detections
                max_iou_distance=0.7,    # Lenient IoU matching
                max_tracks=max_tracks,
                embedder="mobilenet",    # Lightweight appearance model
                embedder_gpu=True,       # Use GPU for embeddings
                half=True,               # FP16 for speed
                bgr=True,                # OpenCV uses BGR
                max_cosine_distance=0.4  # Appearance similarity threshold
            )
        else:
            # Fallback to basic tracker (not recommended for production)
            from phase4_deepsort_tracking import DeepSORTTracker
            tracker = DeepSORTTracker(
                max_iou_distance=0.5,
                max_age=30,
                n_init=3,
                max_tracks=max_tracks
            )

        # Initialize Flow Intelligence (Layer 2)
        flow_analyzer = FlowIntelligence(
            frame_size=(loader.width, loader.height),
            grid_size=grid_size,
            history_length=30,
            choke_threshold=0.7
        )

        flow_visualizer = FlowVisualizer(
            frame_size=(loader.width, loader.height),
            show_density=True,
            show_vectors=True,
            show_choke_points=True,
            show_metrics=True,
            heatmap_alpha=0.4
        )

        print(f"✓ Phase 1: Data Acquisition")
        print(f"   - Frames: {loader.total_frames}")
        print(f"   - Resolution: {loader.width}×{loader.height}")
        print(f"   - FPS: {loader.fps}")
        if use_enhanced_detection:
            print(f"✓ Phase 2: Preprocessing (handled by detector)")
            print(f"✓ Phase 3: Enhanced YOLO Detection (model={model_path}, size={image_size}px, conf={confidence_threshold})")
        else:
            print(f"✓ Phase 2: Preprocessing")
            print(f"✓ Phase 3: Basic YOLO Detection (model={model_path}, size=640px, conf={confidence_threshold})")
        if use_enhanced_tracking:
            print(f"✓ Phase 4: Enhanced DeepSORT Tracking with Appearance ReID")
        else:
            print(f"✓ Phase 4: Basic DeepSORT Tracking (IoU-only)")
        print(f"✓ Phase 5: Flow Intelligence (grid={grid_size}px)")
        print(f"✓ Phase 6: Flow Visualization")

        # Setup output video
        video_writer = None
        if output_video:
            os.makedirs(os.path.dirname(output_video), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_video,
                fourcc,
                loader.fps,
                (loader.width, loader.height)
            )

        # Processing loop
        print(f"\n🔄 Processing video...")
        print(f"   Total frames: {loader.total_frames}")

        frame_count = 0
        start_time = time.time()
        last_log_time = start_time

        while True:
            # Phase 1: Load frame
            success, frame, frame_info = loader.read_frame()
            if not success or frame is None:
                break

            # Phase 2: Preprocess (if using basic detection)
            if preprocessor is not None:
                processed_frame, _ = preprocessor.process(frame)
            else:
                processed_frame = frame  # Enhanced detector handles preprocessing

            # Phase 3: Detect
            detections, _ = detector.detect(processed_frame)

            # Phase 4: Track (pass original frame for appearance features)
            if use_enhanced_tracking:
                tracks = tracker.update(detections, frame=frame)
            else:
                tracks = tracker.update(detections)

            # Phase 5: Flow Intelligence
            flow_data = flow_analyzer.compute_flow_field(tracks)

            # Phase 6: Visualize
            vis_frame = flow_visualizer.visualize(frame, flow_data)

            # Write output
            if video_writer is not None:
                video_writer.write(vis_frame)

            # Live preview
            if show_preview:
                cv2.imshow('PRAVAHA - Flow Intelligence', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  User interrupted")
                    break

            frame_count += 1

            # Progress logging (every 30 frames)
            if frame_count % 30 == 0 or frame_count == loader.total_frames:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / loader.total_frames) * 100
                eta = (loader.total_frames - frame_count) / fps if fps > 0 else 0

                print(f"   Progress: {frame_count}/{loader.total_frames} ({progress:.1f}%) | FPS: {fps:.2f} | ETA: {eta:.1f}s")

        # Cleanup
        loader.release()
        if video_writer is not None:
            video_writer.release()

        if show_preview:
            cv2.destroyAllWindows()

        # Statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print("\n" + "="*70)
        print("📊 Pipeline Statistics")
        print("="*70)

        # Overall performance
        print(f"\n⏱️  Overall Performance:")
        print(f"   Total frames:         {frame_count}")
        print(f"   Total time:           {total_time:.2f} seconds")
        print(f"   Average FPS:          {avg_fps:.2f}")
        print(f"   Time per frame:       {(total_time/frame_count)*1000:.2f} ms")

        # Phase statistics
        prep_stats = preprocessor.get_stats()
        det_stats = detector.get_stats()
        track_stats = tracker.get_stats()
        flow_stats = flow_analyzer.get_statistics()

        print(f"\n🖼️  Phase 2 (Preprocessing):")
        print(f"   Avg time/frame:       {prep_stats.get('avg_time_ms', 0):.2f} ms")

        print(f"\n🎯 Phase 3 (Detection):")
        print(f"   Total detections:     {det_stats['total_detections']}")
        print(f"   Avg persons/frame:    {det_stats['avg_detections_per_frame']:.1f}")
        print(f"   Avg time/frame:       {det_stats.get('avg_time_per_frame', 0):.2f} ms")

        print(f"\n🔍 Phase 4 (Tracking):")
        print(f"   Total tracks created: {track_stats['total_tracks_created']}")
        print(f"   Peak active tracks:   {track_stats['active_tracks_peak']}")
        print(f"   Avg time/frame:       {track_stats.get('avg_time_sec', 0)*1000:.2f} ms")

        print(f"\n🌊 Phase 5 (Flow Intelligence):")
        print(f"   Avg time/frame:       {flow_stats.get('avg_time_per_frame', 0):.2f} ms")

        print("="*70)

        # Save statistics
        if save_stats and output_video:
            stats_file = output_video.replace('.mp4', '_flow_stats.txt')
            with open(stats_file, 'w') as f:
                f.write("PRAVAHA Layer 2 - Flow Intelligence Statistics\n")
                f.write("="*70 + "\n\n")
                f.write(f"Input Video: {input_video}\n")
                f.write(f"Output Video: {output_video}\n\n")
                f.write(f"Total frames: {frame_count}\n")
                f.write(f"Total time: {total_time:.2f} seconds\n")
                f.write(f"Average FPS: {avg_fps:.2f}\n\n")
                f.write(f"Phase 2 avg time: {prep_stats.get('avg_time_ms', 0):.2f} ms\n")
                f.write(f"Phase 3 avg time: {det_stats.get('avg_time_per_frame', 0):.2f} ms\n")
                f.write(f"Phase 4 avg time: {track_stats.get('avg_time_sec', 0)*1000:.2f} ms\n")
                f.write(f"Phase 5 avg time: {flow_stats.get('avg_time_per_frame', 0):.2f} ms\n\n")
                f.write(f"Total detections: {det_stats['total_detections']}\n")
                f.write(f"Total tracks created: {track_stats['total_tracks_created']}\n")
                f.write(f"Peak active tracks: {track_stats['active_tracks_peak']}\n")

            print(f"\n💾 Statistics saved to: {stats_file}")

        if output_video:
            print(f"\n✅ Output video saved to: {output_video}")

        print("\n✅ Flow Intelligence Layer complete!")
        print("   Ready for Layer 3 (Predictive Dynamics Engine)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PRAVAHA Layer 2: Flow Intelligence Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input video file path"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output video file path (default: auto-generated in data/output/)"
    )

    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.35,
        help="YOLO confidence threshold (0.0-1.0)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                 "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="YOLO model (v8 or v11)"
    )

    parser.add_argument(
        "-m", "--max-tracks",
        type=int,
        default=100,
        help="Maximum number of simultaneous tracks"
    )

    parser.add_argument(
        "-g", "--grid-size",
        type=int,
        default=50,
        help="Flow field grid cell size (pixels)"
    )

    parser.add_argument(
        "-p", "--preview",
        action="store_true",
        help="Show live preview window (press 'q' to quit)"
    )

    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't save statistics file"
    )

    parser.add_argument(
        "--enhanced-tracking",
        action="store_true",
        default=True,
        help="Use enhanced tracker with appearance ReID (default: True)"
    )

    parser.add_argument(
        "--basic-tracking",
        action="store_true",
        help="Use basic IoU-only tracker (faster but more ID switches)"
    )

    parser.add_argument(
        "--enhanced-detection",
        action="store_true",
        default=True,
        help="Use enhanced YOLOv11m detection at 1280px (default: True)"
    )

    parser.add_argument(
        "--basic-detection",
        action="store_true",
        help="Use basic YOLOv8s detection at 640px (faster but misses small people)"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=1280,
        help="Detection image size (default: 1280 for enhanced, 640 for basic)"
    )

    args = parser.parse_args()

    # Auto-generate output path
    if args.output is None:
        input_name = Path(args.input).stem
        args.output = f"../data/output/{input_name}_flow.mp4"

    # Determine tracking and detection modes
    use_enhanced_tracking = not args.basic_tracking  # Default is enhanced
    use_enhanced_detection = not args.basic_detection  # Default is enhanced

    # Auto-update model path and confidence for enhanced detection
    model_path = args.model
    confidence = args.confidence
    image_size = args.image_size

    if use_enhanced_detection and args.model == "yolov8s.pt":
        model_path = "yolo11m.pt"  # Use better model
        if args.confidence == 0.25:
            confidence = 0.35  # Optimized for dense crowds

    # Run pipeline
    run_flow_intelligence_pipeline(
        input_video=args.input,
        output_video=args.output,
        confidence_threshold=confidence,
        max_tracks=args.max_tracks,
        model_path=model_path,
        image_size=image_size,
        grid_size=args.grid_size,
        show_preview=args.preview,
        save_stats=not args.no_stats,
        use_enhanced_detection=use_enhanced_detection,
        use_enhanced_tracking=use_enhanced_tracking
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python run_flow_intelligence.py <input_video> [options]")
        print("\nOptions:")
        print("  -o, --output <path>        Output video path")
        print("  -c, --confidence <float>   YOLO confidence threshold (default: 0.35)")
        print("  --model <model>            YOLO model (default: yolo11m.pt)")
        print("  -m, --max-tracks <int>     Maximum simultaneous tracks (default: 100)")
        print("  -g, --grid-size <int>      Flow grid cell size (default: 50px)")
        print("  --image-size <int>         Detection image size (default: 1280)")
        print("  -p, --preview              Show live preview window")
        print("  --no-stats                 Don't save statistics file")
        print("  --basic-detection          Use YOLOv8s@640px (faster, misses small people)")
        print("  --basic-tracking           Use IoU-only tracker (faster, more ID switches)")
        print("                             Default: Enhanced detection + tracking")
        sys.exit(0)

    main()
