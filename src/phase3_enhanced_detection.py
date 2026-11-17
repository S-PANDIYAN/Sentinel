"""
Phase 3 Enhanced: YOLOv11 Detection with Higher Resolution
Improves detection of small, distant, and occluded people

Improvements:
1. YOLOv11m (medium model - better than v8s)
2. Higher resolution (1280px instead of 640px)
3. Optimized confidence thresholds
4. Better for dense crowds
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import time


class EnhancedYOLODetector:
    """
    Enhanced YOLO detector for dense crowd scenarios

    Improvements over basic detector:
    - YOLOv11m model (better accuracy)
    - Higher resolution processing (1280px)
    - Optimized for small person detection
    - Better handling of occlusion
    """

    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        image_size: int = 1280,
        device: str = "cuda",
        verbose: bool = False
    ):
        """
        Initialize Enhanced YOLO detector

        Args:
            model_path: YOLOv11 model ('yolo11n/s/m/l/x.pt')
            confidence_threshold: Detection confidence (0.35 better for dense crowds)
            iou_threshold: NMS IoU threshold
            image_size: Input image size (1280 for small person detection)
            device: 'cuda' or 'cpu'
            verbose: Print model info
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.verbose = verbose

        # Load model
        self.model = None
        self._load_model()

        # Statistics
        self.frames_processed = 0
        self.total_detections = 0
        self.total_time = 0.0

    def _load_model(self):
        """Load YOLOv11 model"""
        try:
            from ultralytics import YOLO

            print(f"🔄 Loading YOLOv11 model: {self.model_path}")
            print(f"   Resolution: {self.image_size}px (higher = better small detection)")

            # Load model
            self.model = YOLO(self.model_path)

            # Set device
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print("⚠️  CUDA not available, falling back to CPU")
                    self.device = "cpu"
            else:
                print("✓ Using CPU")

            print(f"✓ Model loaded successfully")
            print(f"✓ Optimized for dense crowds (conf={self.confidence_threshold}, imgsz={self.image_size})")

        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )

    def detect(
        self,
        frame: np.ndarray,
        return_crops: bool = False
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect people in frame with enhanced settings

        Args:
            frame: Input frame (BGR or RGB, any size)
            return_crops: Return cropped images for each detection

        Returns:
            (detections, annotated_frame)
        """
        start_time = time.time()

        # Convert to uint8 if needed
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame

        # Run inference with enhanced settings
        results = self.model.predict(
            frame_uint8,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,  # High resolution for small people
            classes=[0],  # Person class only
            device=self.device,
            verbose=self.verbose,
            max_det=300,  # Allow more detections for dense crowds
            agnostic_nms=False,
            half=True  # FP16 for speed
        )

        # Parse results
        detections = []

        if len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i].tolist(),
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': 'person'
                    }

                    # Optional: add cropped image
                    if return_crops:
                        x1, y1, x2, y2 = map(int, boxes[i])
                        h, w = frame_uint8.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            detection['crop'] = frame_uint8[y1:y2, x1:x2]

                    detections.append(detection)

        # Update statistics
        self.frames_processed += 1
        self.total_detections += len(detections)
        self.total_time += (time.time() - start_time)

        # Annotated frame
        annotated_frame = results[0].plot() if len(results) > 0 else frame_uint8

        return detections, annotated_frame

    def get_stats(self) -> dict:
        """Get detection statistics"""
        avg_time = (self.total_time / self.frames_processed * 1000) if self.frames_processed > 0 else 0
        avg_dets = self.total_detections / self.frames_processed if self.frames_processed > 0 else 0

        return {
            'frames_processed': self.frames_processed,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': avg_dets,
            'total_time_sec': self.total_time,
            'avg_time_per_frame': avg_time,
            'fps': 1000.0 / avg_time if avg_time > 0 else 0
        }

    def reset_stats(self):
        """Reset statistics"""
        self.frames_processed = 0
        self.total_detections = 0
        self.total_time = 0.0


# Convenience function for quick detection
def detect_people_enhanced(
    video_path: str,
    model: str = "yolo11m.pt",
    conf: float = 0.35,
    imgsz: int = 1280
):
    """
    Quick detection with enhanced settings

    Example:
        detect_people_enhanced("crowd.mp4", conf=0.35, imgsz=1280)
    """
    detector = EnhancedYOLODetector(
        model_path=model,
        confidence_threshold=conf,
        image_size=imgsz
    )

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, vis = detector.detect(frame)
        print(f"Frame: {detector.frames_processed}, Detections: {len(detections)}")

        cv2.imshow("Enhanced Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("Detection Statistics:")
    print(detector.get_stats())
