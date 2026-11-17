"""
Phase 4 Enhanced: DeepSORT with Simple Appearance Features
Adds color histogram-based re-identification to reduce ID switches
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_color_histogram(image_crop: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Compute HSV color histogram for person crop

    Args:
        image_crop: BGR image crop of person
        bins: Number of bins per channel

    Returns:
        Normalized histogram feature vector
    """
    if image_crop is None or image_crop.size == 0:
        return np.zeros(bins * 3)

    # Convert to HSV (more robust to lighting changes)
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)

    # Compute histogram for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # Concatenate and normalize
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = hist / (np.sum(hist) + 1e-10)

    return hist


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compare two histograms using correlation

    Returns:
        Similarity score (0-1, higher = more similar)
    """
    correlation = cv2.compareHist(
        hist1.astype(np.float32),
        hist2.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    # Convert from [-1, 1] to [0, 1]
    return (correlation + 1) / 2


def extract_appearance_feature(frame: np.ndarray, bbox: List[float]) -> np.ndarray:
    """
    Extract appearance feature from bounding box

    Args:
        frame: Full frame image (BGR)
        bbox: [x1, y1, x2, y2]

    Returns:
        Feature vector
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Ensure valid crop
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return compute_color_histogram(None)

    crop = frame[y1:y2, x1:x2]
    return compute_color_histogram(crop)


class TrackWithAppearance:
    """
    Enhanced Track with appearance features

    Add this to existing Track class or use as replacement
    """

    def __init__(self, track_id: int, bbox: List[float], confidence: float, frame: np.ndarray = None):
        # ... existing Track initialization ...

        # Appearance feature
        self.appearance_feature = None
        if frame is not None:
            self.appearance_feature = extract_appearance_feature(frame, bbox)

        # History of features (for temporal smoothing)
        self.appearance_history = []
        if self.appearance_feature is not None:
            self.appearance_history.append(self.appearance_feature)

    def update_with_appearance(self, bbox: List[float], confidence: float, frame: np.ndarray):
        """Update track with new detection and appearance"""
        # ... existing update logic ...

        # Update appearance
        new_feature = extract_appearance_feature(frame, bbox)
        self.appearance_history.append(new_feature)

        # Keep last 10 features
        if len(self.appearance_history) > 10:
            self.appearance_history.pop(0)

        # Average feature (temporal smoothing)
        self.appearance_feature = np.mean(self.appearance_history, axis=0)

    def appearance_similarity(self, detection_feature: np.ndarray, threshold: float = 0.5) -> float:
        """
        Compute appearance similarity with detection

        Returns:
            Similarity score (0-1)
        """
        if self.appearance_feature is None or detection_feature is None:
            return 0.0

        return compare_histograms(self.appearance_feature, detection_feature)


def enhanced_matching_with_appearance(
    detections: List[Dict],
    tracks: List,
    frame: np.ndarray,
    iou_weight: float = 0.6,
    appearance_weight: float = 0.4
) -> List[Tuple[int, int]]:
    """
    Match detections to tracks using both IoU and appearance

    Args:
        detections: Detection list
        tracks: Track list
        frame: Current frame for appearance extraction
        iou_weight: Weight for IoU similarity
        appearance_weight: Weight for appearance similarity

    Returns:
        List of (detection_idx, track_idx) matches
    """
    if len(tracks) == 0 or len(detections) == 0:
        return []

    # Compute cost matrix (combined IoU + appearance)
    cost_matrix = np.zeros((len(detections), len(tracks)))

    for d_idx, det in enumerate(detections):
        det_feature = extract_appearance_feature(frame, det['bbox'])

        for t_idx, track in enumerate(tracks):
            # IoU similarity
            iou = compute_iou(det['bbox'], track.bbox)

            # Appearance similarity
            app_sim = track.appearance_similarity(det_feature) if hasattr(track, 'appearance_similarity') else 0.0

            # Combined score
            combined_score = iou_weight * iou + appearance_weight * app_sim
            cost_matrix[d_idx, t_idx] = combined_score

    # Hungarian algorithm or greedy matching
    matches = []
    threshold = 0.3  # Minimum combined score

    # Greedy matching (can be upgraded to Hungarian)
    used_dets = set()
    used_tracks = set()

    # Sort by score (highest first)
    scores = []
    for d_idx in range(len(detections)):
        for t_idx in range(len(tracks)):
            scores.append((cost_matrix[d_idx, t_idx], d_idx, t_idx))

    scores.sort(reverse=True)

    for score, d_idx, t_idx in scores:
        if score < threshold:
            break
        if d_idx not in used_dets and t_idx not in used_tracks:
            matches.append((d_idx, t_idx))
            used_dets.add(d_idx)
            used_tracks.add(t_idx)

    return matches


# Usage in DeepSORTTracker.update():
# matches = enhanced_matching_with_appearance(detections, self.tracks, frame)
