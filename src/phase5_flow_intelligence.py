"""
Phase 5: Flow Intelligence Layer
Converts tracking data into crowd flow dynamics (fluid model)

Input: Tracking data from Phase 4 (track_id, bbox, velocity)
Output:
- Spatial density grid
- Velocity vector field F(x,y,t)
- Flow metrics (entropy, turbulence, pressure)
- Choke point detection
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import time


class FlowIntelligence:
    """
    Crowd Flow Intelligence Layer

    Transforms discrete tracking data into continuous flow field.
    Treats crowd as fluid with velocity, density, and pressure.
    """

    def __init__(
        self,
        frame_size: Tuple[int, int] = (1920, 1080),
        grid_size: int = 50,
        history_length: int = 30,
        choke_threshold: float = 0.7
    ):
        """
        Initialize Flow Intelligence

        Args:
            frame_size: (width, height) of video frame
            grid_size: Size of grid cells in pixels (e.g., 50x50)
            history_length: Number of frames to keep for temporal analysis
            choke_threshold: Threshold for choke point detection (0-1)
        """
        self.frame_width, self.frame_height = frame_size
        self.grid_size = grid_size
        self.history_length = history_length
        self.choke_threshold = choke_threshold

        # Calculate grid dimensions
        self.grid_cols = int(np.ceil(self.frame_width / grid_size))
        self.grid_rows = int(np.ceil(self.frame_height / grid_size))

        # History buffers for temporal analysis
        self.density_history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length)

        # Statistics
        self.frames_processed = 0
        self.total_time = 0.0

        print(f"✓ Flow Intelligence initialized")
        print(f"   Grid: {self.grid_cols}×{self.grid_rows} cells ({grid_size}px)")
        print(f"   History: {history_length} frames")


    def compute_flow_field(
        self,
        tracks: List[Dict]
    ) -> Dict:
        """
        Compute crowd flow field from tracking data

        Args:
            tracks: List of active tracks from DeepSORT
                Each track: {
                    'track_id': int,
                    'bbox': [x1, y1, x2, y2],
                    'velocity': [vx, vy],  # Optional
                }

        Returns:
            flow_data: Dictionary containing:
                - density_grid: np.array (grid_rows, grid_cols) - people count per cell
                - velocity_field: np.array (grid_rows, grid_cols, 2) - avg velocity per cell
                - density_map: np.array (H, W) - smooth density heatmap
                - flow_metrics: dict with entropy, turbulence, pressure indices
                - choke_points: List of detected choke point cells
        """
        start_time = time.time()

        # Initialize grids
        density_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        velocity_field = np.zeros((self.grid_rows, self.grid_cols, 2), dtype=np.float32)
        velocity_count = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

        # Process each track
        for track in tracks:
            bbox = track['bbox']

            # Get person center position
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # Map to grid cell
            grid_x = int(cx / self.grid_size)
            grid_y = int(cy / self.grid_size)

            # Boundary check
            if 0 <= grid_x < self.grid_cols and 0 <= grid_y < self.grid_rows:
                # Increment density
                density_grid[grid_y, grid_x] += 1

                # Add velocity if available
                if 'velocity' in track and track['velocity'] is not None:
                    vx, vy = track['velocity']
                    velocity_field[grid_y, grid_x, 0] += vx
                    velocity_field[grid_y, grid_x, 1] += vy
                    velocity_count[grid_y, grid_x] += 1

        # Average velocities
        mask = velocity_count > 0
        if np.any(mask):
            velocity_field[mask, 0] /= velocity_count[mask]
            velocity_field[mask, 1] /= velocity_count[mask]

        # Create smooth density heatmap
        density_map = self._create_density_heatmap(density_grid)

        # Compute flow metrics
        flow_metrics = self._compute_flow_metrics(
            density_grid,
            velocity_field,
            velocity_count
        )

        # Detect choke points
        choke_points = self._detect_choke_points(
            density_grid,
            velocity_field,
            flow_metrics
        )

        # Store in history
        self.density_history.append(density_grid.copy())
        self.velocity_history.append(velocity_field.copy())

        # Update statistics
        self.frames_processed += 1
        self.total_time += (time.time() - start_time)

        return {
            'density_grid': density_grid,
            'velocity_field': velocity_field,
            'density_map': density_map,
            'flow_metrics': flow_metrics,
            'choke_points': choke_points,
            'grid_shape': (self.grid_rows, self.grid_cols)
        }


    def _create_density_heatmap(self, density_grid: np.ndarray) -> np.ndarray:
        """
        Create smooth density heatmap from grid

        Args:
            density_grid: (grid_rows, grid_cols) density counts

        Returns:
            density_map: (frame_height, frame_width) smooth heatmap
        """
        # Upscale grid to frame size
        density_upscaled = cv2.resize(
            density_grid,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Apply Gaussian smoothing
        kernel_size = max(self.grid_size, 51)
        if kernel_size % 2 == 0:
            kernel_size += 1

        density_map = cv2.GaussianBlur(
            density_upscaled,
            (kernel_size, kernel_size),
            sigmaX=self.grid_size/2
        )

        return density_map


    def _compute_flow_metrics(
        self,
        density_grid: np.ndarray,
        velocity_field: np.ndarray,
        velocity_count: np.ndarray
    ) -> Dict:
        """
        Compute flow dynamics metrics

        Args:
            density_grid: People count per cell
            velocity_field: Velocity vectors per cell
            velocity_count: Number of tracks contributing to each cell

        Returns:
            metrics: Dictionary with flow analysis metrics
        """
        metrics = {}

        # Total crowd size
        metrics['total_people'] = int(np.sum(density_grid))

        # Average density
        metrics['avg_density'] = float(np.mean(density_grid))
        metrics['max_density'] = float(np.max(density_grid))

        # Velocity statistics (only for cells with people)
        active_cells = velocity_count > 0
        if np.any(active_cells):
            velocities = velocity_field[active_cells]
            speeds = np.linalg.norm(velocities, axis=1)

            metrics['avg_speed'] = float(np.mean(speeds))
            metrics['max_speed'] = float(np.max(speeds))
            metrics['min_speed'] = float(np.min(speeds))

            # Direction entropy (how chaotic is movement)
            # Higher entropy = more random/chaotic movement
            angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
            hist = hist / np.sum(hist) + 1e-10  # Normalize
            entropy = -np.sum(hist * np.log2(hist))
            metrics['direction_entropy'] = float(entropy)

            # Turbulence coefficient (velocity variation)
            velocity_std = np.std(speeds)
            velocity_mean = np.mean(speeds) + 1e-10
            metrics['turbulence'] = float(velocity_std / velocity_mean)
        else:
            metrics['avg_speed'] = 0.0
            metrics['max_speed'] = 0.0
            metrics['min_speed'] = 0.0
            metrics['direction_entropy'] = 0.0
            metrics['turbulence'] = 0.0

        # Bidirectional pressure (opposing flows)
        metrics['bidirectional_pressure'] = self._compute_flow_opposition(
            velocity_field,
            velocity_count
        )

        return metrics


    def _compute_flow_opposition(
        self,
        velocity_field: np.ndarray,
        velocity_count: np.ndarray
    ) -> float:
        """
        Compute bidirectional pressure index
        Measures opposing flows (dangerous in crowds)

        Returns:
            pressure: 0-1, higher = more opposing flows
        """
        # Look at neighboring cells
        opposition_sum = 0
        pair_count = 0

        for i in range(self.grid_rows - 1):
            for j in range(self.grid_cols - 1):
                # Only consider cells with people
                if velocity_count[i, j] > 0:
                    v1 = velocity_field[i, j]

                    # Check right neighbor
                    if velocity_count[i, j+1] > 0:
                        v2 = velocity_field[i, j+1]
                        # Dot product < 0 means opposing directions
                        dot = np.dot(v1, v2)
                        if dot < 0:
                            opposition_sum += abs(dot)
                        pair_count += 1

                    # Check bottom neighbor
                    if velocity_count[i+1, j] > 0:
                        v2 = velocity_field[i+1, j]
                        dot = np.dot(v1, v2)
                        if dot < 0:
                            opposition_sum += abs(dot)
                        pair_count += 1

        if pair_count == 0:
            return 0.0

        return float(opposition_sum / pair_count)


    def _detect_choke_points(
        self,
        density_grid: np.ndarray,
        velocity_field: np.ndarray,
        flow_metrics: Dict
    ) -> List[Dict]:
        """
        Detect choke points using composite index

        ChokeIndex = Density × FlowOpposition × SpeedDrop

        Args:
            density_grid: People density per cell
            velocity_field: Velocity vectors
            flow_metrics: Flow analysis metrics

        Returns:
            choke_points: List of detected choke point cells
                Each: {'grid_pos': (row, col), 'choke_index': float, 'severity': str}
        """
        choke_points = []

        # Compute speed magnitude per cell
        speeds = np.linalg.norm(velocity_field, axis=2)

        # Normalize metrics to 0-1
        density_norm = density_grid / (flow_metrics['max_density'] + 1e-10)
        speeds_norm = speeds / (flow_metrics['max_speed'] + 1e-10)

        # Speed drop (1 - normalized speed)
        speed_drop = 1.0 - speeds_norm

        # Local flow opposition (simplified)
        # For each cell, check if it has high density and low speed
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                if density_grid[i, j] > 0:
                    # Composite choke index
                    choke_index = (
                        density_norm[i, j] *
                        speed_drop[i, j] *
                        (1 + flow_metrics['bidirectional_pressure'])
                    )

                    # Threshold check
                    if choke_index > self.choke_threshold:
                        severity = 'high' if choke_index > 0.85 else 'medium'

                        choke_points.append({
                            'grid_pos': (i, j),
                            'pixel_pos': (
                                j * self.grid_size + self.grid_size // 2,
                                i * self.grid_size + self.grid_size // 2
                            ),
                            'choke_index': float(choke_index),
                            'severity': severity,
                            'density': float(density_grid[i, j]),
                            'speed': float(speeds[i, j])
                        })

        return choke_points


    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        if self.frames_processed == 0:
            return {}

        return {
            'frames_processed': self.frames_processed,
            'total_time': self.total_time,
            'avg_time_per_frame': self.total_time / self.frames_processed * 1000,  # ms
            'avg_fps': self.frames_processed / self.total_time
        }
