"""
Phase 6: Flow Visualization
Visualizes crowd flow dynamics on video frames

Features:
- Density heatmap overlay
- Velocity vector field arrows
- Choke point markers
- Flow metrics display
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class FlowVisualizer:
    """
    Visualizes crowd flow intelligence data
    """

    def __init__(
        self,
        frame_size: Tuple[int, int] = (1920, 1080),
        show_density: bool = True,
        show_vectors: bool = True,
        show_choke_points: bool = True,
        show_metrics: bool = True,
        heatmap_alpha: float = 0.5
    ):
        """
        Initialize Flow Visualizer

        Args:
            frame_size: (width, height) of video frames
            show_density: Show density heatmap
            show_vectors: Show velocity vector field
            show_choke_points: Show choke point markers
            show_metrics: Show flow metrics overlay
            heatmap_alpha: Transparency of heatmap (0=invisible, 1=opaque)
        """
        self.frame_width, self.frame_height = frame_size
        self.show_density = show_density
        self.show_vectors = show_vectors
        self.show_choke_points = show_choke_points
        self.show_metrics = show_metrics
        self.heatmap_alpha = heatmap_alpha

        # Color maps
        self.colormap = cv2.COLORMAP_JET  # Blue (low) to Red (high)

        print(f"✓ Flow Visualizer initialized")
        print(f"   Density: {show_density} | Vectors: {show_vectors}")
        print(f"   Choke Points: {show_choke_points} | Metrics: {show_metrics}")


    def visualize(
        self,
        frame: np.ndarray,
        flow_data: Dict
    ) -> np.ndarray:
        """
        Overlay flow visualization on frame

        Args:
            frame: BGR image (H, W, 3)
            flow_data: Flow intelligence output from phase5

        Returns:
            vis_frame: Frame with flow visualization overlays
        """
        vis_frame = frame.copy()

        # 1. Density heatmap overlay
        if self.show_density and 'density_map' in flow_data:
            vis_frame = self._draw_density_heatmap(
                vis_frame,
                flow_data['density_map']
            )

        # 2. Velocity vector field
        if self.show_vectors and 'velocity_field' in flow_data:
            vis_frame = self._draw_velocity_vectors(
                vis_frame,
                flow_data['velocity_field'],
                flow_data.get('grid_shape', (22, 38))
            )

        # 3. Choke point markers
        if self.show_choke_points and 'choke_points' in flow_data:
            vis_frame = self._draw_choke_points(
                vis_frame,
                flow_data['choke_points']
            )

        # 4. Flow metrics overlay
        if self.show_metrics and 'flow_metrics' in flow_data:
            vis_frame = self._draw_metrics_overlay(
                vis_frame,
                flow_data['flow_metrics'],
                flow_data.get('choke_points', [])
            )

        return vis_frame


    def _draw_density_heatmap(
        self,
        frame: np.ndarray,
        density_map: np.ndarray
    ) -> np.ndarray:
        """
        Draw density heatmap overlay

        Args:
            frame: BGR frame
            density_map: (H, W) density values

        Returns:
            frame: Frame with heatmap overlay
        """
        # Normalize density map to 0-255
        if density_map.max() > 0:
            density_normalized = (density_map / density_map.max() * 255).astype(np.uint8)
        else:
            density_normalized = np.zeros_like(density_map, dtype=np.uint8)

        # Resize to match frame if needed
        if density_normalized.shape != (self.frame_height, self.frame_width):
            density_normalized = cv2.resize(
                density_normalized,
                (self.frame_width, self.frame_height)
            )

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(density_normalized, self.colormap)

        # Blend with original frame
        frame = cv2.addWeighted(
            frame,
            1 - self.heatmap_alpha,
            heatmap_colored,
            self.heatmap_alpha,
            0
        )

        return frame


    def _draw_velocity_vectors(
        self,
        frame: np.ndarray,
        velocity_field: np.ndarray,
        grid_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Draw velocity vector field as arrows

        Args:
            frame: BGR frame
            velocity_field: (grid_rows, grid_cols, 2) velocity vectors
            grid_shape: (rows, cols) grid dimensions

        Returns:
            frame: Frame with velocity arrows
        """
        grid_rows, grid_cols = grid_shape
        grid_size = self.frame_width // grid_cols

        # Draw arrow for each grid cell
        for i in range(grid_rows):
            for j in range(grid_cols):
                vx, vy = velocity_field[i, j]

                # Skip if no movement
                speed = np.sqrt(vx**2 + vy**2)
                if speed < 0.5:  # Threshold for drawing
                    continue

                # Cell center
                cx = j * grid_size + grid_size // 2
                cy = i * grid_size + grid_size // 2

                # Arrow end point (scale velocity for visibility)
                scale = 2.0
                ex = int(cx + vx * scale)
                ey = int(cy + vy * scale)

                # Color based on speed (green = slow, yellow = fast)
                color = self._speed_to_color(speed)

                # Draw arrow
                cv2.arrowedLine(
                    frame,
                    (cx, cy),
                    (ex, ey),
                    color,
                    thickness=2,
                    tipLength=0.3
                )

        return frame


    def _draw_choke_points(
        self,
        frame: np.ndarray,
        choke_points: List[Dict]
    ) -> np.ndarray:
        """
        Draw choke point markers

        Args:
            frame: BGR frame
            choke_points: List of detected choke points

        Returns:
            frame: Frame with choke point markers
        """
        for cp in choke_points:
            px, py = cp['pixel_pos']
            severity = cp['severity']
            choke_index = cp['choke_index']

            # Color based on severity
            if severity == 'high':
                color = (0, 0, 255)  # Red
                radius = 30
            else:
                color = (0, 165, 255)  # Orange
                radius = 20

            # Draw warning circle
            cv2.circle(frame, (px, py), radius, color, 3)
            cv2.circle(frame, (px, py), radius + 5, color, 1)

            # Draw warning icon
            cv2.putText(
                frame,
                "!",
                (px - 8, py + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # Label
            label = f"CHOKE {choke_index:.2f}"
            cv2.putText(
                frame,
                label,
                (px - 40, py - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return frame


    def _draw_metrics_overlay(
        self,
        frame: np.ndarray,
        flow_metrics: Dict,
        choke_points: List[Dict]
    ) -> np.ndarray:
        """
        Draw flow metrics text overlay

        Args:
            frame: BGR frame
            flow_metrics: Flow analysis metrics
            choke_points: Detected choke points

        Returns:
            frame: Frame with metrics overlay
        """
        # Create semi-transparent panel
        overlay = frame.copy()
        panel_height = 180
        cv2.rectangle(
            overlay,
            (10, 10),
            (400, panel_height),
            (0, 0, 0),
            -1
        )
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        y_offset = 30

        # Draw metrics
        metrics_text = [
            f"FLOW INTELLIGENCE",
            f"People: {flow_metrics.get('total_people', 0)}",
            f"Avg Density: {flow_metrics.get('avg_density', 0):.2f}",
            f"Avg Speed: {flow_metrics.get('avg_speed', 0):.1f} px/frame",
            f"Direction Entropy: {flow_metrics.get('direction_entropy', 0):.2f}",
            f"Turbulence: {flow_metrics.get('turbulence', 0):.2f}",
            f"Flow Opposition: {flow_metrics.get('bidirectional_pressure', 0):.2f}",
            f"Choke Points: {len(choke_points)}"
        ]

        for i, text in enumerate(metrics_text):
            y_pos = y_offset + i * 20
            if i == 0:  # Title
                cv2.putText(frame, text, (20, y_pos), font, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, text, (20, y_pos), font, font_scale, color, thickness)

        # Risk indicator
        risk_level = self._compute_risk_level(flow_metrics, choke_points)
        risk_color = self._risk_to_color(risk_level)

        cv2.rectangle(frame, (10, panel_height + 10), (400, panel_height + 50), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, panel_height + 10), (400, panel_height + 50), risk_color, 2)

        risk_text = f"RISK LEVEL: {risk_level.upper()}"
        cv2.putText(
            frame,
            risk_text,
            (20, panel_height + 35),
            font,
            0.6,
            risk_color,
            2
        )

        return frame


    def _speed_to_color(self, speed: float) -> Tuple[int, int, int]:
        """Convert speed to BGR color (green -> yellow -> red)"""
        # Normalize speed (0-20 px/frame typical range)
        speed_norm = min(speed / 20.0, 1.0)

        if speed_norm < 0.5:
            # Green to Yellow
            ratio = speed_norm * 2
            return (0, int(255 * (1 - ratio)), int(255 * ratio))
        else:
            # Yellow to Red
            ratio = (speed_norm - 0.5) * 2
            return (0, int(255 * (1 - ratio)), 255)


    def _compute_risk_level(
        self,
        flow_metrics: Dict,
        choke_points: List[Dict]
    ) -> str:
        """
        Compute overall crowd risk level

        Returns:
            'low', 'medium', 'high', or 'critical'
        """
        density = flow_metrics.get('avg_density', 0)
        turbulence = flow_metrics.get('turbulence', 0)
        opposition = flow_metrics.get('bidirectional_pressure', 0)
        num_chokes = len(choke_points)

        # Risk score (0-1)
        risk_score = (
            min(density / 10.0, 1.0) * 0.3 +
            min(turbulence, 1.0) * 0.2 +
            min(opposition, 1.0) * 0.3 +
            min(num_chokes / 5.0, 1.0) * 0.2
        )

        if risk_score < 0.25:
            return 'low'
        elif risk_score < 0.5:
            return 'medium'
        elif risk_score < 0.75:
            return 'high'
        else:
            return 'critical'


    def _risk_to_color(self, risk_level: str) -> Tuple[int, int, int]:
        """Convert risk level to BGR color"""
        colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 255, 255),  # Yellow
            'high': (0, 165, 255),    # Orange
            'critical': (0, 0, 255)   # Red
        }
        return colors.get(risk_level, (255, 255, 255))
