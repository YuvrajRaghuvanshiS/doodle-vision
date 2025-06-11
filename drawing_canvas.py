"""
Manages the drawing canvas and rendering
"""

from typing import List, Tuple

import cv2
import numpy as np


class DrawingCanvas:
    """Manages drawing canvas and stroke rendering"""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.stroke_history: List[List[List[float]]] = []

        # Drawing settings
        self.line_color = (255, 255, 255)  # White
        self.line_thickness = 2
        self.ui_color = (100, 100, 100)  # Gray for UI text

    def clear(self):
        """Clear the canvas"""
        self.canvas.fill(0)
        self.stroke_history = []

    def add_stroke(self, stroke: List[List[float]]):
        """Add completed stroke to history"""
        if stroke:
            self.stroke_history.append(stroke.copy())

    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Draw line segment on canvas"""
        cv2.line(self.canvas, start, end, self.line_color, self.line_thickness)

    def get_canvas_copy(self) -> np.ndarray:
        """Get copy of current canvas"""
        return self.canvas.copy()

    def add_ui_text(
        self,
        canvas: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = 0.6,
    ) -> np.ndarray:
        """Add UI text to canvas"""
        cv2.putText(
            canvas,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self.ui_color,
            1,
        )
        return canvas

    def get_stroke_stats(self) -> dict:
        """Get statistics about current drawing"""
        total_points = sum(len(stroke) for stroke in self.stroke_history)
        return {"num_strokes": len(self.stroke_history), "total_points": total_points}
