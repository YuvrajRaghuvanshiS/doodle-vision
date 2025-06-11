"""
Mouse-based drawing input handler
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from inputs.base_input import BaseInputHandler


class MouseInputHandler(BaseInputHandler):
    """Handles mouse input for drawing"""

    def __init__(self, window_name: str, canvas_size: Tuple[int, int]):
        super().__init__()
        self.window_name = window_name
        self.canvas_width, self.canvas_height = canvas_size
        self.last_point: Optional[Tuple[int, int]] = None
        self.min_point_distance = 4

    def start_capture(self):
        """Start mouse capture"""
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def stop_capture(self):
        """Stop mouse capture"""
        cv2.setMouseCallback(self.window_name, lambda *args: None)

    def is_point_valid(self, x: int, y: int) -> bool:
        """Check if point is within canvas bounds"""
        return 0 <= x < self.canvas_width and 0 <= y < self.canvas_height

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if not self.is_point_valid(x, y):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._start_stroke(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            self._continue_stroke(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._end_stroke()

    def _start_stroke(self, x: int, y: int):
        """Start new stroke"""
        self.is_drawing = True
        self.last_point = (x, y)
        self.current_stroke = [[x, y, 0.0]]  # pen down
        self.trigger_callback("stroke_start", {"x": x, "y": y})

    def _continue_stroke(self, x: int, y: int):
        """Continue current stroke"""
        if self.last_point is None:
            return

        # Only add point if moved minimum distance
        distance = np.hypot(x - self.last_point[0], y - self.last_point[1])
        if distance >= self.min_point_distance:
            self.current_stroke.append([x, y, 0.0])
            self.trigger_callback(
                "stroke_continue", {"from": self.last_point, "to": (x, y)}
            )
            self.last_point = (x, y)

    def _end_stroke(self):
        """End current stroke"""
        if self.current_stroke:
            self.current_stroke[-1][2] = 1.0  # pen up
        self.is_drawing = False
        self.trigger_callback("stroke_end", {"stroke": self.current_stroke.copy()})
        self.current_stroke = []
