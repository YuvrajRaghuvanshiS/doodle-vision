"""
Camera-based finger gesture input handler
"""

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from inputs.base_input import BaseInputHandler


class GestureInputHandler(BaseInputHandler):
    """Handles finger gesture input for drawing"""

    def __init__(self, canvas_size: Tuple[int, int], camera_id: int = 0):
        super().__init__()
        self.canvas_width, self.canvas_height = canvas_size
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None

        # MediaPipe components
        self.mp_hands = None
        self.hands = None
        self.mp_drawing = None

        # Gesture detection state
        self.drawing_finger_tip = None
        self.last_drawing_point = None
        self.min_point_distance = 8  # Larger for gesture input
        self.is_drawing_gesture = False

        # Gesture-specific parameters
        self.drawing_threshold = 0.8  # Confidence threshold for drawing gesture
        self.gesture_smoothing = 3  # Points to average for smoothing
        self.point_buffer = []

        # Camera to canvas mapping
        self.camera_width = 640
        self.camera_height = 480

    def start_capture(self):
        """Start camera and gesture detection"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self._initialize_hand_detector()
        print("Gesture input started. Extend your index finger to draw.")

    def stop_capture(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def is_point_valid(self, x: int, y: int) -> bool:
        """Check if gesture point is within canvas bounds"""
        return 0 <= x < self.canvas_width and 0 <= y < self.canvas_height

    def _initialize_hand_detector(self):
        """Initialize MediaPipe hand detection"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Configure MediaPipe Hands with optimized settings for drawing [2][3]
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Focus on single hand for drawing
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1,  # Balance between accuracy and speed
        )

    def process_frame(self) -> Optional[np.ndarray]:
        """Process camera frame and detect drawing gestures"""
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Flip frame horizontally for natural mirror interaction [3]
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        results = self.hands.process(frame_rgb)

        # Draw hand landmarks on frame for visual feedback
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Process drawing gesture
                self._process_drawing_gesture(hand_landmarks, frame)
        else:
            # No hand detected - end any current stroke
            if self.is_drawing_gesture:
                self._end_drawing_gesture()

        # Add UI instructions
        cv2.putText(
            frame,
            "Extend index finger to draw",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if self.is_drawing_gesture:
            cv2.putText(
                frame,
                "DRAWING",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return frame

    def _process_drawing_gesture(self, hand_landmarks, frame):
        """Process hand landmarks to detect drawing gestures"""
        # Get key landmark positions [16][17]
        landmarks = hand_landmarks.landmark

        # Index finger landmarks
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]

        # Middle finger landmarks for gesture detection
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

        # Ring finger landmarks
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]

        # Pinky finger landmarks
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]

        # Detect drawing gesture: index finger extended, others folded [12][15]
        is_index_extended = index_tip.y < index_pip.y < index_mcp.y
        is_middle_folded = middle_tip.y > middle_pip.y
        is_ring_folded = ring_tip.y > ring_pip.y
        is_pinky_folded = pinky_tip.y > pinky_pip.y

        drawing_gesture_detected = (
            is_index_extended
            and is_middle_folded
            and is_ring_folded
            and is_pinky_folded
        )

        if drawing_gesture_detected:
            # Convert normalized coordinates to canvas coordinates
            canvas_x, canvas_y = self._map_to_canvas(index_tip.x, index_tip.y)

            if self.is_point_valid(canvas_x, canvas_y):
                # Apply smoothing to reduce jitter [6]
                smooth_x, smooth_y = self._smooth_point(canvas_x, canvas_y)

                if not self.is_drawing_gesture:
                    # Start new stroke
                    self._start_drawing_gesture(smooth_x, smooth_y)
                else:
                    # Continue current stroke
                    self._continue_drawing_gesture(smooth_x, smooth_y)

                # Visual feedback on camera feed
                cv2.circle(
                    frame,
                    (
                        int(index_tip.x * self.camera_width),
                        int(index_tip.y * self.camera_height),
                    ),
                    10,
                    (0, 255, 0),
                    -1,
                )
        else:
            # End drawing if gesture is no longer detected
            if self.is_drawing_gesture:
                self._end_drawing_gesture()

    def _map_to_canvas(
        self, normalized_x: float, normalized_y: float
    ) -> Tuple[int, int]:
        """Map normalized camera coordinates to canvas coordinates"""
        # MediaPipe returns normalized coordinates [0, 1] [16]
        canvas_x = int(normalized_x * self.canvas_width)
        canvas_y = int(normalized_y * self.canvas_height)
        return canvas_x, canvas_y

    def _smooth_point(self, x: int, y: int) -> Tuple[int, int]:
        """Apply smoothing to reduce gesture jitter"""
        self.point_buffer.append((x, y))
        if len(self.point_buffer) > self.gesture_smoothing:
            self.point_buffer.pop(0)

        # Average recent points
        avg_x = sum(p[0] for p in self.point_buffer) // len(self.point_buffer)
        avg_y = sum(p[1] for p in self.point_buffer) // len(self.point_buffer)

        return avg_x, avg_y

    def _start_drawing_gesture(self, x: int, y: int):
        """Start a new drawing stroke"""
        self.is_drawing_gesture = True
        self.last_drawing_point = (x, y)
        self.current_stroke = [[x, y, 0.0]]  # Pen down
        self.point_buffer = [(x, y)]  # Reset smoothing buffer

        # Trigger callback for stroke start
        self.trigger_callback("stroke_start", {"x": x, "y": y})

    def _continue_drawing_gesture(self, x: int, y: int):
        """Continue the current drawing stroke"""
        if self.last_drawing_point is None:
            return

        # Check minimum distance to avoid too many points [9]
        distance = np.hypot(
            x - self.last_drawing_point[0], y - self.last_drawing_point[1]
        )

        if distance >= self.min_point_distance:
            self.current_stroke.append([x, y, 0.0])  # Pen stays down

            # Trigger callback for stroke continuation
            self.trigger_callback(
                "stroke_continue", {"from": self.last_drawing_point, "to": (x, y)}
            )

            self.last_drawing_point = (x, y)

    def _end_drawing_gesture(self):
        """End the current drawing stroke"""
        if self.is_drawing_gesture and self.current_stroke:
            # Set last point to pen up
            self.current_stroke[-1][2] = 1.0

            # Trigger callback for stroke end
            self.trigger_callback("stroke_end", {"stroke": self.current_stroke.copy()})

        self.is_drawing_gesture = False
        self.last_drawing_point = None
        self.point_buffer = []

    def get_gesture_info(self) -> dict:
        """Get current gesture detection information"""
        return {
            "is_drawing": self.is_drawing_gesture,
            "camera_size": (self.camera_width, self.camera_height),
            "canvas_size": (self.canvas_width, self.canvas_height),
            "smoothing_points": len(self.point_buffer),
        }
