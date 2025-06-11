"""
Handles canvas image processing for model input
"""

import cv2
import numpy as np


class ImageProcessor:
    """Processes canvas images for model input"""

    def __init__(self, target_size: int = 28):
        self.target_size = target_size
        self.padding = 20

    def process_canvas(self, canvas: np.ndarray) -> np.ndarray:
        """Process canvas image for model input"""
        # Convert to grayscale
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        # Find drawing content
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.zeros((self.target_size, self.target_size, 1), dtype=np.float32)

        # Get bounding box with padding
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        x = max(0, x - self.padding)
        y = max(0, y - self.padding)
        w = min(canvas.shape[1] - x, w + 2 * self.padding)
        h = min(canvas.shape[0] - y, h + 2 * self.padding)

        # Crop and make square
        cropped = gray[y : y + h, x : x + w]
        cropped = self._make_square(cropped)

        # Resize and normalize
        resized = cv2.resize(
            cropped, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA
        )
        normalized = resized.astype(np.float32) / 255.0

        return np.expand_dims(normalized, axis=-1)

    def _make_square(self, image: np.ndarray) -> np.ndarray:
        """Make image square by padding"""
        h, w = image.shape

        if w > h:
            pad = (w - h) // 2
            image = np.pad(image, ((pad, pad), (0, 0)), constant_values=0)
        elif h > w:
            pad = (h - w) // 2
            image = np.pad(image, ((0, 0), (pad, pad)), constant_values=0)

        return image
