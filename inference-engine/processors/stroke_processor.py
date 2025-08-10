"""
Handles stroke optimization and preprocessing
"""

from typing import List

import numpy as np


class StrokeProcessor:
    """Handles stroke data processing and optimization"""

    def __init__(self, max_seq_len: int = 130, target_points: int = 130):
        self.max_seq_len = max_seq_len
        self.target_points = target_points
        self.stroke_features = 3
        self.douglas_peucker_epsilon = 2.0
        self.coordinate_scale = 100.0

    def douglas_peucker_simplify(
        self, points: List[List[float]], epsilon: float = None
    ) -> List[List[float]]:
        """Simplify stroke using Douglas-Peucker algorithm"""
        if epsilon is None:
            epsilon = self.douglas_peucker_epsilon

        if len(points) <= 2:
            return points

        def perpendicular_distance(pt, line_start, line_end):
            p = np.array(pt[:2])
            a = np.array(line_start[:2])
            b = np.array(line_end[:2])

            if np.allclose(a, b):
                return np.linalg.norm(p - a)

            ab = b - a
            ap = p - a
            return np.abs(np.cross(ab, ap)) / np.linalg.norm(ab)

        def simplify_segment(segment):
            if len(segment) <= 2:
                return segment

            start, end = segment[0], segment[-1]
            max_dist = 0
            index = -1

            for i in range(1, len(segment) - 1):
                dist = perpendicular_distance(segment[i], start, end)
                if dist > max_dist:
                    max_dist = dist
                    index = i

            if max_dist > epsilon:
                left = simplify_segment(segment[: index + 1])
                right = simplify_segment(segment[index:])
                return left[:-1] + right
            else:
                return [start, end]

        return simplify_segment(points)

    def optimize_strokes(self, strokes_list: List[List[List[float]]]) -> np.ndarray:
        """Optimize stroke sequences for model input"""
        all_points = []

        # Simplify each stroke
        for stroke in strokes_list:
            if len(stroke) >= 2:
                simplified = self.douglas_peucker_simplify(stroke)
                all_points.extend(simplified)

        if not all_points:
            return np.zeros((self.max_seq_len, self.stroke_features), dtype=np.float32)

        all_points = np.array(all_points, dtype=np.float32)

        # Intelligent downsampling if too many points
        if len(all_points) > self.target_points:
            all_points = self._select_important_points(all_points)

        return all_points

    def _select_important_points(self, points: np.ndarray) -> np.ndarray:
        """Select most important points for stroke representation"""
        important_indices = [0]  # First point

        # Keep pen state transitions
        for i in range(1, len(points)):
            if points[i, 2] != points[i - 1, 2]:
                important_indices.append(i)

        # Last point
        important_indices.append(len(points) - 1)
        important_indices = sorted(set(important_indices))

        # Fill remaining slots if needed
        if len(important_indices) < self.target_points:
            remaining = self.target_points - len(important_indices)
            all_indices = set(range(len(points)))
            available = sorted(list(all_indices - set(important_indices)))

            if available and remaining > 0:
                step = max(1, len(available) // remaining)
                additional = available[::step][:remaining]
                important_indices.extend(additional)
                important_indices = sorted(important_indices)

        return points[important_indices[: self.target_points]]

    def preprocess_for_model(self, stroke_data: np.ndarray) -> np.ndarray:
        """Preprocess strokes for model input"""
        if len(stroke_data) == 0:
            return np.zeros((self.max_seq_len, self.stroke_features), dtype=np.float32)

        strokes = stroke_data.copy().astype(np.float32)

        # Center coordinates
        strokes[:, 0] -= np.mean(strokes[:, 0])
        strokes[:, 1] -= np.mean(strokes[:, 1])

        # Scale to [-100, 100] range
        max_x = np.max(np.abs(strokes[:, 0]))
        max_y = np.max(np.abs(strokes[:, 1]))
        max_extent = max(max_x, max_y)

        if max_extent > 0:
            scale = self.coordinate_scale / max_extent
            strokes[:, :2] *= scale

        # Pad or truncate to fixed length
        if len(strokes) > self.max_seq_len:
            strokes = strokes[: self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(strokes)
            padding = np.zeros((pad_len, self.stroke_features), dtype=np.float32)
            strokes = np.vstack([strokes, padding])

        return strokes
