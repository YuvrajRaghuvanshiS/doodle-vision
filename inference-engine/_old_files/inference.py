import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Constants (match your training setup)
MAX_SEQ_LEN = 130
STROKE_FEATURES = 3
IMG_SIZE = 28
TARGET_STROKE_POINTS = 130  # Optimized based on training data analysis
DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Global variables
drawing = False
last_point = None
current_strokes = []
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
model = load_model("../models/best_hybrid_model_strokes_scaled.keras")


def douglas_peucker_simplify(points, epsilon=2.0):
    """
    Simplifies a stroke using the Douglas-Peucker algorithm.

    Args:
        points (list of [x, y, ...]): List of stroke points.
        epsilon (float): Distance threshold for simplification.

    Returns:
        List of simplified points.
    """
    if len(points) <= 2:
        return points

    def perpendicular_distance(pt, line_start, line_end):
        """
        Computes perpendicular distance from a point to a line.

        Args:
            pt (list): The point to measure from.
            line_start (list): Start of the line segment.
            line_end (list): End of the line segment.

        Returns:
            float: Perpendicular distance.
        """
        p = np.array(pt[:2])
        a = np.array(line_start[:2])
        b = np.array(line_end[:2])

        # If line is degenerate (start == end), return Euclidean distance
        if np.allclose(a, b):
            return np.linalg.norm(p - a)

        # Vector projection formula for perpendicular distance
        ab = b - a
        ap = p - a
        return np.abs(np.cross(ab, ap)) / np.linalg.norm(ab)

    def simplify_segment(segment):
        """
        Recursively simplifies a segment of the stroke.

        Args:
            segment (list of points): The current segment to simplify.

        Returns:
            list: Simplified points for this segment.
        """
        if len(segment) <= 2:
            return segment

        start, end = segment[0], segment[-1]
        max_dist = 0
        index = -1

        # Find the point with the maximum perpendicular distance
        for i in range(1, len(segment) - 1):
            dist = perpendicular_distance(segment[i], start, end)
            if dist > max_dist:
                max_dist = dist
                index = i

        # If distance exceeds epsilon, recursively simplify
        if max_dist > epsilon:
            left = simplify_segment(segment[: index + 1])
            right = simplify_segment(segment[index:])
            return left[:-1] + right  # Avoid duplicating the middle point
        else:
            return [start, end]

    return simplify_segment(points)


def optimize_stroke_capture(strokes_list, target_points=TARGET_STROKE_POINTS):
    """
    Simplifies and resamples strokes to match the format of training data.

    Steps:
    1. Applies Douglas-Peucker simplification to each stroke.
    2. Resamples to retain important points (start, end, pen state changes).
    3. Ensures the final stroke sequence fits the target point budget.

    Args:
        strokes_list (list of list of [x, y, pen_state]): List of strokes.
        target_points (int): Desired number of total stroke points.

    Returns:
        np.ndarray: A float32 array of shape (<=target_points, STROKE_FEATURES)
    """
    all_points = []

    # Step 1: Simplify each stroke using Douglas-Peucker
    for stroke in strokes_list:
        if len(stroke) < 2:
            continue
        simplified = douglas_peucker_simplify(stroke, epsilon=2.0)
        all_points.extend(simplified)

    if len(all_points) == 0:
        # Return zeroed-out array if no valid points found
        return np.zeros((MAX_SEQ_LEN, STROKE_FEATURES), dtype=np.float32)

    all_points = np.array(all_points, dtype=np.float32)

    # Step 2: Downsample intelligently if too many points
    if len(all_points) > target_points:
        important_indices = [0]  # Always keep the first point

        # Keep points where pen state changes (stroke breaks)
        for i in range(1, len(all_points)):
            if all_points[i, 2] != all_points[i - 1, 2]:
                important_indices.append(i)

        # Always keep the last point
        important_indices.append(len(all_points) - 1)

        # Deduplicate and sort
        important_indices = sorted(set(important_indices))

        if len(important_indices) > target_points:
            # Too many important points: uniform subsampling
            step = len(important_indices) // target_points
            important_indices = important_indices[::step][:target_points]
        elif len(important_indices) < target_points:
            # Not enough: fill in from other points uniformly
            remaining = target_points - len(important_indices)
            all_indices = set(range(len(all_points)))
            available = sorted(all_indices - set(important_indices))

            if available and remaining > 0:
                step = max(1, len(available) // remaining)
                additional = available[::step][:remaining]
                important_indices.extend(additional)
                important_indices = sorted(important_indices)

        all_points = all_points[important_indices]

    return all_points


def preprocess_stroke_improved(stroke_data, max_len=MAX_SEQ_LEN):
    """
    Preprocesses a stroke sequence by centering and scaling coordinates
    to the [-100, 100] range, and padding/truncating to max_len.

    Args:
        stroke_data (list or ndarray): List of [x, y, pen_state] points.
        max_len (int): Desired sequence length for output.

    Returns:
        np.ndarray: Preprocessed stroke of shape (max_len, STROKE_FEATURES)
    """
    if len(stroke_data) == 0:
        # Return padded zeros if input is empty
        return np.zeros((max_len, STROKE_FEATURES), dtype=np.float32)

    # Convert input to float array and copy to avoid mutation
    stroke = np.array(stroke_data, dtype=np.float32).copy()

    # Step 1: Center the coordinates at origin
    stroke[:, 0] -= np.mean(stroke[:, 0])
    stroke[:, 1] -= np.mean(stroke[:, 1])

    # Step 2: Scale uniformly to fit within [-100, 100] range
    max_x = np.max(np.abs(stroke[:, 0]))
    max_y = np.max(np.abs(stroke[:, 1]))
    max_extent = max(max_x, max_y)

    if max_extent > 0:
        scale = 100.0 / max_extent
        stroke[:, 0] *= scale
        stroke[:, 1] *= scale

    # Step 3: Pad with zeros or truncate to fixed length
    if len(stroke) > max_len:
        stroke = stroke[:max_len]
    else:
        pad_len = max_len - len(stroke)
        pad = np.zeros((pad_len, STROKE_FEATURES), dtype=np.float32)
        stroke = np.vstack([stroke, pad])

    return stroke


def process_image_corrected(canvas):
    """
    Processes a drawing canvas to produce a model-compatible grayscale image.

    Steps:
    1. Convert to grayscale.
    2. Extract the drawing region using contours.
    3. Crop the image to the bounding box with padding.
    4. Pad to square shape (if needed).
    5. Resize to model input size and normalize.

    Args:
        canvas (np.ndarray): Input BGR image of the drawing (usually from OpenCV canvas).

    Returns:
        np.ndarray: Preprocessed image of shape (IMG_SIZE, IMG_SIZE, 1), float32 in [0, 1].
    """
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Step 2: Extract contours of non-background pixels
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return blank image
    if not contours:
        return np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    # Step 3: Compute bounding box around all contours
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Step 4: Add padding (clamped to canvas size)
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(canvas.shape[1] - x, w + 2 * padding)
    h = min(canvas.shape[0] - y, h + 2 * padding)

    # Step 5: Crop the region of interest
    cropped = gray[y : y + h, x : x + w]

    # Step 6: Pad to make it square (centered content)
    if w > h:
        pad = (w - h) // 2
        cropped = np.pad(
            cropped, ((pad, pad), (0, 0)), mode="constant", constant_values=0
        )
    elif h > w:
        pad = (h - w) // 2
        cropped = np.pad(
            cropped, ((0, 0), (pad, pad)), mode="constant", constant_values=0
        )

    # Step 7: Resize to target model input size (28 x 28) and normalize to [0, 1]
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    # Step 8: Add channel dimension for grayscale
    return np.expand_dims(normalized, axis=-1)


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing"""
    global drawing, last_point, current_strokes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
        current_strokes.append([])
        current_strokes[-1].append([x, y, 0.0])  # Pen down (0 = down)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if last_point is None:
            return

        # Adaptive point spacing to prevent overflow
        distance = np.hypot(x - last_point[0], y - last_point[1])
        min_distance = 4  # Increased to reduce point density

        if distance >= min_distance:
            current_strokes[-1].append([x, y, 0.0])  # Pen stays down
            cv2.line(canvas, last_point, (x, y), (255, 255, 255), 2)
            last_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if current_strokes and current_strokes[-1]:
            current_strokes[-1][-1][2] = 1.0  # Set last point to pen up (1 = up)


# Class names for prediction output
class_names = [
    "backpack",
    "banana",
    "bat",
    "beard",
    "bicycle",
    "bird",
    "book",
    "bread",
    "bridge",
    "bucket",
    "bush",
    "butterfly",
    "cactus",
    "camel",
    "camera",
    "candle",
    "cow",
    "crab",
    "crown",
    "cup",
    "donut",
    "dumbbell",
    "elbow",
    "eye",
    "fish",
    "flashlight",
    "flip flops",
    "flower",
    "foot",
    "hat",
    "helicopter",
    "hot air balloon",
    "leaf",
    "leg",
    "light bulb",
    "lightning",
    "motorbike",
    "mouth",
    "nail",
    "pencil",
    "pillow",
    "river",
    "school bus",
    "sock",
    "spoon",
    "table",
    "telephone",
    "tooth",
    "tree",
    "umbrella",
]


# Setup OpenCV window
cv2.namedWindow("Improved Drawing App", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Improved Drawing App", mouse_callback)

print("Improved Sketch Recognition App Started!")
print("Features:")
print("- Improved stroke preprocessing with [-100, 100] scaling")
print("- Optimized stroke point capture (targeting 30-70 points)")
print("- Douglas-Peucker simplification with epsilon=2.0")
print("- Coordinate centering and consistent normalization")
print("")
print("Controls:")
print("- Left click and drag to draw")
print("- Press 'p' to predict")
print("- Press 'c' to clear canvas")
print("- Press 'ESC' to exit")

while True:
    # === Display Updated Canvas ===
    display_canvas = canvas.copy()
    total_points = sum(len(stroke) for stroke in current_strokes)

    # Show status info and instructions
    cv2.putText(
        display_canvas,
        f"Strokes: {len(current_strokes)}, Points: {total_points}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 100, 100),
        1,
    )
    cv2.putText(
        display_canvas,
        "Press 'p' to predict, 'c' to clear, ESC to exit",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 100, 100),
        1,
    )
    cv2.imshow("Improved Drawing App", display_canvas)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("p"):  # === Predict ===
        if not current_strokes or all(len(stroke) <= 1 for stroke in current_strokes):
            print("No drawing detected! Please draw something first.")
            continue

        print("\n=== Processing Drawing ===")
        print(f"Input: {len(current_strokes)} strokes, {total_points} total points")

        # Step 1: Stroke optimization
        optimized_points = optimize_stroke_capture(
            current_strokes, TARGET_STROKE_POINTS
        )
        print(f"After optimization: {len(optimized_points)} points")

        # Step 2: Normalize and scale strokes
        stroke_input = preprocess_stroke_improved(optimized_points)

        # Step 3: Extract image input from canvas
        img_input = process_image_corrected(canvas)

        # Step 4: Create visual debug summary
        timestamp = datetime.now().strftime("%H%M%S")
        debug_path = f"{DEBUG_DIR}/improved_debug_{timestamp}.png"

        plt.figure(figsize=(20, 5))
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]

        # Subplot 1: Raw strokes
        plt.subplot(1, 5, 1)
        for i, stroke in enumerate(current_strokes):
            if len(stroke) > 0:
                stroke_arr = np.array(stroke)
                plt.plot(
                    stroke_arr[:, 0],
                    stroke_arr[:, 1],
                    color=colors[i % len(colors)],
                    marker="o",
                    markersize=2,
                    linewidth=2,
                    alpha=0.7,
                )
        plt.title(f"Raw Input ({len(current_strokes)} strokes, {total_points} points)")
        plt.gca().set_aspect("equal")

        # Subplot 2: Optimized points
        plt.subplot(1, 5, 2)
        if len(optimized_points) > 0:
            plt.plot(
                optimized_points[:, 0],
                optimized_points[:, 1],
                "g-o",
                linewidth=2,
                alpha=0.7,
            )
            plt.scatter(
                optimized_points[:, 0],
                optimized_points[:, 1],
                c=optimized_points[:, 2],
                cmap="RdYlBu",
                s=30,
                edgecolors="black",
                linewidth=0.5,
            )
        plt.title(f"Optimized ({len(optimized_points)} points)")
        plt.gca().set_aspect("equal")

        # Subplot 3: Scaled strokes
        plt.subplot(1, 5, 3)
        valid_mask = np.any(stroke_input != 0, axis=1)
        if np.any(valid_mask):
            plt.plot(
                stroke_input[valid_mask, 0],
                stroke_input[valid_mask, 1],
                "b-",
                linewidth=2,
                alpha=0.7,
            )
            plt.scatter(
                stroke_input[valid_mask, 0],
                stroke_input[valid_mask, 1],
                c=stroke_input[valid_mask, 2],
                cmap="RdYlBu",
                s=20,
            )
        plt.title("Scaled [-100,100]")
        plt.xlim(-110, 110)
        plt.ylim(-110, 110)
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect("equal")

        # Subplot 4: Canvas image
        plt.subplot(1, 5, 4)
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cmap="gray")
        plt.title("Canvas Image")
        plt.axis("off")

        # Subplot 5: Final model input image
        plt.subplot(1, 5, 5)
        plt.imshow(img_input.squeeze(), cmap="gray")
        plt.title("Model Input Image")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(debug_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Debug visualization saved: {debug_path}")

        # Step 5: Run model prediction
        stroke_input_batch = np.expand_dims(stroke_input, axis=0)
        img_input_batch = np.expand_dims(img_input, axis=0)

        print("Running model inference...")
        predictions = model.predict([stroke_input_batch, img_input_batch], verbose=0)

        # Step 6: Show top-5 predictions
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        top5_probs = predictions[0][top5_indices]

        print("\n=== Top 5 Predictions ===")
        for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
            class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            print(f"{i + 1}. {class_name}: {prob * 100:.1f}%")

        # Step 7: Confidence rating
        best_class = (
            class_names[top5_indices[0]]
            if top5_indices[0] < len(class_names)
            else f"Class_{top5_indices[0]}"
        )
        best_conf = top5_probs[0] * 100

        if best_conf > 50:
            status = "🎯 High confidence"
        elif best_conf > 30:
            status = "✅ Good confidence"
        elif best_conf > 15:
            status = "⚠️  Moderate confidence"
        else:
            status = "❌ Low confidence"

        print(f"\n{status}: {best_class} ({best_conf:.1f}%)")

        # Step 8: Coordinate coverage info
        if np.any(valid_mask):
            range_x = np.ptp(stroke_input[valid_mask, 0])
            range_y = np.ptp(stroke_input[valid_mask, 1])
            print(f"Coordinate ranges - X: {range_x:.1f}, Y: {range_y:.1f}")
            if best_conf < 15:
                print("💡 Suggestion: Try drawing more clearly or add more detail")

    elif key == ord("c"):  # === Clear canvas ===
        canvas.fill(0)
        current_strokes = []
        print("Canvas cleared!")

    elif key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
print("Improved inference app closed.")
