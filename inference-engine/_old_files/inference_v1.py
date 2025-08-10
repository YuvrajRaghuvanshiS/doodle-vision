"""
Last working code:
1. Here everything works. Next strokes scaling must be implemented.
"""

import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Constants matching our training setup
MAX_SEQ_LEN = 130
STROKE_FEATURES = 3
IMG_SIZE = 28
TARGET_STROKE_POINTS = 50  # Optimized based on training data analysis
DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Global variables
drawing = False
last_point = None
current_strokes = []
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
model = load_model("../models/best_hybrid_model.keras")


def douglas_peucker_simplify(points, epsilon=2.0):
    """
    Simplify stroke using Douglas-Peucker algorithm with epsilon=2.0
    as used in QuickDraw preprocessing
    """
    if len(points) <= 2:
        return points

    def perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line"""
        if np.allclose(line_start[:2], line_end[:2]):
            return np.linalg.norm(np.array(point[:2]) - np.array(line_start[:2]))

        line_vec = np.array(line_end[:2]) - np.array(line_start[:2])
        point_vec = np.array(point[:2]) - np.array(line_start[:2])
        line_len = np.linalg.norm(line_vec)

        if line_len == 0:
            return np.linalg.norm(point_vec)

        return abs(np.cross(line_vec, point_vec)) / line_len

    def recursive_simplify(points_subset):
        if len(points_subset) <= 2:
            return points_subset

        # Find point with maximum distance from line
        dmax = 0
        index = 0
        start_point = points_subset[0]
        end_point = points_subset[-1]

        for i in range(1, len(points_subset) - 1):
            d = perpendicular_distance(points_subset[i], start_point, end_point)
            if d > dmax:
                index = i
                dmax = d

        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            left_results = recursive_simplify(points_subset[: index + 1])
            right_results = recursive_simplify(points_subset[index:])
            return left_results[:-1] + right_results
        else:
            return [start_point, end_point]

    return recursive_simplify(points)


def optimize_stroke_capture(strokes_list, target_points=TARGET_STROKE_POINTS):
    """
    Efficiently capture and process strokes to match training data patterns
    """
    all_points = []

    # Step 1: Apply Douglas-Peucker simplification to each stroke
    for stroke in strokes_list:
        if len(stroke) < 2:
            continue

        # Simplify stroke using QuickDraw standard epsilon
        simplified = douglas_peucker_simplify(stroke, epsilon=2.0)
        all_points.extend(simplified)

    if len(all_points) == 0:
        return np.zeros((MAX_SEQ_LEN, STROKE_FEATURES), dtype=np.float32)

    all_points = np.array(all_points, dtype=np.float32)

    # Step 2: Intelligent resampling if too many points
    if len(all_points) > target_points:
        # Priority-based selection: keep start, end, and pen state transitions
        important_indices = [0]  # Always keep first point

        # Keep pen state transitions (stroke boundaries)
        for i in range(1, len(all_points)):
            if all_points[i, 2] != all_points[i - 1, 2]:
                important_indices.append(i)

        # Add last point
        if len(all_points) > 1:
            important_indices.append(len(all_points) - 1)

        # Remove duplicates and sort
        important_indices = sorted(list(set(important_indices)))

        # If still too many, subsample evenly
        if len(important_indices) > target_points:
            step = len(important_indices) // target_points
            important_indices = important_indices[::step][:target_points]
        elif len(important_indices) < target_points:
            # Fill remaining slots with evenly distributed points
            remaining = target_points - len(important_indices)
            all_indices = set(range(len(all_points)))
            available = sorted(list(all_indices - set(important_indices)))

            if available and remaining > 0:
                step = max(1, len(available) // remaining)
                additional = available[::step][:remaining]
                important_indices.extend(additional)
                important_indices = sorted(important_indices)

        all_points = all_points[important_indices]

    return all_points


def preprocess_stroke_optimized(stroke_data, max_len=MAX_SEQ_LEN):
    """
    Optimized stroke preprocessing matching training data format exactly
    """
    if len(stroke_data) == 0:
        return np.zeros((max_len, STROKE_FEATURES), dtype=np.float32)

    stroke = np.array(stroke_data, dtype=np.float32).copy()

    # Step 1: Convert canvas coordinates to training coordinate space
    canvas_height, canvas_width = 480, 640

    # Apply coordinate transformation to match training data
    # Center coordinates around canvas center
    stroke[:, 0] = stroke[:, 0] - canvas_width / 2
    stroke[:, 1] = stroke[:, 1] - canvas_height / 2

    # Scale to match training data coordinate ranges (~[-250,250], [-150,170])
    scale_factor = 0.7  # Adjusted based on training data analysis
    stroke[:, 0] *= scale_factor
    stroke[:, 1] *= scale_factor

    # Step 2: Convert to delta coordinates (relative movements)
    if len(stroke) > 1:
        deltas = np.diff(stroke, axis=0)
        deltas = np.vstack([[stroke[0, 0], stroke[0, 1], stroke[0, 2]], deltas])
        stroke = deltas

    # Step 3: Apply identical preprocessing as training
    stroke[:, 0] = np.cumsum(stroke[:, 0])
    stroke[:, 1] = np.cumsum(stroke[:, 1])
    stroke[:, 0] -= stroke[:, 0].mean()
    stroke[:, 1] -= stroke[:, 1].mean()

    # Step 4: Pad or truncate to max_len
    if len(stroke) > max_len:
        return stroke[:max_len]

    pad = np.zeros((max_len - len(stroke), STROKE_FEATURES), dtype=np.float32)
    return np.vstack([stroke, pad])


def process_image_corrected(canvas):
    """
    Process drawing canvas to model-compatible image format
    """
    # Convert to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Find drawing content
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(canvas.shape[1] - x, w + 2 * padding)
    h = min(canvas.shape[0] - y, h + 2 * padding)

    # Crop to content
    cropped = gray[y : y + h, x : x + w]

    # Make square by padding shorter dimension
    if w > h:
        pad_h = (w - h) // 2
        cropped = np.pad(
            cropped, ((pad_h, pad_h), (0, 0)), mode="constant", constant_values=0
        )
    elif h > w:
        pad_w = (h - w) // 2
        cropped = np.pad(
            cropped, ((0, 0), (pad_w, pad_w)), mode="constant", constant_values=0
        )

    # Resize to 28x28 and normalize
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0

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
cv2.namedWindow("Drawing App", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Drawing App", mouse_callback)

print("Drawing App Started!")
print("Features:")
print("- Optimized stroke point capture (targeting 30-70 points)")
print("- Douglas-Peucker simplification with epsilon=2.0")
print("- Coordinate system aligned with training data")
print("- Intelligent point sampling")
print("")
print("Controls:")
print("- Left click and drag to draw")
print("- Press 'p' to predict")
print("- Press 'c' to clear canvas")
print("- Press 'ESC' to exit")

while True:
    # Display canvas with instructions
    display_canvas = canvas.copy()

    # Add status information
    total_points = sum(len(stroke) for stroke in current_strokes)
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

    cv2.imshow("Drawing App", display_canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("p"):  # Predict
        if not current_strokes or all(len(stroke) <= 1 for stroke in current_strokes):
            print("No drawing detected! Please draw something first.")
            continue

        print(f"\n=== Processing Drawing ===")
        print(
            f"Input: {len(current_strokes)} strokes, {sum(len(s) for s in current_strokes)} total points"
        )

        # Step 1: Optimize stroke capture
        optimized_points = optimize_stroke_capture(
            current_strokes, TARGET_STROKE_POINTS
        )
        print(f"After optimization: {len(optimized_points)} points")

        # Step 2: Preprocess strokes
        stroke_input = preprocess_stroke_optimized(optimized_points)

        # Step 3: Process image
        img_input = process_image_corrected(canvas)

        # Step 4: Validation
        valid_stroke_points = np.count_nonzero(np.any(stroke_input != 0, axis=1))
        print(f"Valid stroke points for model: {valid_stroke_points}/{MAX_SEQ_LEN}")

        if valid_stroke_points < 5:
            print("⚠️  Drawing too simple - please add more detail!")
            continue

        # Step 5: Create comprehensive debug visualization
        plt.figure(figsize=(20, 5))

        # Raw strokes
        plt.subplot(1, 5, 1)
        plt.title("Raw Input Strokes")
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]
        for i, stroke in enumerate(current_strokes):
            if len(stroke) > 0:
                stroke_arr = np.array(stroke)
                plt.plot(
                    stroke_arr[:, 0],
                    stroke_arr[:, 1],
                    color=colors[i % len(colors)],
                    alpha=0.7,
                    linewidth=2,
                    marker="o",
                    markersize=2,
                )
        plt.gca().set_aspect("equal")
        plt.title(
            f"Raw Input ({len(current_strokes)} strokes, {sum(len(s) for s in current_strokes)} points)"
        )

        # Optimized points
        plt.subplot(1, 5, 2)
        plt.title("Optimized Points")
        if len(optimized_points) > 0:
            plt.plot(
                optimized_points[:, 0],
                optimized_points[:, 1],
                "g-o",
                alpha=0.7,
                linewidth=2,
                markersize=3,
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
        plt.gca().set_aspect("equal")
        plt.title(f"Optimized ({len(optimized_points)} points)")

        # Processed strokes
        plt.subplot(1, 5, 3)
        plt.title("Model Input Strokes")
        valid_mask = np.any(stroke_input != 0, axis=1)
        if np.any(valid_mask):
            plt.plot(
                stroke_input[valid_mask, 0],
                stroke_input[valid_mask, 1],
                "b-",
                alpha=0.7,
                linewidth=2,
            )
            plt.scatter(
                stroke_input[valid_mask, 0],
                stroke_input[valid_mask, 1],
                c=stroke_input[valid_mask, 2],
                cmap="RdYlBu",
                s=20,
            )
        plt.gca().set_aspect("equal")
        plt.title(f"Model Input ({valid_stroke_points} points)")

        # Canvas image
        plt.subplot(1, 5, 4)
        plt.title("Canvas Image")
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        plt.imshow(canvas_gray, cmap="gray")
        plt.axis("off")

        # Processed image
        plt.subplot(1, 5, 5)
        plt.title("Model Input Image")
        plt.imshow(img_input.squeeze(), cmap="gray")
        plt.axis("off")

        timestamp = datetime.now().strftime("%H%M%S")
        plt.tight_layout()
        plt.savefig(
            f"{DEBUG_DIR}/optimized_debug_{timestamp}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"Debug visualization saved: {DEBUG_DIR}/optimized_debug_{timestamp}.png")

        # Step 6: Make prediction
        stroke_input_batch = np.expand_dims(stroke_input, axis=0)
        img_input_batch = np.expand_dims(img_input, axis=0)

        print("Running model inference...")
        predictions = model.predict([stroke_input_batch, img_input_batch], verbose=0)

        # Get top 5 predictions
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        top5_probs = predictions[0][top5_indices]

        print(f"\n=== Top 5 Predictions ===")
        for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
            confidence = prob * 100
            class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            print(f"{i + 1}. {class_name}: {confidence:.1f}%")

        # Confidence assessment
        best_confidence = top5_probs[0] * 100
        if best_confidence > 50:
            status = "🎯 High confidence"
        elif best_confidence > 30:
            status = "✅ Good confidence"
        elif best_confidence > 15:
            status = "⚠️  Moderate confidence"
        else:
            status = "❌ Low confidence"

        best_class = (
            class_names[top5_indices[0]]
            if top5_indices[0] < len(class_names)
            else f"Class_{top5_indices[0]}"
        )
        print(f"\n{status}: {best_class} ({best_confidence:.1f}%)")

        if best_confidence < 15:
            print("💡 Suggestion: Try drawing more clearly or add more detail")

    elif key == ord("c"):  # Clear
        canvas.fill(0)
        current_strokes = []
        print("Canvas cleared!")

    elif key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
print("Optimized inference app closed.")
