import os

# from datetime import datetime
from typing import List, Tuple, Union

# import cv2
# import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorflow.keras.models import load_model as keras_load_model
except ImportError:
    keras_load_model = None

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class PredictionEngine:
    """Prediction engine with streamlined debug visualization"""

    def __init__(self, model_path: str, debug_dir: str = "debug", debug: bool = False):
        self.debug = debug
        self.model_path = model_path
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)

        if model_path.endswith(".tflite"):
            if tflite is None:
                raise ImportError(
                    "tflite_runtime is required for loading .tflite models."
                )
            self.model_type = "tflite"
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        elif model_path.endswith(".onnx"):
            if ort is None:
                raise ImportError("onnxruntime is required for loading .onnx models.")
            self.model_type = "onnx"
            self.session = ort.InferenceSession(model_path)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_name = self.session.get_outputs()[0].name
        else:
            self.model_type = "keras"
            if keras_load_model is None:
                raise ImportError("TensorFlow is required for loading .keras models.")
            self.model = keras_load_model(model_path)

        self.class_names = [
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

    def predict(
        self,
        stroke_input: np.ndarray,
        image_input: np.ndarray,
        raw_strokes: List[List[List[float]]],
        optimized_points: np.ndarray,
        canvas: np.ndarray,
    ) -> Tuple[List[Tuple[str, float]], Union[str, None]]:
        """Make prediction with optional debug visualization"""

        # Prepare batch inputs
        stroke_batch = np.expand_dims(stroke_input, axis=0)
        image_batch = np.expand_dims(image_input, axis=0)

        if self.model_type == "tflite":
            # Ensure stroke_batch is (1, 130, 3)
            if stroke_batch.ndim == 3:
                pass  # already fine
            elif stroke_batch.ndim == 2:
                stroke_batch = np.expand_dims(stroke_batch, axis=0)
            else:
                raise ValueError(
                    f"Unexpected shape for stroke_batch: {stroke_batch.shape}"
                )

            # Ensure image_batch is (1, 28, 28, 1)
            if image_batch.ndim == 3:
                image_batch = np.expand_dims(image_batch, axis=0)
            elif image_batch.ndim == 4:
                pass  # already correct
            else:
                raise ValueError(
                    f"Unexpected shape for image_batch: {image_batch.shape}"
                )

            self.interpreter.set_tensor(self.input_details[0]["index"], image_batch)
            self.interpreter.set_tensor(self.input_details[1]["index"], stroke_batch)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]["index"])
        elif self.model_type == "onnx":
            inputs = {
                self.input_names[0]: image_batch.astype(np.float32),
                self.input_names[1]: stroke_batch.astype(np.float32),
            }
            predictions = self.session.run([self.output_name], inputs)[0]
        else:
            predictions = self.model.predict([stroke_batch, image_batch], verbose=0)

        # Get top 5 predictions
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        top5_probs = predictions[0][top5_indices] * 100
        results = [
            (
                self.class_names[idx]
                if idx < len(self.class_names)
                else f"Class_{idx}",
                prob,
            )
            for idx, prob in zip(top5_indices, top5_probs)
        ]

        debug_path = None
        if self.debug:
            debug_path = self._create_ordered_debug_viz(
                stroke_input,
                image_input,
                raw_strokes,
                optimized_points,
                canvas,
                results,
            )

        return results, debug_path

    # def _create_ordered_debug_viz(
    #     self,
    #     stroke_input: np.ndarray,
    #     image_input: np.ndarray,
    #     raw_strokes: List[List[List[float]]],
    #     optimized_points: np.ndarray,
    #     canvas: np.ndarray,
    #     predictions: List[Tuple[str, float]],
    # ) -> str:
    #     """Create simplified debug visualization with specific plot order"""

    #     # Create figure with 6 subplots in 2 rows
    #     fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    #     # Color palette for raw strokes
    #     colors = [
    #         "red",
    #         "blue",
    #         "green",
    #         "orange",
    #         "purple",
    #         "cyan",
    #         "magenta",
    #         "brown",
    #         "pink",
    #         "gray",
    #         "olive",
    #         "navy",
    #     ]

    #     total_points = sum(len(stroke) for stroke in raw_strokes)

    #     # === IMAGE PLOTS (Top Row) ===

    #     # Plot 1: Original Canvas Image
    #     axes[0, 0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cmap="gray")
    #     axes[0, 0].set_title("Original Canvas Image", fontweight="bold", fontsize=14)
    #     axes[0, 0].axis("off")

    #     # Plot 2: Model Input Image
    #     axes[0, 1].imshow(image_input.squeeze(), cmap="gray")
    #     axes[0, 1].set_title("Model Input Image", fontweight="bold", fontsize=14)
    #     axes[0, 1].axis("off")

    #     # Plot 3: Top Five Predictions Map
    #     classes = [pred[0] for pred in predictions]
    #     probs = [pred[1] for pred in predictions]

    #     # Create horizontal bar chart with color gradient
    #     bars = axes[0, 2].barh(
    #         range(len(classes)),
    #         probs,
    #         color=plt.cm.viridis(np.linspace(0.2, 0.8, len(classes))),
    #     )
    #     axes[0, 2].set_yticks(range(len(classes)))
    #     axes[0, 2].set_yticklabels(classes, fontsize=11)
    #     axes[0, 2].set_xlabel("Confidence (%)", fontweight="bold")
    #     axes[0, 2].set_title("Top Five Predictions Map", fontweight="bold", fontsize=14)
    #     axes[0, 2].invert_yaxis()
    #     axes[0, 2].grid(axis="x", alpha=0.3)

    #     # Add percentage labels on bars
    #     for i, (bar, prob) in enumerate(zip(bars, probs)):
    #         axes[0, 2].text(
    #             prob + max(probs) * 0.01,
    #             i,
    #             f"{prob:.1f}%",
    #             va="center",
    #             fontweight="bold",
    #             fontsize=10,
    #         )

    #     # === STROKE PLOTS (Bottom Row) ===

    #     # Plot 4: Raw (strokes) Input
    #     for i, stroke in enumerate(raw_strokes):
    #         if len(stroke) > 0:
    #             stroke_arr = np.array(stroke)
    #             axes[1, 0].plot(
    #                 stroke_arr[:, 0],
    #                 stroke_arr[:, 1],
    #                 color=colors[i % len(colors)],
    #                 marker="o",
    #                 markersize=3,
    #                 linewidth=2,
    #                 alpha=0.7,
    #                 label=f"Stroke {i + 1}",
    #             )

    #     axes[1, 0].set_title(
    #         f"Raw (strokes) Input\n({len(raw_strokes)} strokes, {total_points} points)",
    #         fontweight="bold",
    #         fontsize=14,
    #     )
    #     axes[1, 0].set_aspect("equal")
    #     axes[1, 0].grid(True, alpha=0.3)

    #     # Add legend only if reasonable number of strokes
    #     if len(raw_strokes) <= 5:
    #         axes[1, 0].legend(fontsize=9, loc="upper right")

    #     # Plot 5: Optimized Points
    #     if len(optimized_points) > 0:
    #         axes[1, 1].plot(
    #             optimized_points[:, 0],
    #             optimized_points[:, 1],
    #             "g-o",
    #             linewidth=2,
    #             alpha=0.8,
    #             markersize=4,
    #         )
    #         # Color-code by pen state
    #         scatter = axes[1, 1].scatter(
    #             optimized_points[:, 0],
    #             optimized_points[:, 1],
    #             c=optimized_points[:, 2],
    #             cmap="RdYlBu",
    #             s=50,
    #             edgecolors="black",
    #             linewidth=0.5,
    #             alpha=0.9,
    #         )
    #         # Add colorbar
    #         cbar = plt.colorbar(scatter, ax=axes[1, 1], shrink=0.8)
    #         cbar.set_label("Pen State", fontweight="bold")

    #     axes[1, 1].set_title(
    #         f"Optimized Points\n({len(optimized_points)} points)",
    #         fontweight="bold",
    #         fontsize=14,
    #     )
    #     axes[1, 1].set_aspect("equal")
    #     axes[1, 1].grid(True, alpha=0.3)

    #     # Plot 6: Scaled Strokes
    #     valid_mask = np.any(stroke_input != 0, axis=1)
    #     if np.any(valid_mask):
    #         axes[1, 2].plot(
    #             stroke_input[valid_mask, 0],
    #             stroke_input[valid_mask, 1],
    #             "b-",
    #             linewidth=2,
    #             alpha=0.8,
    #         )
    #         scatter = axes[1, 2].scatter(
    #             stroke_input[valid_mask, 0],
    #             stroke_input[valid_mask, 1],
    #             c=stroke_input[valid_mask, 2],
    #             cmap="RdYlBu",
    #             s=30,
    #             alpha=0.9,
    #         )

    #     axes[1, 2].set_title(
    #         "Scaled Strokes\n[-100, 100] Range", fontweight="bold", fontsize=14
    #     )
    #     axes[1, 2].set_xlim(-110, 110)
    #     axes[1, 2].set_ylim(-110, 110)
    #     axes[1, 2].grid(True, alpha=0.3)
    #     axes[1, 2].set_aspect("equal")

    #     # Add overall title
    #     best_class, best_confidence = predictions[0]
    #     fig.suptitle(
    #         f"Drawing Recognition Debug Report\n"
    #         f"Best Prediction: {best_class} ({best_confidence:.1f}%)",
    #         fontsize=16,
    #         fontweight="bold",
    #         y=0.98,
    #     )

    #     # Add metadata
    #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     fig.text(
    #         0.02,
    #         0.02,
    #         f"Generated: {timestamp} | "
    #         f"Strokes: {len(raw_strokes)} | Total Points: {total_points} | "
    #         f"Optimized: {len(optimized_points)} points",
    #         fontsize=10,
    #         style="italic",
    #     )

    #     # Adjust layout
    #     plt.tight_layout()
    #     plt.subplots_adjust(top=0.91, bottom=0.08, hspace=0.3, wspace=0.3)

    #     # Save with timestamp
    #     timestamp_file = datetime.now().strftime("%H%M%S")
    #     debug_path = f"{self.debug_dir}/debug_{timestamp_file}.png"
    #     plt.savefig(debug_path, dpi=200, bbox_inches="tight", facecolor="white")
    #     plt.close()

    #     return debug_path

    def get_confidence_status(self, confidence: float) -> str:
        """Get confidence status message"""
        if confidence > 50:
            return "🎯 High confidence"
        elif confidence > 30:
            return "✅ Good confidence"
        elif confidence > 15:
            return "⚠️  Moderate confidence"
        else:
            return "❌ Low confidence"
