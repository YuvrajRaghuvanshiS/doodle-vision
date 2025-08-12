"""
Main application that orchestrates all components
"""

from typing import Optional

import cv2
from drawing_canvas import DrawingCanvas
from engine.prediction_engine import PredictionEngine
from inputs.base_input import BaseInputHandler
from inputs.gesture_input import GestureInputHandler
from inputs.mouse_input import MouseInputHandler
from processors.image_processor import ImageProcessor
from processors.stroke_processor import StrokeProcessor


class DrawingRecognitionApp:
    """Main application controller"""

    def __init__(
        self, model_path: str = "models/best_model_345_classes_30000_examples.keras"
    ):
        # Initialize components
        self.canvas = DrawingCanvas(640, 480)
        self.stroke_processor = StrokeProcessor()
        self.image_processor = ImageProcessor()
        self.prediction_engine = PredictionEngine(model_path, num_classes=345)

        # Input handler (will be set based on mode)
        self.input_handler: Optional[BaseInputHandler] = None
        self.input_mode = "mouse"  # or "gesture"

        # UI settings
        self.window_name = "Drawing Recognition App"
        self.running = False

    def set_input_mode(self, mode: str):
        """Switch between input modes"""
        if self.input_handler:
            self.input_handler.stop_capture()

        self.input_mode = mode

        if mode == "mouse":
            self.input_handler = MouseInputHandler(
                self.window_name, (self.canvas.width, self.canvas.height)
            )
        elif mode == "gesture":
            self.input_handler = GestureInputHandler(
                (self.canvas.width, self.canvas.height)
            )
        else:
            raise ValueError(f"Unknown input mode: {mode}")

        # Setup callbacks
        self._setup_input_callbacks()
        self.input_handler.start_capture()

    def _setup_input_callbacks(self):
        """Setup callbacks for input events"""
        self.input_handler.add_callback("stroke_start", self._on_stroke_start)
        self.input_handler.add_callback("stroke_continue", self._on_stroke_continue)
        self.input_handler.add_callback("stroke_end", self._on_stroke_end)

    def _on_stroke_start(self, data):
        """Handle stroke start event"""
        print(f"Stroke started at ({data['x']}, {data['y']})")

    def _on_stroke_continue(self, data):
        """Handle stroke continue event"""
        self.canvas.draw_line(data["from"], data["to"])

    def _on_stroke_end(self, data):
        """Handle stroke end event"""
        self.canvas.add_stroke(data["stroke"])
        print(f"Stroke completed with {len(data['stroke'])} points")

    def run(self):
        """Run the application"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self.set_input_mode(self.input_mode)
        self.running = True

        print("Drawing Recognition App Started!")
        print(f"Current input mode: {self.input_mode}")
        print("Controls:")
        print("- 'p' to predict")
        print("- 'c' to clear canvas")
        print("- 'm' to switch to mouse mode")
        print("- 'g' to switch to gesture mode")
        print("- ESC to exit")

        while self.running:
            self._update_display()
            self._handle_keyboard()

            # Process gesture input if in gesture mode
            if self.input_mode == "gesture" and isinstance(
                self.input_handler, GestureInputHandler
            ):
                frame = self.input_handler.process_frame()
                if frame is not None:
                    cv2.imshow("Camera Feed", frame)

        self._cleanup()

    def _update_display(self):
        """Update the main display"""
        display = self.canvas.get_canvas_copy()
        stats = self.canvas.get_stroke_stats()

        # Add UI information
        self.canvas.add_ui_text(
            display,
            f"Mode: {self.input_mode} | Strokes: {stats['num_strokes']} | Points: {stats['total_points']}",
            (10, 30),
        )
        self.canvas.add_ui_text(
            display,
            "Press 'p' to predict, 'c' to clear, 'm'/'g' to switch mode, ESC to exit",
            (10, 460),
        )

        cv2.imshow(self.window_name, display)

    def _handle_keyboard(self):
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF

        if key == ord("p"):
            self._predict_drawing()
        elif key == ord("c"):
            self._clear_canvas()
        elif key == ord("m"):
            self.set_input_mode("mouse")
            print("Switched to mouse input mode")
        elif key == ord("g"):
            self.set_input_mode("gesture")
            print("Switched to gesture input mode")
        elif key == 27:  # ESC
            self.running = False

    def _predict_drawing(self):
        """Prediction with debug visualization"""
        strokes = self.canvas.stroke_history

        if not strokes or all(len(stroke) <= 1 for stroke in strokes):
            print("No drawing detected! Please draw something first.")
            return

        print("\n=== Processing Drawing ===")
        stats = self.canvas.get_stroke_stats()
        print(f"Input: {stats['num_strokes']} strokes, {stats['total_points']} points")

        # Process stroke data (keep intermediate results)
        optimized_points = self.stroke_processor.optimize_strokes(strokes)
        processed_strokes = self.stroke_processor.preprocess_for_model(optimized_points)

        # Process image data
        canvas_copy = self.canvas.get_canvas_copy()
        processed_image = self.image_processor.process_canvas(canvas_copy)

        # Make prediction with ordered debug visualization
        predictions, debug_path = self.prediction_engine.predict(
            processed_strokes, processed_image, strokes, optimized_points, canvas_copy
        )

        # Display results
        print("\n=== Top 5 Predictions ===")
        for i, (class_name, confidence) in enumerate(predictions):
            print(f"{i + 1}. {class_name}: {confidence:.1f}%")

        best_class, best_confidence = predictions[0]
        status = self.prediction_engine.get_confidence_status(best_confidence)
        print(f"\n{status}: {best_class} ({best_confidence:.1f}%)")

        if debug_path:
            print(f"Debug visualization saved: {debug_path}")

    def _clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.clear()
        print("Canvas cleared!")

    def _cleanup(self):
        """Clean up resources"""
        if self.input_handler:
            self.input_handler.stop_capture()
        cv2.destroyAllWindows()
        print("Application closed.")


if __name__ == "__main__":
    app = DrawingRecognitionApp()
    app.run()
