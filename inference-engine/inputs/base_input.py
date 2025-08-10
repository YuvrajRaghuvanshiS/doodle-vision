"""
Abstract base class for different input methods
"""

from abc import ABC, abstractmethod
from typing import List


class BaseInputHandler(ABC):
    """Abstract base class for input handlers"""

    def __init__(self):
        self.is_drawing = False
        self.current_stroke: List[List[float]] = []
        self.callbacks = {"stroke_start": [], "stroke_continue": [], "stroke_end": []}

    def add_callback(self, event_type: str, callback):
        """Add callback for input events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)

    def trigger_callback(self, event_type: str, data=None):
        """Trigger callbacks for specific events"""
        for callback in self.callbacks.get(event_type, []):
            callback(data)

    @abstractmethod
    def start_capture(self):
        """Start capturing input"""
        pass

    @abstractmethod
    def stop_capture(self):
        """Stop capturing input"""
        pass

    @abstractmethod
    def is_point_valid(self, x: int, y: int) -> bool:
        """Check if point is within valid drawing area"""
        pass
