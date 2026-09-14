import time
import cv2
import numpy as np
from collections import deque
from typing import Dict

class Profiler:
    """Tracks and calculates performance metrics per frame."""
    
    def __init__(self, history_size: int = 60, enabled: bool = False):
        self.enabled = enabled
        self.history_size = history_size
        
        self._starts: Dict[str, float] = {}
        self.history: Dict[str, deque] = {}
        
    def start(self, name: str):
        """Start timing a specific stage."""
        if not self.enabled:
            return
        self._starts[name] = time.perf_counter()
        
    def stop(self, name: str):
        """Stop timing a stage and record it."""
        if not self.enabled or name not in self._starts:
            return
            
        dt = time.perf_counter() - self._starts[name]
        
        if name not in self.history:
            self.history[name] = deque(maxlen=self.history_size)
            
        self.history[name].append(dt)
        del self._starts[name]
        
    def get_fps(self) -> float:
        """Calculate rolling average FPS."""
        if "total" not in self.history or len(self.history["total"]) == 0:
            return 0.0
        avg_time = sum(self.history["total"]) / len(self.history["total"])
        return 1.0 / avg_time if avg_time > 0 else 0.0
        
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance metrics onto the given BGR frame."""
        if not self.enabled:
            return frame
            
        y = 30
        fps = self.get_fps()
        
        # Color coding FPS
        color = (0, 255, 0)
        if fps < 20: color = (0, 0, 255)
        elif fps < 28: color = (0, 165, 255)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 25
        
        # Draw individual stages
        for name in ["camera", "hand_tracking", "interaction", "scene_update", "render", "composite"]:
            if name in self.history and len(self.history[name]) > 0:
                hist = self.history[name]
                avg_ms = (sum(hist) / len(hist)) * 1000.0
                max_ms = max(hist) * 1000.0
                
                text = f"{name}: {avg_ms:.1f}ms (max {max_ms:.1f})"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y += 20
                
        return frame
