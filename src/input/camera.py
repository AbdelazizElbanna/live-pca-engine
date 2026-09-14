import cv2
import numpy as np
import logging
from typing import Optional
from src.config import CameraConfig

logger = logging.getLogger(__name__)

class CameraCapture:
    """Manages OpenCV webcam capture and frame lifecycle."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
        self._is_open = False
        self._mock_frame = None

    def start(self) -> bool:
        """
        Open the camera using the configured device ID.
        Returns True if successful, False otherwise.
        """
        if self._mock_frame is not None:
            self._is_open = True
            return True
            
        try:
            self.cap = cv2.VideoCapture(self.config.device_id)
            if not self.cap.isOpened():
                logger.warning(f"Could not open camera device {self.config.device_id}")
                return False
                
            # Request specific resolution and framerate
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            self._is_open = True
            return True
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture and return a single frame.
        Returns None if the camera is unavailable or an error occurs.
        """
        if not self._is_open:
            return None
            
        if self._mock_frame is not None:
            return self._mock_frame.copy()
            
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to grab frame from camera")
                return None
            
            # Flip horizontally for natural mirror (selfie) reflection
            frame = cv2.flip(frame, 1)
            return frame
            
        return None

    def stop(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_open = False

    def inject_mock_frame(self, frame: np.ndarray):
        """Inject a synthetic frame for testing without physical camera hardware."""
        self._mock_frame = frame
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
