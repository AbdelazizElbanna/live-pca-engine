import os
import urllib.request
import logging
# pyrefly: ignore [missing-import]
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional

from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python import BaseOptions

logger = logging.getLogger(__name__)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "hand_landmarker.task")

@dataclass
class HandLandmarks:
    """Normalized 3D landmarks (x, y, z) for a single hand. Shape (21, 3)."""
    positions: np.ndarray
    handedness: str
    confidence: float

class HandTracker:
    """Detects hand landmarks using MediaPipe Tasks API."""
    
    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.5):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.landmarker = None
        
        self._ensure_model_exists()
        self._initialize_landmarker()
        
    def _ensure_model_exists(self):
        """Download the model file if it doesn't exist."""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            logger.info(f"Downloading MediaPipe hand landmarker model to {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("Download complete.")
            
    def _initialize_landmarker(self):
        """Initialize the MediaPipe HandLandmarker."""
        base_options = BaseOptions(model_asset_path=MODEL_PATH)
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE, # IMAGE mode avoids the tracking graph ROI bug entirely
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
    def process_frame(self, frame_bgr: np.ndarray) -> Optional[HandLandmarks]:
        """
        Process a BGR frame and return landmarks for the first detected hand.
        
        Args:
            frame_bgr: OpenCV BGR frame.
            
        Returns:
            List of HandLandmarks, empty list if no hands detected.
        """
        # Convert BGR to RGB and ensure memory is contiguous
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        
        # CRITICAL FIX for MediaPipe C++ Core Dump:
        # Pad image to a perfect square so the initial IMAGE mode ROI is perfectly square.
        import cv2
        h, w = frame_rgb.shape[:2]
        size = max(h, w)
        pad_top = (size - h) // 2
        pad_bottom = size - h - pad_top
        pad_left = (size - w) // 2
        pad_right = size - w - pad_left
        
        square_rgb = cv2.copyMakeBorder(frame_rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
        
        # To permanently kill the Core Dump, we forcefully resize the padded square to exactly 256x256,
        # which is the native input size of the Hand Landmarker model.
        square_rgb = cv2.resize(square_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=square_rgb)
        
        # Detect using IMAGE mode
        result = self.landmarker.detect(mp_image)
        
        if not result.hand_landmarks:
            return []
            
        hands = []
        for i, landmarks in enumerate(result.hand_landmarks):
            handedness_info = result.handedness[i][0]
            
            # Since camera frame is horizontally mirrored (selfie mode),
            # reflection inverts chirality: MediaPipe's "Left" label corresponds to the user's physical RIGHT hand.
            is_physical_right = (handedness_info.category_name == "Left")
            
            # User explicitly requested: ONLY track the physical RIGHT hand.
            # Ignore the left hand completely.
            if not is_physical_right:
                continue
                
            positions = np.zeros((21, 3), dtype=np.float32)
            for j, lm in enumerate(landmarks):
                positions[j] = [lm.x, lm.y, lm.z]
                
            hands.append(HandLandmarks(
                positions=positions,
                handedness="Right",
                confidence=handedness_info.score
            ))
            
        return hands
        
    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
