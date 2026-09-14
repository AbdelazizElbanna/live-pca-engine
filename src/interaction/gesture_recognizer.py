import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict
import time
from src.input.hand_tracker import HandLandmarks
from src.config import InteractionConfig

class GestureType(Enum):
    NONE = auto()
    ACTIVE = auto()
    STARTED = auto()

class GestureState(Enum):
    NONE = auto()
    STARTED = auto()
    ACTIVE = auto()
    RELEASED = auto()

@dataclass
class InteractionIntent:
    gesture_type: GestureType
    state: GestureState = GestureState.NONE
    target_rotation: tuple[float, float, float] = None
    rotation_delta: tuple[float, float] = None
    zoom_delta: float = 0.0
    target_compression: float = None
    compression_delta: float = 0.0
    hand_velocity: np.ndarray = None

class GestureRecognizer:
    """Maps smoothed hand landmarks to interaction intents using robust state machines."""
    
    def __init__(self, config: InteractionConfig):
        self.config = config
        self.previous_hands: Dict[str, HandLandmarks] = {}
        
        # State Tracking
        self.current_gesture = GestureType.NONE
        self.gesture_state = GestureState.NONE
        
        # EMA distance for zoom
        self.smoothed_hand_distance = 0.0
        
        # Zoom Tracking (Two-hand)
        self.initial_hand_distance = 0.0
        self.previous_hand_distance = 0.0
        
        # Deadzones & Hysteresis
        self.zoom_start_threshold = 0.02
        self.rotate_deadzone = 0.002
        
    def _is_fist(self, landmarks: HandLandmarks) -> bool:
        """Check if hand is closed in a fist (for compression)."""
        wrist = landmarks.positions[0]
        fingertips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky tips
        dists = [np.linalg.norm(landmarks.positions[i] - wrist) for i in fingertips]
        return all(d < 0.2 for d in dists)

    def _get_hand_center(self, landmarks: HandLandmarks) -> np.ndarray:
        """Get a highly stable center using Middle finger MCP to ensure robust tracking without finger-curl interference."""
        return landmarks.positions[9]

    def _is_back_facing(self, landmarks: HandLandmarks) -> bool:
        """
        Determines if the BACK of the hand is facing the screen (user sees their palm).
        Uses a robust 2D cross product of the Index and Pinky vectors from the wrist.
        """
        v_index = landmarks.positions[5] - landmarks.positions[0]
        v_pinky = landmarks.positions[17] - landmarks.positions[0]
        # 2D cross product gives the Z component (perpendicular to screen)
        cross_z = v_index[0] * v_pinky[1] - v_index[1] * v_pinky[0]
        
        # In a mirrored camera feed, for the physical Right hand:
        # Index is to the left of pinky when back of hand faces screen -> cross_z > 0
        # Palm facing screen -> cross_z < 0
        return cross_z > 0

    def recognize(self, current_landmarks: List[HandLandmarks]) -> InteractionIntent:
        """
        Supports 1-hand absolute rotation, 1-hand depth zoom, and momentum-based compression.
        """
        intent = InteractionIntent(gesture_type=GestureType.NONE, state=GestureState.NONE)
        
        if not current_landmarks:
            self.previous_hands.clear()
            self.current_gesture = GestureType.NONE
            self.frames_visible = 0
            return intent
            
        def non_linear_scale(val: float, power: float, multiplier: float) -> float:
            return np.sign(val) * (abs(val) ** power) * multiplier
            
        # 1-Hand Logic ONLY
        primary_hand = current_landmarks[0]
        # Ignore reported handedness for tracking continuity to prevent drops when the AI flips its guess
        h_id = "primary" 
        
        if h_id in self.previous_hands:
            prev_hand = self.previous_hands[h_id]
            self.frames_visible = getattr(self, 'frames_visible', 0) + 1
            
            intent.gesture_type = GestureType.ACTIVE
            intent.state = GestureState.ACTIVE
            
            # --- ROTATION (Momentum-based Dragging) ---
            curr_center = self._get_hand_center(primary_hand)
            
            if not hasattr(self, 'previous_hand_center'):
                self.previous_hand_center = curr_center
                self.smoothed_hand_center = curr_center
            else:
                # Slight EMA on hand center for jitter removal
                alpha_pos = 0.6
                self.smoothed_hand_center = (
                    alpha_pos * curr_center[0] + (1.0 - alpha_pos) * self.smoothed_hand_center[0],
                    alpha_pos * curr_center[1] + (1.0 - alpha_pos) * self.smoothed_hand_center[1],
                    alpha_pos * curr_center[2] + (1.0 - alpha_pos) * self.smoothed_hand_center[2]
                )
                
                # Calculate movement delta in screen space
                dx = self.smoothed_hand_center[0] - self.previous_hand_center[0]
                dy = self.smoothed_hand_center[1] - self.previous_hand_center[1]
                
                # Correct for normalized screen coordinates (16:9 aspect ratio)
                dx = dx * (16.0 / 9.0)
                
                # Use a tiny noise threshold instead of a hard deadzone to prevent tracking drops
                noise_threshold = 0.0001
                
                yaw_delta = 0.0
                pitch_delta = 0.0
                
                # Clutching mechanism (Ratchet): only rotate if BACK of hand is facing the screen (user sees their palm)
                if self._is_back_facing(primary_hand):
                    # Horizontal movement -> Yaw (Rotation around Z axis)
                    # Because camera frame is mirrored (selfie mode), moving physical hand RIGHT moves image RIGHT (dx > 0).
                    # Negating dx preserves the exact physical rotation direction as before.
                    if abs(dx) > noise_threshold:
                        yaw_delta = -dx * 95.0
                        
                    # Vertical movement -> Pitch (Rotation around X axis)
                    # NEGATIVE sign inverted here: moving hand UP (negative dy) should rotate object UP (positive pitch delta)
                    if abs(dy) > noise_threshold:
                        pitch_delta = -dy * 95.0
                        
                intent.rotation_delta = (yaw_delta, pitch_delta)
                
                self.previous_hand_center = self.smoothed_hand_center
            # -------------------------------
            
            # --- ZOOM & COMPRESSION (Depth) ---
            def get_robust_hand_size(hand: HandLandmarks):
                # Calculate 2D distances (ignoring Z to avoid depth scale inconsistencies)
                # 1. Palm width (Index MCP to Pinky MCP)
                w = np.linalg.norm(hand.positions[5][:2] - hand.positions[17][:2])
                # 2. Palm height (Wrist to Middle MCP)
                h = np.linalg.norm(hand.positions[0][:2] - hand.positions[9][:2])
                # Taking the max of width and height makes this metric completely robust
                # to the hand tilting (pitching forward or yawing sideways), as the 
                # non-foreshortened axis will always be preserved!
                return max(w, h)
                
            curr_size = get_robust_hand_size(primary_hand)
            
            # Delay initialization until tracking stabilizes
            if self.frames_visible < 5:
                self.smoothed_hand_size = curr_size
                self.previous_hand_size = curr_size
                intent.gesture_type = GestureType.STARTED
                return intent
                
            if self.current_gesture != GestureType.ACTIVE:
                self.current_gesture = GestureType.ACTIVE
                self.smoothed_hand_size = curr_size
                self.previous_hand_size = curr_size
                intent.gesture_type = GestureType.STARTED
                return intent
                
            # EMA smoothing for depth
            alpha_depth = 0.3 
            self.smoothed_hand_size = alpha_depth * curr_size + (1.0 - alpha_depth) * getattr(self, 'smoothed_hand_size', curr_size)
            
            # Momentum-based delta (ds)
            ds = self.smoothed_hand_size - self.previous_hand_size
            depth_deadzone = 0.0005
            
            if abs(ds) > depth_deadzone:
                clean_ds = np.sign(ds) * (abs(ds) - depth_deadzone)
                # Zoom velocity
                intent.zoom_delta = non_linear_scale(clean_ds, power=1.5, multiplier=2000.0 * self.config.zoom_sensitivity)
                # Compression velocity (same mechanics, drastically increased multiplier)
                intent.compression_delta = non_linear_scale(clean_ds, power=1.5, multiplier=400.0 * self.config.compression_sensitivity)
            else:
                intent.zoom_delta = 0.0
                intent.compression_delta = 0.0
                
            self.previous_hand_size = self.smoothed_hand_size

            
        else:
            # First frame of detection
            self.current_gesture = GestureType.STARTED
            intent.gesture_type = GestureType.STARTED
            
        self.previous_hands.clear()
        self.previous_hands[h_id] = primary_hand
            
        return intent
