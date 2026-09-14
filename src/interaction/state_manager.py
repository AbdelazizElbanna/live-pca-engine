import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.config import AppConfig
from src.math_core.pca_engine import PCAResult, compute_pca
from src.math_core.pca_box import PCABox, compute_pca_box
from src.math_core.data_generator import generate_dataset
from src.interaction.gesture_recognizer import InteractionIntent, GestureType
from enum import Enum

class InteractionMode(Enum):
    IDLE = 0
    ROTATION = 1
    ZOOM = 2
    COMPRESSION = 3

@dataclass
class SceneState:
    """Immutable snapshot of the scene state for the renderer."""
    dataset: np.ndarray
    pca_result: PCAResult
    pca_box: PCABox
    camera_rotation: tuple[float, float]
    object_rotation_mat: np.ndarray
    camera_zoom: float
    compression_c: float

class StateManager:
    """Central scene state management."""
    
    def __init__(self, config: AppConfig, initial_mode: InteractionMode = InteractionMode.IDLE):
        self.config = config
        
        # Camera State
        self.camera_rotation = (0.0, 0.0) # (yaw, pitch) (kept for default viewing angle)
        self.camera_zoom = 1.0 # 1.0 is default distance
        self.current_mode = initial_mode
        self.last_active_mode = InteractionMode.ROTATION
        
        # Object Orientation State (Trackball Matrix)
        self.object_rotation_mat = np.eye(3, dtype=np.float32)
        self.velocity_rotation = [0.0, 0.0] # (Yaw velocity, Pitch velocity)
        
        # Physics / Momentum State
        self.velocity_zoom = 0.0
        self.velocity_compression = 0.0
        self.friction = 0.94 # slightly heavier friction for smooth stop
        
        # Math State (Absolute)
        self.compression_c = 0.0
        
        # Data
        self.dataset = generate_dataset(
            seed=config.data.seed,
            n_points=config.data.n_points,
            axis_lengths=config.data.axis_lengths,
            noise_std=config.data.noise_std
        )
        self.pca_result = compute_pca(self.dataset)
        self.pca_box = compute_pca_box(self.pca_result)
        
        # Handlers
        from src.interaction.compression_handler import CompressionHandler
        self.compression_handler = CompressionHandler(config.interaction)
        
    def set_mode(self, mode: InteractionMode) -> None:
        """Set the current interaction mode."""
        if mode == InteractionMode.IDLE:
            self.velocity_rotation = [0.0, 0.0]
            self.velocity_zoom = 0.0
            self.velocity_compression = 0.0
        else:
            self.last_active_mode = mode
        self.current_mode = mode

    def _rot_x(self, angle_deg: float) -> np.ndarray:
        r = np.radians(angle_deg)
        c, s = np.cos(r), np.sin(r)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        
    def _rot_z(self, angle_deg: float) -> np.ndarray:
        r = np.radians(angle_deg)
        c, s = np.cos(r), np.sin(r)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    def update(self, dt: float) -> None:
        """Apply absolute state tracking and momentum every frame."""
        if self.current_mode == InteractionMode.IDLE:
            self.velocity_rotation = [0.0, 0.0]
            self.velocity_zoom = 0.0
            self.velocity_compression = 0.0
            return

        time_scale = dt * 60.0
        
        # 1. Rotation (World-Space Trackball Momentum)
        yaw_vel = self.velocity_rotation[0] * time_scale
        pitch_vel = self.velocity_rotation[1] * time_scale
        
        if abs(yaw_vel) > 0.001 or abs(pitch_vel) > 0.001:
            # We apply yaw (Z) and pitch (X) in WORLD space to get a trackball effect.
            # Pre-multiply applies the rotation in world space.
            delta_mat = self._rot_z(-yaw_vel) @ self._rot_x(-pitch_vel)
            self.object_rotation_mat = delta_mat @ self.object_rotation_mat
            
            # Re-orthogonalize to prevent floating point drift
            u, _, vh = np.linalg.svd(self.object_rotation_mat)
            self.object_rotation_mat = u @ vh
        
        # 2. Multiplicative Zoom (Momentum-based)
        self.camera_zoom *= (1.0 + self.velocity_zoom * time_scale)
        self.camera_zoom = np.clip(self.camera_zoom, 0.1, 20.0)
        
        # 3. Compression (Momentum-based)
        self.compression_c += self.velocity_compression * time_scale
        
        # Add a magnetic pull (spring) to the extremes (0.0 or 1.0) so it doesn't get stuck in the middle.
        # This solves the user's issue where the plane doesn't fully cut the data if the gesture is incomplete.
        # IMPORTANT: Only engage the spring if the user is explicitly NOT actively trying to compress.
        # This prevents the spring from fighting the user when they are moving their hand slowly.
        is_actively_compressing = (self.current_mode == InteractionMode.COMPRESSION and abs(self.velocity_compression) > 0.001)
        
        if not is_actively_compressing:
            if self.compression_c > 0.3: # bias towards compression if they made a decent effort
                self.compression_c += (1.0 - self.compression_c) * 0.08 * time_scale
            elif self.compression_c > 0.0:
                self.compression_c -= self.compression_c * 0.08 * time_scale
                
        self.compression_c = np.clip(self.compression_c, 0.0, 1.0)
        
        # SNAPPING: This mathematically guarantees the exact tangent intersection at the end of the gesture!
        if self.velocity_compression > 0 and (1.0 - self.compression_c) < 0.005:
            self.compression_c = 1.0
        if self.velocity_compression < 0 and self.compression_c < 0.005:
            self.compression_c = 0.0
        
        # Apply friction to momentum-based variables
        decay = self.friction ** time_scale
        self.velocity_zoom *= decay
        self.velocity_compression *= decay
        self.velocity_rotation[0] *= decay
        self.velocity_rotation[1] *= decay
        
        if abs(self.velocity_zoom) < 0.001: self.velocity_zoom = 0.0
        if abs(self.velocity_compression) < 0.001: self.velocity_compression = 0.0
        if abs(self.velocity_rotation[0]) < 0.001: self.velocity_rotation[0] = 0.0
        if abs(self.velocity_rotation[1]) < 0.001: self.velocity_rotation[1] = 0.0

    def apply_intent(self, intent: InteractionIntent) -> None:
        """Apply an interaction intent to update absolute targets or add momentum."""
        if self.current_mode == InteractionMode.IDLE:
            return

        if intent.gesture_type == GestureType.NONE or intent.gesture_type == GestureType.STARTED:
            return
            
        if self.current_mode == InteractionMode.ROTATION:
            if hasattr(intent, 'rotation_delta') and intent.rotation_delta is not None:
                # Add momentum based on hand movement
                self.velocity_rotation[0] += intent.rotation_delta[0]
                self.velocity_rotation[1] += intent.rotation_delta[1]
            
        elif self.current_mode == InteractionMode.ZOOM:
            self.velocity_zoom = intent.zoom_delta
            
        elif self.current_mode == InteractionMode.COMPRESSION:
            if hasattr(intent, 'compression_delta'):
                # Lerp the velocity for ultra smooth starts and stops (removes stutter/frame drops)
                self.velocity_compression += (intent.compression_delta - self.velocity_compression) * 0.15
                
    def get_scene_state(self) -> SceneState:
        """Get a snapshot of the current state."""
        return SceneState(
            dataset=self.dataset,
            pca_result=self.pca_result,
            pca_box=self.pca_box,
            camera_rotation=self.camera_rotation,
            object_rotation_mat=self.object_rotation_mat.copy(),
            camera_zoom=self.camera_zoom,
            compression_c=self.compression_c
        )
