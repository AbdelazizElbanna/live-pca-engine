"""
Central configuration for the Interactive Spatial PCA Visualization system.

All configurable parameters are defined here. Modules should import from
this file rather than hardcoding values.

This module has ZERO dependencies on any other project module.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class DataConfig:
    """Configuration for synthetic data generation."""
    seed: int = 42
    n_points: int = 480
    # Semi-axis lengths for the ellipsoid (determines variance structure).
    # Deliberately non-equal to ensure meaningful PCA results.
    # axis_lengths[0] > axis_lengths[1] > axis_lengths[2] guarantees
    # that PC1 captures the most variance and PC3 the least.
    axis_lengths: Tuple[float, float, float] = (5.0, 3.0, 1.0)
    noise_std: float = 0.3


@dataclass(frozen=True)
class RenderConfig:
    """Configuration for the 3D renderer."""
    window_width: int = 1280
    window_height: int = 720
    point_size: float = 0.5
    background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # Transparent for compositing


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for webcam capture."""
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(frozen=True)
class InteractionConfig:
    """Configuration for gesture-based interaction."""
    smoothing_alpha: float = 0.3  # EMA smoothing factor (0=no smoothing, 1=no change)
    rotation_sensitivity: float = 4.0
    zoom_sensitivity: float = 0.3
    compression_sensitivity: float = 1.0


@dataclass(frozen=True)
class VisualConfig:
    """Visual styling configuration.

    Aesthetic: modern, clean, scientific, premium.
    NO neon, cyberpunk, gamer HUD elements.
    """
    # Point cloud (Turquoise/Tech aesthetic)
    point_color: Tuple[float, float, float, float] = (0.40, 0.92, 0.88, 0.9) # Slightly lighter turquoise

    # Principal component vectors (Contrasting warm colors)
    pc1_color: Tuple[float, float, float, float] = (1.0, 0.2, 0.6, 1.0)  # Magenta
    pc2_color: Tuple[float, float, float, float] = (1.0, 0.6, 0.1, 1.0)  # Orange
    pc3_color: Tuple[float, float, float, float] = (1.0, 0.9, 0.1, 1.0)  # Yellow

    # PCA box (Holographic Cyan Wireframe)
    box_edge_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.9, 0.8)
    box_face_color: Tuple[float, float, float, float] = (0.0, 1.0, 0.9, 0.0) # Transparent faces

    # Projection plane
    plane_color: Tuple[float, float, float, float] = (0.9, 0.9, 0.95, 0.15)

    # Projection guides
    guide_color: Tuple[float, float, float, float] = (0.6, 0.6, 0.7, 0.3)


@dataclass
class AppConfig:
    """Top-level application configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)

    # Performance
    target_fps: int = 30
    enable_profiling: bool = False


# Default global configuration instance
DEFAULT_CONFIG = AppConfig()
