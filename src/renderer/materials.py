from panda3d.core import LColor
from src.config import VisualConfig

class MaterialSystem:
    """
    Manages visual styling, colors, and opacities for the rendering system.
    Provides methods to retrieve colors with dynamic alpha for animation effects.
    """
    def __init__(self, config: VisualConfig):
        self.config = config

    def _with_alpha(self, color_tuple: tuple, alpha_multiplier: float) -> LColor:
        """Returns an LColor with the base alpha multiplied by alpha_multiplier."""
        r, g, b, a = color_tuple
        return LColor(r, g, b, a * alpha_multiplier)

    def get_point_color(self, alpha: float = 1.0) -> LColor:
        """Points are the most prominent element in the visual hierarchy."""
        return self._with_alpha(self.config.point_color, alpha)

    def get_pc_color(self, axis: int, alpha: float = 1.0) -> LColor:
        """PCs are secondary in the visual hierarchy."""
        if axis == 0:
            return self._with_alpha(self.config.pc1_color, alpha)
        elif axis == 1:
            return self._with_alpha(self.config.pc2_color, alpha)
        else:
            return self._with_alpha(self.config.pc3_color, alpha)

    def get_box_edge_color(self, alpha: float = 1.0) -> LColor:
        """Box edges are tertiary in the visual hierarchy."""
        return self._with_alpha(self.config.box_edge_color, alpha)
        
    def get_box_face_color(self, alpha: float = 1.0) -> LColor:
        """Box faces are subtle background elements."""
        return self._with_alpha(self.config.box_face_color, alpha)

    def get_plane_color(self, alpha: float = 1.0) -> LColor:
        """Projection plane becomes visible during compression."""
        return self._with_alpha(self.config.plane_color, alpha)

    def get_guide_color(self, alpha: float = 1.0) -> LColor:
        """Projection guides are the subtlest elements, visible only during animation."""
        return self._with_alpha(self.config.guide_color, alpha)
