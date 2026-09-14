import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomLines,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    LColor,
    TransparencyAttrib
)
from src.config import VisualConfig
from src.math_core.projection import ProjectionState

class ProjectionGuidesVisual:
    """Renders paths showing point trajectories during compression."""
    
    def __init__(self, parent_node: NodePath, config: VisualConfig, max_points: int):
        self.parent_node = parent_node
        self.config = config
        self.max_points = max_points
        self.n_points = 0
        
        # 1. Setup Vertex Format
        self.format = GeomVertexFormat.getV3c4()
        
        # 2. Create Vertex Data (2 vertices per line)
        self.vdata = GeomVertexData("guides_data", self.format, Geom.UHDynamic)
        self.vdata.setNumRows(max_points * 2)
        
        # 3. Create Geometry and Primitive
        self.primitive = GeomLines(Geom.UHDynamic)
        for i in range(max_points):
            self.primitive.addVertices(i * 2, i * 2 + 1)
        self.primitive.closePrimitive()
        
        self.geom = Geom(self.vdata)
        self.geom.addPrimitive(self.primitive)
        
        # 4. Create Node
        self.geom_node = GeomNode("projection_guides")
        self.geom_node.addGeom(self.geom)
        self.node_path = self.parent_node.attachNewNode(self.geom_node)
        
        # 5. Apply Visual Styles
        self.node_path.setRenderModeThickness(1.0)
        self.node_path.setAntialias(True)
        self.node_path.setTransparency(TransparencyAttrib.MAlpha)
        self.node_path.setLightOff()

    def update_guides(self, proj_state: ProjectionState, base_color: LColor):
        """
        Update the guides' positions and opacity.
        
        Args:
            proj_state: Current ProjectionState containing positions and target_projections.
            base_color: Base color for the guides (LColor with updated alpha).
        """
        n_points = proj_state.positions.shape[0]
        if n_points > self.max_points:
            n_points = self.max_points
            
        self.vdata.setNumRows(n_points * 2)
        
        # We only show guides for a subset to avoid visual clutter
        # Let's say we show all of them if N <= 100, or a sampled subset if larger.
        # For simplicity, we just draw all of them since in-place update is fast.
        
        vertex_writer = GeomVertexWriter(self.vdata, "vertex")
        color_writer = GeomVertexWriter(self.vdata, "color")
        
        # Alpha is max at c=0.5, 0 at c=0 and c=1.
        # Or simpler: Alpha = pc3_alpha * c * 2 (fades in as c>0, fades out as c->1)
        
        r, g, b, a = base_color.getX(), base_color.getY(), base_color.getZ(), base_color.getW()
        
        for i in range(n_points):
            # Start vertex (current position)
            vertex_writer.setData3f(proj_state.positions[i, 0], proj_state.positions[i, 1], proj_state.positions[i, 2])
            color_writer.setData4f(r, g, b, a)
            
            # End vertex (target projection)
            vertex_writer.setData3f(proj_state.target_projections[i, 0], proj_state.target_projections[i, 1], proj_state.target_projections[i, 2])
            # Fade out towards the target for a cool effect
            color_writer.setData4f(r, g, b, a * 0.2)
