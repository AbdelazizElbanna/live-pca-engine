import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomPoints,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    LColor,
    Texture,
    TexGenAttrib,
    TransparencyAttrib,
    TextureStage,
    Filename
)
from src.config import VisualConfig
import os

class PointCloudVisual:
    """Renders a 3D point cloud using highly optimized Panda3D GeomPoints."""
    
    def __init__(self, parent_node: NodePath, config: VisualConfig, max_points: int):
        self.parent_node = parent_node
        self.config = config
        self.max_points = max_points
        self.n_points = 0
        
        self.format = GeomVertexFormat.getV3c4()
        self.vdata = GeomVertexData("points", self.format, Geom.UHStatic)
        self.vdata.setNumRows(max_points)
        
        self.points = GeomPoints(Geom.UHStatic)
        self.geom = Geom(self.vdata)
        self.geom.addPrimitive(self.points)
        
        self.node = GeomNode("point_cloud")
        self.node.addGeom(self.geom)
        
        self.node_path = self.parent_node.attachNewNode(self.node)
        
        # Load the texture using cross-platform absolute path resolution
        assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "sphere.png")
        tex = loader.loadTexture(Filename.fromOsSpecific(assets_path))
        self.node_path.setTexture(tex, 1)
        
        # Enable Point Sprites to map the texture onto each point
        self.node_path.setTexGen(TextureStage.getDefault(), TexGenAttrib.MPointSprite)
        
        # Set DepthWrite and MAlpha so the points physically write to the depth buffer and can be cut by the plane
        self.node_path.setTransparency(TransparencyAttrib.MAlpha)
        self.node_path.setDepthWrite(True)
        self.node_path.setDepthTest(True)
        
        # Ensure it renders before the plane (sort 10 vs 30)
        self.node_path.setBin("fixed", 10)
        
        self.node_path.setRenderModeThickness(20.0)

    def set_point_size(self, size: float):
        """Dynamically update the point size in pixels."""
        self.node_path.setRenderModeThickness(size)

    def update_points(self, positions: np.ndarray, colors: np.ndarray = None):
        """
        Update the positions (and optionally colors) of the point cloud.
        """
        n_points = positions.shape[0]
        if n_points > self.max_points:
            raise ValueError(f"Cannot render {n_points} points, max_points is {self.max_points}")
            
        self.vdata.setNumRows(n_points)
        
        # Position
        vertex_writer = GeomVertexWriter(self.vdata, "vertex")
        for i in range(n_points):
            vertex_writer.setData3f(*positions[i])
            
        # Color
        color_writer = GeomVertexWriter(self.vdata, "color")
        default_color = self.config.point_color
        for i in range(n_points):
            if colors is not None:
                color_writer.setData4f(*colors[i])
            else:
                color_writer.setData4f(*default_color)
                
        # Primitive structure
        if n_points != self.n_points:
            self.points.clearVertices()
            self.points.addConsecutiveVertices(0, n_points)
            self.points.closePrimitive()
            self.n_points = n_points
