import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    LColor,
    Mat4,
    TransparencyAttrib
)
from src.config import VisualConfig
from src.math_core.pca_box import PCABox

class PlaneVisual:
    """Renders the 2D projection plane."""
    
    def __init__(self, parent_node: NodePath, config: VisualConfig):
        self.parent_node = parent_node
        self.config = config
        
        self.node_path = self.parent_node.attachNewNode("projection_plane")
        
        # Create a unit quad [-0.5, 0.5]^2 on the XY plane
        self._create_unit_quad()
        
    def _create_unit_quad(self):
        format = GeomVertexFormat.getV3()
        vdata = GeomVertexData("plane_data", format, Geom.UHStatic)
        vdata.setNumRows(4)
        
        vertex = GeomVertexWriter(vdata, "vertex")
        
        corners = [
            (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)
        ]
        
        for c in corners:
            vertex.addData3f(*c)
            
        triangles = GeomTriangles(Geom.UHStatic)
        triangles.addVertices(0, 1, 2)
        triangles.addVertices(0, 2, 3)
        triangles.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        
        geom_node = GeomNode("plane_faces")
        geom_node.addGeom(geom)
        self.faces_np = self.node_path.attachNewNode(geom_node)
        self.faces_np.setColor(LColor(*self.config.plane_color))
        # Use MAlpha with DepthWrite=False so you can see through it, but DepthTest=True so points in front occlude it
        self.faces_np.setTransparency(TransparencyAttrib.MAlpha)
        self.faces_np.setDepthWrite(False)
        self.faces_np.setDepthTest(True)
        self.faces_np.setTwoSided(True)
        self.faces_np.setLightOff()
        
        # Force plane into the fixed bin so it draws after the point cloud ALWAYS
        self.faces_np.setBin("fixed", 30)

    def update_plane(self, pca_box: PCABox, pca_result, alpha: float):
        """
        Update plane transform and opacity.
        
        Args:
            pca_box: PCABox for extents
            pca_result: PCAResult for the exact mean and eigenvectors
            alpha: Opacity, typically mapped to compression state c
        """
        if alpha < 0.01:
            self.node_path.hide()
            return
            
        self.node_path.show()
        
        # Update color alpha
        r, g, b, _ = self.config.plane_color
        self.faces_np.setColor(LColor(r, g, b, alpha))
        
        # Transform (similar to box, but we don't care about Z extent)
        mat = Mat4.identMat()
        
        # 1. Scale by L1 and L2
        # Adding a bit of padding to the plane for aesthetic reasons
        padding = 1.2
        scale_x = max(pca_box.extents[0] * padding, 1e-5)
        scale_y = max(pca_box.extents[1] * padding, 1e-5)
        mat = mat * Mat4.scaleMat(scale_x, scale_y, 1.0)
        
        # 2. Rotation
        rot = pca_result.eigenvectors
        rot_mat = Mat4(
            rot[0,0], rot[1,0], rot[2,0], 0.0,
            rot[0,1], rot[1,1], rot[2,1], 0.0,
            rot[0,2], rot[1,2], rot[2,2], 0.0,
            0.0,      0.0,      0.0,      1.0
        )
        mat = mat * rot_mat
        
        # 3. Translation
        # The plane MUST be strictly at Z=0 in PCA space, which means it passes through the PCA mean,
        # offset only in X and Y by the box center to align visually with the box.
        # We calculate the box center in PCA space:
        center_proj = (pca_box.center - pca_result.mean) @ pca_result.eigenvectors
        center_proj[2] = 0.0 # Force Z to 0 so it perfectly aligns with target_projections!
        
        plane_center = pca_result.mean + center_proj @ pca_result.eigenvectors.T
        
        mat = mat * Mat4.translateMat(plane_center[0], plane_center[1], plane_center[2])
        
        self.node_path.setMat(mat)
