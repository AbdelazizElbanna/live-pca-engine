import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    GeomLines,
    NodePath,
    LColor,
    Mat4,
    TransparencyAttrib
)
from src.config import VisualConfig
from src.math_core.pca_box import PCABox

class PCABoxVisual:
    """Renders the PCA bounding box."""
    
    def __init__(self, parent_node: NodePath, config: VisualConfig):
        self.parent_node = parent_node
        self.config = config
        
        self.node_path = self.parent_node.attachNewNode("pca_box_transform")
        
        # We will create a unit box [-0.5, 0.5]^3 and apply transforms to it
        self._create_unit_box()
        
    def _create_unit_box(self):
        """Create a 1x1x1 unit box at the origin."""
        format = GeomVertexFormat.getV3()
        vdata = GeomVertexData("box_data", format, Geom.UHStatic)
        vdata.setNumRows(8)
        
        vertex = GeomVertexWriter(vdata, "vertex")
        
        # 8 corners of the unit box
        corners = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
            (-0.5, -0.5,  0.5), (0.5, -0.5,  0.5), (0.5, 0.5,  0.5), (-0.5, 0.5,  0.5)
        ]
        
        for c in corners:
            vertex.addData3f(*c)
            
        # Faces (Triangles)
        triangles = GeomTriangles(Geom.UHStatic)
        faces = [
            (0, 2, 1), (0, 3, 2), # Bottom
            (4, 5, 6), (4, 6, 7), # Top
            (0, 1, 5), (0, 5, 4), # Front
            (1, 2, 6), (1, 6, 5), # Right
            (2, 3, 7), (2, 7, 6), # Back
            (3, 0, 4), (3, 4, 7)  # Left
        ]
        for f in faces:
            triangles.addVertices(*f)
        triangles.closePrimitive()
        
        # Edges (Lines)
        lines = GeomLines(Geom.UHStatic)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4), # Top
            (0, 4), (1, 5), (2, 6), (3, 7)  # Pillars
        ]
        for e in edges:
            lines.addVertices(*e)
        lines.closePrimitive()
        
        # Setup Geoms
        face_geom = Geom(vdata)
        face_geom.addPrimitive(triangles)
        
        line_geom = Geom(vdata)
        line_geom.addPrimitive(lines)
        
        # Setup Nodes
        face_node = GeomNode("box_faces")
        face_node.addGeom(face_geom)
        self.faces_np = self.node_path.attachNewNode(face_node)
        self.faces_np.setColor(LColor(*self.config.box_face_color))
        self.faces_np.setTransparency(TransparencyAttrib.MAlpha)
        self.faces_np.setTwoSided(True)
        self.faces_np.setDepthWrite(True) # Force depth write so it cuts the points
        # Disable lighting for flat shaded look
        self.faces_np.setLightOff()
        self.faces_np.setBin("fixed", 20)
        
        line_node = GeomNode("box_edges")
        line_node.addGeom(line_geom)
        self.edges_np = self.node_path.attachNewNode(line_node)
        self.edges_np.setColor(LColor(*self.config.box_edge_color))
        self.edges_np.setRenderModeThickness(1.5) # Thinner, cleaner lines
        self.edges_np.setTransparency(TransparencyAttrib.MAlpha)
        self.edges_np.setAntialias(True)
        self.edges_np.setLightOff()
        self.edges_np.setBin("fixed", 21)


    def update_box(self, pca_box: PCABox, box_extent_3: float = None):
        """
        Update the box transform based on PCA results and compression state.
        
        Args:
            pca_box: PCABox containing center, extents, and axes
            box_extent_3: Optional override for the 3rd dimension extent (for compression)
        """
        # Create transformation matrix
        mat = Mat4.identMat()
        
        # 1. Scale
        extents = pca_box.extents.copy()
        if box_extent_3 is not None:
            extents[2] = box_extent_3
            
        # Prevent zero scale which might cause matrix issues
        extents = np.maximum(extents, 1e-5)
        
        scale_mat = Mat4.scaleMat(extents[0], extents[1], extents[2])
        mat = mat * scale_mat
        
        # 2. Rotation
        # pca_box.axes is a 3x3 rotation matrix, columns are eigenvectors
        rot = pca_box.axes
        rot_mat = Mat4(
            rot[0,0], rot[1,0], rot[2,0], 0.0,
            rot[0,1], rot[1,1], rot[2,1], 0.0,
            rot[0,2], rot[1,2], rot[2,2], 0.0,
            0.0,      0.0,      0.0,      1.0
        )
        mat = mat * rot_mat
        
        # 3. Translation
        c = pca_box.center
        trans_mat = Mat4.translateMat(c[0], c[1], c[2])
        mat = mat * trans_mat
        
        self.node_path.setMat(mat)
