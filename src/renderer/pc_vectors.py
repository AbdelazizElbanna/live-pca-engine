import numpy as np
from panda3d.core import (
    LineSegs,
    NodePath,
    LColor
)
from src.config import VisualConfig
from src.math_core.pca_engine import PCAResult

class PCVectorsVisual:
    """Renders the principal component vectors using LineSegs."""
    
    def __init__(self, parent_node: NodePath, config: VisualConfig):
        self.parent_node = parent_node
        self.config = config
        self.node_path = None
        
        # We will hold references to individual line node paths if we want to change alpha independently
        self.pc_nodes = []
        
        # Arrow settings
        self.line_thickness = 8.0 # Make arrows much thicker and visible
        self.arrow_head_size = 0.6
        self.arrow_head_angle = np.pi / 6  # 30 degrees

    def update_vectors(self, pca_result: PCAResult, pc3_alpha: float = 1.0):
        """
        Update or create the PC vectors.
        
        Convention (DEC-012): length is proportional to sqrt(eigenvalue) to represent std dev.
        """
        # Clear existing geometry
        if self.node_path is not None:
            self.node_path.removeNode()
        
        self.node_path = self.parent_node.attachNewNode("pc_vectors")
        self.pc_nodes = []
        
        mean = pca_result.mean
        eigenvalues = pca_result.eigenvalues
        eigenvectors = pca_result.eigenvectors
        
        colors = [
            self.config.pc1_color,
            self.config.pc2_color,
            self.config.pc3_color
        ]
        
        for i in range(3):
            # Calculate length: sqrt of eigenvalue corresponds to standard deviation
            length = np.sqrt(max(0, eigenvalues[i]))
            
            # The direction vector
            direction = eigenvectors[:, i]
            
            # Alpha/Scale handling for compression
            color = list(colors[i])
            if i == 2:  # PC3 (smallest)
                # Instead of fading alpha, we shrink the length smoothly
                length = length * pc3_alpha
                
            # End point is recalculated because length might have changed
            end_point = mean + direction * length
            
            # Create line segment
            lines = LineSegs()
            lines.setThickness(self.line_thickness)
            lines.setColor(LColor(*color))
            
            # Draw main line
            lines.moveTo(mean[0], mean[1], mean[2])
            lines.drawTo(end_point[0], end_point[1], end_point[2])
            
            # Draw arrow head (simple approach: two small lines)
            # Find arbitrary orthogonal vectors to create an arrow head in 3D
            if length > 0.001:
                # Up vector for the arrow head
                up = np.array([0.0, 0.0, 1.0])
                if np.abs(np.dot(direction, up)) > 0.99:
                    up = np.array([0.0, 1.0, 0.0])
                    
                right = np.cross(direction, up)
                right = right / np.linalg.norm(right)
                up = np.cross(right, direction)
                
                # Arrow head size
                head_len = min(length * 0.2, self.arrow_head_size)
                
                # Arrow lines
                p1 = end_point - direction * head_len + right * (head_len * np.tan(self.arrow_head_angle))
                p2 = end_point - direction * head_len - right * (head_len * np.tan(self.arrow_head_angle))
                p3 = end_point - direction * head_len + up * (head_len * np.tan(self.arrow_head_angle))
                p4 = end_point - direction * head_len - up * (head_len * np.tan(self.arrow_head_angle))
                
                lines.moveTo(end_point[0], end_point[1], end_point[2])
                lines.drawTo(p1[0], p1[1], p1[2])
                lines.moveTo(end_point[0], end_point[1], end_point[2])
                lines.drawTo(p2[0], p2[1], p2[2])
                lines.moveTo(end_point[0], end_point[1], end_point[2])
                lines.drawTo(p3[0], p3[1], p3[2])
                lines.moveTo(end_point[0], end_point[1], end_point[2])
                lines.drawTo(p4[0], p4[1], p4[2])
            
            # Create node
            node = lines.create()
            np_node = self.node_path.attachNewNode(node)
            
            # Disable depth test so vectors are always clearly visible over points
            np_node.setDepthTest(False)
            
            # Enable transparency if alpha < 1
            if color[3] < 1.0:
                np_node.setTransparency(True)
                
            if color[3] <= 0.001:
                np_node.hide()
                
            self.pc_nodes.append(np_node)
