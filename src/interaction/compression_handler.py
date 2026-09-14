import numpy as np
from src.config import InteractionConfig
from src.math_core.pca_engine import PCAResult

class CompressionHandler:
    """Handles mapping hand velocity to compression state."""
    
    def __init__(self, config: InteractionConfig):
        self.config = config
        
    def compute_delta(self, hand_velocity: np.ndarray, pca_result: PCAResult) -> float:
        """
        Compute the compression delta based on hand velocity projected along the smallest PC axis.
        
        Args:
            hand_velocity: 3D vector (dx, dy, dz) of hand movement.
            pca_result: Current PCA result to get the smallest PC.
            
        Returns:
            Scalar compression delta.
        """
        if np.linalg.norm(hand_velocity) < 1e-6:
            return 0.0
            
        # The smallest PC is the 3rd eigenvector (index 2)
        v3 = pca_result.eigenvectors[:, 2]
        
        # Project hand velocity onto v3
        projection = np.dot(hand_velocity, v3)
        
        # Map to compression delta
        # Since v3 direction is arbitrary, we might want to take the absolute projection
        # or require the user to move towards the origin.
        # Let's assume motion towards the plane (collapsing the box) increases compression.
        # This can be tricky if we don't know which way the hand is relative to the box.
        # A simpler approach: use the magnitude of velocity along v3, but how to handle reversal?
        # Reversal: If moving away from the center, decrease c.
        # Let's say: we want the hand to "push" the point cloud.
        # Since this is an abstract interaction, we can just use the absolute projection 
        # but how do we decrease?
        # If the user's hand velocity aligns with pushing (say, -Z in camera space), it compresses.
        # Let's map negative Z to positive compression, and positive Z to negative compression.
        # We can just project it on the camera's Z axis for simplicity, OR project onto v3 
        # and use the sign of the projection relative to the camera's forward vector.
        
        # For now, let's keep it simple: project hand velocity on v3.
        # To determine sign (increase/decrease):
        # We will assume that moving the hand "forward" (negative dz in MediaPipe) means compress,
        # and "backward" (positive dz) means decompress.
        # So we align v3 with the Z-axis to check its sign.
        if v3[2] > 0:
            v3 = -v3
            
        # Now v3 points "forward" into the screen.
        # Motion along v3 increases compression.
        delta = np.dot(hand_velocity, v3) * self.config.compression_sensitivity * 10.0
        
        return delta
