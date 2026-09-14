import numpy as np
from dataclasses import dataclass
from src.math_core.pca_engine import PCAResult
from src.math_core.pca_box import PCABox

@dataclass
class ProjectionState:
    """Stores the state of the continuous projection at a specific compression factor c."""
    positions: np.ndarray      # Shape (N, 3), interpolated positions
    target_projections: np.ndarray # Shape (N, 3), fully projected positions on 2D plane
    box_extent_3: float        # The extent of the 3rd box dimension at state c
    pc3_alpha: float           # Visibility alpha of the 3rd PC vector at state c


def compute_projection(
    data: np.ndarray, 
    pca_result: PCAResult, 
    pca_box: PCABox, 
    c: float
) -> ProjectionState:
    """
    Compute parameterized point projection for compression state c ∈ [0,1].
    
    Args:
        data: Original points (N, 3)
        pca_result: PCA result containing mean and eigenvectors
        pca_box: PCA box containing extents
        c: Compression state from 0.0 (original) to 1.0 (projected to 2D)
        
    Returns:
        ProjectionState containing positions and visual parameters.
    """
    c = np.clip(c, 0.0, 1.0)
    
    # 1. Project points onto the plane spanned by v1, v2 (discarding v3)
    # The smallest PC is v3, which is the last column (index 2) of eigenvectors
    # p_i = μ + z_i1 * v1 + z_i2 * v2
    
    # Get projections z = X_c @ V
    z = pca_result.centered_data @ pca_result.eigenvectors
    
    # Zero out the 3rd component to project onto the v1-v2 plane
    z_proj = z.copy()
    z_proj[:, 2] = 0.0
    
    # Target projections in world space
    target_projections = pca_result.mean + z_proj @ pca_result.eigenvectors.T
    
    # 2. Interpolate positions: pos(c) = (1-c) * data + c * target_projections
    positions = (1.0 - c) * data + c * target_projections
    
    # 3. PCA box extent for dimension 3 at state c
    box_extent_3 = pca_box.extents[2] * (1.0 - c)
    
    # 4. PC3 vector visibility alpha
    pc3_alpha = 1.0 - c
    
    return ProjectionState(
        positions=positions,
        target_projections=target_projections,
        box_extent_3=box_extent_3,
        pc3_alpha=pc3_alpha
    )
