import numpy as np
from dataclasses import dataclass
from src.math_core.pca_engine import PCAResult

@dataclass
class PCABox:
    """Stores the mathematically-derived PCA bounding box properties."""
    center: np.ndarray             # Shape (3,) world-space center
    extents: np.ndarray            # Shape (3,) extents along each PC axis
    axes: np.ndarray               # Shape (3, 3) orientation axes (eigenvectors)
    min_projections: np.ndarray    # Shape (3,) min projection along each axis
    max_projections: np.ndarray    # Shape (3,) max projection along each axis


def compute_pca_box(pca_result: PCAResult) -> PCABox:
    """
    Compute the bounding box of the dataset aligned with the principal components.
    
    Args:
        pca_result: PCAResult from compute_pca
        
    Returns:
        PCABox containing center, extents, and orientation.
    """
    # 1. Project each centered point onto each PC axis: Z = X_c @ V
    # X_c is (N, 3), V is (3, 3). So Z is (N, 3)
    projections = pca_result.centered_data @ pca_result.eigenvectors
    
    # 2. Min and max projections for each axis
    min_projs = np.min(projections, axis=0)
    max_projs = np.max(projections, axis=0)
    
    # 3. Box extent on each axis
    extents = max_projs - min_projs
    
    # 4. Box center in PCA space
    center_pca = (min_projs + max_projs) / 2.0
    
    # 5. Transform box center back to world space
    # The world space center is μ + V @ center_pca
    center_world = pca_result.mean + pca_result.eigenvectors @ center_pca
    
    return PCABox(
        center=center_world,
        extents=extents,
        axes=pca_result.eigenvectors,
        min_projections=min_projs,
        max_projections=max_projs
    )
