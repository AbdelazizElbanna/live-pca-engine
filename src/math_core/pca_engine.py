import numpy as np
from dataclasses import dataclass

@dataclass
class PCAResult:
    """Stores the result of a Principal Component Analysis."""
    mean: np.ndarray          # Shape (3,)
    eigenvalues: np.ndarray   # Shape (3,), sorted descending
    eigenvectors: np.ndarray  # Shape (3, 3), columns are PCs
    centered_data: np.ndarray # Shape (N, 3)


def compute_pca(data: np.ndarray) -> PCAResult:
    """
    Compute Principal Component Analysis on a 3D dataset.
    
    Args:
        data: Input dataset of shape (N, 3)
        
    Returns:
        PCAResult containing mean, eigenvalues, eigenvectors, and centered data.
    """
    n_samples = data.shape[0]
    
    # 1. Mean centering
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # 2. Covariance matrix
    # Using formula C = (1/(N-1)) * X_c^T X_c
    cov_matrix = (centered_data.T @ centered_data) / (n_samples - 1)
    
    # 3. Eigendecomposition (use eigh since covariance is symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 4. Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 5. Enforce deterministic sign convention:
    # Make the largest absolute component of each eigenvector positive
    for i in range(eigenvectors.shape[1]):
        max_idx = np.argmax(np.abs(eigenvectors[:, i]))
        if eigenvectors[max_idx, i] < 0:
            eigenvectors[:, i] *= -1
            
    return PCAResult(
        mean=mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        centered_data=centered_data
    )
