import numpy as np

def generate_dataset(
    seed: int, 
    n_points: int, 
    axis_lengths: tuple[float, float, float], 
    noise_std: float = 0.0
) -> np.ndarray:
    """
    Generate synthetic 3D data points representing an elongated ellipsoid with correlated dimensions.
    
    Args:
        seed: Random seed for determinism.
        n_points: Number of points to generate.
        axis_lengths: Semi-axis lengths (x, y, z) determining variance structure.
        noise_std: Standard deviation of Gaussian noise added to the data.
        
    Returns:
        np.ndarray: Generated dataset of shape (n_points, 3).
    """
    rng = np.random.default_rng(seed)
    
    # Generate Gaussian data
    data = rng.standard_normal((n_points, 3))
    
    # Scale by axis lengths
    data = data * np.array(axis_lengths)
    
    # Apply a fixed rotation to introduce correlation between dimensions
    theta_x = np.pi / 4  # 45 deg
    theta_y = np.pi / 3  # 60 deg
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    R = Ry @ Rx
    data = data @ R.T
    
    # Apply isotropic Gaussian noise
    if noise_std > 0:
        noise = rng.normal(0, noise_std, (n_points, 3))
        data += noise
        
    return data
