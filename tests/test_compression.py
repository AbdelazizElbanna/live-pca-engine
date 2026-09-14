import numpy as np
from src.config import InteractionConfig
from src.math_core.pca_engine import PCAResult
from src.interaction.compression_handler import CompressionHandler

def test_compression_handler():
    config = InteractionConfig(compression_sensitivity=1.0)
    handler = CompressionHandler(config)
    
    # Mock PCA result where smallest PC is Z-axis
    mean = np.zeros(3)
    eigenvalues = np.array([3.0, 2.0, 1.0])
    eigenvectors = np.eye(3) # v3 is [0, 0, 1]
    
    pca_result = PCAResult(
        mean=mean,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        centered_data=np.zeros((10, 3))
    )
    
    # Moving hand forward (-Z) should increase compression
    vel_forward = np.array([0.0, 0.0, -0.1])
    delta = handler.compute_delta(vel_forward, pca_result)
    assert delta > 0.0
    
    # Hand stops
    vel_stop = np.zeros(3)
    delta_stop = handler.compute_delta(vel_stop, pca_result)
    assert delta_stop == 0.0
    
    # Moving hand backward (+Z) should decrease compression
    vel_backward = np.array([0.0, 0.0, 0.1])
    delta_backward = handler.compute_delta(vel_backward, pca_result)
    assert delta_backward < 0.0
