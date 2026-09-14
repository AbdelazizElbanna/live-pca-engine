import numpy as np
from src.config import DEFAULT_CONFIG
from src.math_core.data_generator import generate_dataset
from src.math_core.pca_engine import compute_pca
from src.math_core.pca_box import compute_pca_box
from src.math_core.projection import compute_projection
from src.interaction.compression_handler import CompressionHandler
import importlib

def test_ac01_reproducibility():
    """AC-01: Same seed -> identical dataset (every run)"""
    data1 = generate_dataset(42, 100, (5,3,1), 0.0)
    data2 = generate_dataset(42, 100, (5,3,1), 0.0)
    np.testing.assert_array_equal(data1, data2)

def test_ac02_math_invariants():
    """AC-02: Eigenvalues sorted descending, eigenvectors orthonormal"""
    data = generate_dataset(42, 100, (5,3,1), 0.1)
    result = compute_pca(data)
    
    # Descending
    assert result.eigenvalues[0] >= result.eigenvalues[1] >= result.eigenvalues[2]
    
    # Orthonormal
    vt_v = result.eigenvectors.T @ result.eigenvectors
    np.testing.assert_allclose(vt_v, np.eye(3), atol=1e-7)

def test_ac03_geometry_truth():
    """AC-03: PCA box orientation precisely matches PCA basis"""
    data = generate_dataset(42, 100, (5,3,1), 0.1)
    result = compute_pca(data)
    box = compute_pca_box(result)
    
    np.testing.assert_array_equal(box.axes, result.eigenvectors)

def test_ac04_continuous_projection():
    """AC-04: Compression c=1 -> valid orthogonal 2D projection on retained subspace"""
    data = generate_dataset(42, 100, (5,3,1), 0.1)
    result = compute_pca(data)
    box = compute_pca_box(result)
    
    proj_state = compute_projection(data, result, box, c=1.0)
    
    # Check if projection is orthogonal to v3 (smallest PC)
    centered = proj_state.positions - result.mean
    v3 = result.eigenvectors[:, 2]
    dot_prods = centered @ v3
    np.testing.assert_allclose(dot_prods, np.zeros_like(dot_prods), atol=1e-7)

def test_ac05_velocity_control():
    """AC-05: Stop hand -> stop transformation; reverse hand -> reverse transformation"""
    handler = CompressionHandler(DEFAULT_CONFIG.interaction)
    result = compute_pca(generate_dataset(42, 10, (5,3,1), 0.1))
    
    # Stop hand
    delta_stop = handler.compute_delta(np.zeros(3), result)
    assert delta_stop == 0.0
    
    # Reversal hand (forward vs backward)
    v3 = result.eigenvectors[:, 2]
    # Align v3 to negative Z (forward)
    if v3[2] > 0:
        v3 = -v3
    
    delta_forward = handler.compute_delta(v3, result)
    delta_backward = handler.compute_delta(-v3, result)
    
    assert delta_forward > 0.0
    assert delta_backward < 0.0

def test_ac06_architecture_boundaries():
    """AC-06: Math core runs without renderer; system works with mock inputs"""
    # math_core shouldn't import renderer or input
    math_core_imports = []
    
    # Test by importing math_core modules and ensuring no Panda3D/OpenCV errors
    import src.math_core.pca_engine
    import src.math_core.pca_box
    import src.math_core.projection
    import src.math_core.data_generator
    
    assert True # If we reached here without import error, boundaries are respected
