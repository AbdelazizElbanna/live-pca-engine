import numpy as np
from src.math_core.data_generator import generate_dataset
from src.math_core.pca_engine import compute_pca
from src.math_core.pca_box import compute_pca_box
from src.math_core.projection import compute_projection

def test_projection_continuity():
    data = generate_dataset(42, 100, (5.0, 3.0, 1.0), 0.1)
    pca_result = compute_pca(data)
    pca_box = compute_pca_box(pca_result)
    
    # Test c=0 (original positions)
    state_0 = compute_projection(data, pca_result, pca_box, 0.0)
    np.testing.assert_allclose(state_0.positions, data, atol=1e-7)
    assert np.isclose(state_0.box_extent_3, pca_box.extents[2])
    assert np.isclose(state_0.pc3_alpha, 1.0)
    
    # Test c=1 (fully projected)
    state_1 = compute_projection(data, pca_result, pca_box, 1.0)
    assert np.isclose(state_1.box_extent_3, 0.0, atol=1e-7)
    assert np.isclose(state_1.pc3_alpha, 0.0, atol=1e-7)
    
    # At c=1, orthogonality: the vector from point to target should be parallel to v3
    # Wait, the vectors from projected points to mean should have 0 dot product with v3
    centered_proj = state_1.positions - pca_result.mean
    dot_products = centered_proj @ pca_result.eigenvectors[:, 2]
    np.testing.assert_allclose(dot_products, np.zeros_like(dot_products), atol=1e-7)
    
    # Test intermediate c=0.5
    state_half = compute_projection(data, pca_result, pca_box, 0.5)
    expected_half_pos = (data + state_1.target_projections) / 2.0
    np.testing.assert_allclose(state_half.positions, expected_half_pos, atol=1e-7)
    assert np.isclose(state_half.box_extent_3, pca_box.extents[2] * 0.5)
    assert np.isclose(state_half.pc3_alpha, 0.5)
