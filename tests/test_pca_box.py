import numpy as np
from src.math_core.pca_engine import compute_pca
from src.math_core.pca_box import compute_pca_box

def test_pca_box_computation():
    # Construct a simple axis-aligned dataset
    # Points at extremities so we know the bounding box exactly
    data = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -3.0]
    ])
    
    # Shift the data to test centering
    shift = np.array([10.0, 20.0, 30.0])
    data = data + shift
    
    # Rotate the data to test orientation
    theta = np.pi / 6
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    data = data @ R.T
    
    pca_result = compute_pca(data)
    box = compute_pca_box(pca_result)
    
    # The extents should be 6, 4, 2 since the max distances from origin are 3, 2, 1
    # PCA sorts by variance (which correlates with max distance here). 
    # Extent for Z (originally Z, length 6) should be first
    # Extent for Y (originally Y, length 4) should be second
    # Extent for X (originally X, length 2) should be third
    expected_extents = np.array([6.0, 4.0, 2.0])
    np.testing.assert_allclose(box.extents, expected_extents, atol=1e-7)
    
    # The center in world space should be exactly the mean of the dataset
    # Since the points are perfectly symmetric around the mean
    expected_center = np.mean(data, axis=0)
    np.testing.assert_allclose(box.center, expected_center, atol=1e-7)
    
    # The axes should be exactly the eigenvectors
    np.testing.assert_array_equal(box.axes, pca_result.eigenvectors)
