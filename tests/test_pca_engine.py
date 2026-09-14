import numpy as np
from sklearn.decomposition import PCA
from src.math_core.pca_engine import compute_pca
from src.math_core.data_generator import generate_dataset

def test_pca_engine_properties():
    data = generate_dataset(42, 150, (5.0, 3.0, 1.0), 0.1)
    result = compute_pca(data)
    
    # Check shapes
    assert result.mean.shape == (3,)
    assert result.eigenvalues.shape == (3,)
    assert result.eigenvectors.shape == (3, 3)
    assert result.centered_data.shape == data.shape
    
    # Check eigenvalues are sorted descending
    assert result.eigenvalues[0] >= result.eigenvalues[1] >= result.eigenvalues[2]
    
    # Check eigenvectors are orthonormal: V^T V = I
    vt_v = result.eigenvectors.T @ result.eigenvectors
    np.testing.assert_allclose(vt_v, np.eye(3), atol=1e-7)

def test_pca_engine_vs_sklearn():
    data = generate_dataset(123, 200, (6.0, 4.0, 2.0), 0.2)
    
    # Our PCA
    result = compute_pca(data)
    
    # Sklearn PCA
    sklearn_pca = PCA(n_components=3)
    sklearn_pca.fit(data)
    
    # Compare eigenvalues (sklearn's explained_variance_)
    np.testing.assert_allclose(result.eigenvalues, sklearn_pca.explained_variance_, rtol=1e-5)
    
    # Compare eigenvectors (sklearn's components_ are rows, ours are columns)
    # Sklearn might use a different sign convention, so we check absolute values or dot products
    dot_prods = np.abs(np.diag(result.eigenvectors.T @ sklearn_pca.components_.T))
    np.testing.assert_allclose(dot_prods, np.ones(3), atol=1e-5)

def test_pca_engine_degenerate_data():
    # 2D data embedded in 3D
    np.random.seed(42)
    data = np.random.randn(100, 3)
    data[:, 2] = 0  # Zero variance in Z
    
    result = compute_pca(data)
    
    # Third eigenvalue should be very close to zero
    assert result.eigenvalues[2] < 1e-10
    
    # The third eigenvector should be along the Z axis (or close to it depending on convention)
    # Actually, it must be the Z axis [0, 0, 1] or [0, 0, -1] because X and Y have variance.
    # Our sign convention enforces positive max element, so it should be exactly [0, 0, 1]
    np.testing.assert_allclose(result.eigenvectors[:, 2], np.array([0, 0, 1]), atol=1e-7)
