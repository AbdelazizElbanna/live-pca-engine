import numpy as np
from src.math_core.data_generator import generate_dataset
from src.config import DEFAULT_CONFIG

def test_generate_dataset_determinism():
    config = DEFAULT_CONFIG.data
    data1 = generate_dataset(config.seed, config.n_points, config.axis_lengths, config.noise_std)
    data2 = generate_dataset(config.seed, config.n_points, config.axis_lengths, config.noise_std)
    
    np.testing.assert_array_equal(data1, data2)
    assert data1.shape == (config.n_points, 3)

def test_generate_dataset_shape():
    data = generate_dataset(123, 150, (5.0, 3.0, 1.0), 0.1)
    assert data.shape == (150, 3)
    assert data.dtype == float

def test_generate_dataset_variance_structure():
    # Verify that the generated data has different variances along its principal axes
    data = generate_dataset(42, 10000, (10.0, 5.0, 1.0), 0.0)
    
    # Center the data
    mean = np.mean(data, axis=0)
    centered = data - mean
    
    # Compute covariance
    cov = np.cov(centered, rowvar=False)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    
    # Expected eigenvalues are roughly square of axis lengths (since it's std normal * axis_length)
    # The eigenvalues should be distinct if axis lengths are distinct
    # eigvalsh returns them in ascending order
    assert eigenvalues[2] > eigenvalues[1] > eigenvalues[0]
