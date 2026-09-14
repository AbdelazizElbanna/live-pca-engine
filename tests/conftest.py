"""
Shared pytest fixtures for the test suite.

Provides deterministic sample data, PCA results, and mock objects
that are reused across multiple test files.
"""

import numpy as np
import pytest

from src.config import DataConfig, AppConfig


@pytest.fixture
def default_config():
    """Default application configuration."""
    return AppConfig()


@pytest.fixture
def data_config():
    """Default data generation configuration."""
    return DataConfig()


@pytest.fixture
def small_data_config():
    """Small dataset configuration for fast tests."""
    return DataConfig(seed=42, n_points=50, axis_lengths=(5.0, 3.0, 1.0), noise_std=0.3)


@pytest.fixture
def deterministic_rng():
    """A deterministic random number generator for test reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_dataset():
    """A small deterministic 3D dataset for testing.

    This fixture generates a simple elongated ellipsoid dataset
    directly (without depending on the data_generator module)
    so it can be used to test the PCA engine independently.
    """
    rng = np.random.default_rng(42)
    n = 100
    # Generate points with known variance structure
    raw = rng.standard_normal((n, 3))
    # Scale axes: x has most variance, z has least
    raw[:, 0] *= 5.0
    raw[:, 1] *= 3.0
    raw[:, 2] *= 1.0
    # Add a rotation to create correlation
    theta = np.pi / 6  # 30 degrees
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    data = raw @ R.T
    # Add small noise
    data += rng.normal(0, 0.3, data.shape)
    return data


@pytest.fixture
def axis_aligned_dataset():
    """A dataset aligned with coordinate axes (no rotation).

    Useful for testing where expected PCA results are trivially known:
    - PC1 should align with x-axis (highest variance)
    - PC2 should align with y-axis
    - PC3 should align with z-axis
    """
    rng = np.random.default_rng(123)
    n = 200
    data = np.zeros((n, 3))
    data[:, 0] = rng.normal(0, 5.0, n)
    data[:, 1] = rng.normal(0, 3.0, n)
    data[:, 2] = rng.normal(0, 1.0, n)
    return data
