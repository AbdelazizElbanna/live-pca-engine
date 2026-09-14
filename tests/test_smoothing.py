import numpy as np
from src.config import InteractionConfig
from src.input.hand_tracker import HandLandmarks
from src.input.smoothing import LandmarkSmoother

def test_landmark_smoothing():
    config = InteractionConfig(smoothing_alpha=0.5)
    smoother = LandmarkSmoother(config)
    
    # Create two frames of landmarks
    pos1 = np.ones((21, 3))
    
    # Frame 1: all ones
    lm1 = HandLandmarks(
        positions=pos1,
        handedness="Right",
        confidence=0.9
    )
    
    # Frame 2: all twos
    lm2 = HandLandmarks(
        positions=pos1,
        handedness="Right",
        confidence=0.9
    )
    
    # First frame should not be smoothed (no history)
    result1 = smoother.smooth([lm1])
    assert len(result1) == 1
    np.testing.assert_array_equal(result1[0].positions, pos1)
    
    # Move hand
    pos2 = np.ones((21, 3)) * 0.1
    lm2 = HandLandmarks(positions=pos2, handedness="Right", confidence=0.9)
    
    result2 = smoother.smooth([lm2])
    assert len(result2) == 1
    
    # Should be interpolated: 0.5 * 1.0 + 0.5 * 0.1 = 0.55
    expected_pos2 = np.ones((21, 3)) * 0.55
    np.testing.assert_allclose(result2[0].positions, expected_pos2, atol=1e-6)
    
    # No hand should reset the filter
    result_none = smoother.smooth([])
    assert result_none == []
    
    # Next frame should start fresh without history
    lm3 = HandLandmarks(
        positions=np.zeros((21, 3)),
        handedness="Right",
        confidence=0.9
    )
    result3 = smoother.smooth([lm3])
    np.testing.assert_allclose(result3[0].positions, np.zeros((21, 3)))
