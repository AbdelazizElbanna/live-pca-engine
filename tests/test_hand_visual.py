import numpy as np
from src.compositor.hand_visual import draw_aesthetic_hand_landmarks
from src.input.hand_tracker import HandLandmarks

def test_draw_aesthetic_hand_landmarks_valid():
    """Verify landmark drawing runs without error and preserves frame shape."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    positions = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    hand = HandLandmarks(positions=positions, handedness="Right", confidence=0.99)
    
    result = draw_aesthetic_hand_landmarks(frame, [hand], is_square_coords=True)
    assert result is not None
    assert result.shape == (480, 640, 3)
    # The output frame should no longer be completely black
    assert np.any(result > 0)

def test_draw_aesthetic_hand_landmarks_empty_or_none():
    """Verify drawing functions handle empty landmarks or None gracefully."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # None / Empty landmarks
    res1 = draw_aesthetic_hand_landmarks(frame, [])
    assert np.array_equal(res1, frame)
    
    res2 = draw_aesthetic_hand_landmarks(None, [])
    assert res2 is None
