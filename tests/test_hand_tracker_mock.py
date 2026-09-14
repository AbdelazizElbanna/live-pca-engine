import numpy as np
from src.input.hand_tracker import HandTracker, HandLandmarks

def test_hand_tracker_no_hand():
    # Use a black image which definitely has no hand
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    with HandTracker() as tracker:
        result = tracker.process_frame(black_frame)
        assert result == []

# Note: We can't easily test positive detection without a real image of a hand.
# For unit tests, testing the graceful degradation (None) and initialization is sufficient.
# We could load a test image if one was provided in tests/fixtures/.
