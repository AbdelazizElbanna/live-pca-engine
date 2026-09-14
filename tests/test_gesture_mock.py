import numpy as np
from src.config import InteractionConfig
from src.input.hand_tracker import HandLandmarks
from src.interaction.gesture_recognizer import GestureRecognizer, GestureType

def test_gesture_recognizer_active():
    """Test general active gesture logic (1 hand)."""
    config = InteractionConfig()
    recognizer = GestureRecognizer(config)
    
    # Mock landmarks in mirrored camera coordinates
    pos1 = np.zeros((21, 3))
    pos1[5] = [-0.1, -0.1, 0.0]  # Index on left (Back of right hand faces screen in mirror)
    pos1[17] = [0.1, -0.1, 0.0]  # Pinky on right
    lm1 = HandLandmarks(positions=pos1, handedness="Right", confidence=0.9)
    
    pos2 = pos1.copy()
    pos2[9] = [0.1, -0.2, 0.1] # move center up and right
    lm2 = HandLandmarks(positions=pos2, handedness="Right", confidence=0.9)
    
    # Run 6 times to bypass the frames_visible initialization delay
    for _ in range(6):
        intent1 = recognizer.recognize([lm1])
        
    # Now it should be ACTIVE on next frame
    intent2 = recognizer.recognize([lm2])
    assert intent2.gesture_type == GestureType.ACTIVE
    assert intent2.rotation_delta is not None
    assert len(intent2.rotation_delta) == 2

def test_gesture_recognizer_zoom():
    """Test zoom gesture logic (1 hand depth)."""
    config = InteractionConfig()
    recognizer = GestureRecognizer(config)
    
    # Mock one hand
    pos1 = np.zeros((21, 3))
    pos1[0] = [0.0, 0.0, 0.0]
    pos1[5] = [0.1, -0.1, 0.0]
    pos1[17] = [-0.1, -0.1, 0.0]
    pos1[9] = [0.1, 0.1, 0.0] # Distance is approx 0.14
    lm1 = [HandLandmarks(positions=pos1.copy(), handedness="Right", confidence=0.9)]
    
    # Initialize 6 frames
    for _ in range(6):
        recognizer.recognize(lm1)
    
    # Third frame, make hand even larger
    pos3 = pos1.copy()
    pos3[5] = [0.2, -0.2, 0.0]
    pos3[9] = [0.3, 0.3, 0.0] 
    pos3[13] = [0.3, 0.3, 0.0] 
    pos3[17] = [-0.2, -0.2, 0.0] 
    lm3 = [HandLandmarks(positions=pos3.copy(), handedness="Right", confidence=0.9)]
    
    intent3 = recognizer.recognize(lm3)
    
    assert intent3.gesture_type == GestureType.ACTIVE
    assert intent3.zoom_delta > 0 # Scaled up

def test_gesture_recognizer_compress():
    """Test compress gesture logic (1 hand depth)."""
    config = InteractionConfig()
    recognizer = GestureRecognizer(config)
    
    pos1 = np.zeros((21, 3))
    pos1[0] = [0.0, 0.0, 0.0]
    pos1[5] = [0.1, -0.1, 0.0]
    pos1[17] = [-0.1, -0.1, 0.0]
    pos1[9] = [0.1, 0.1, 0.0]
    lm1 = [HandLandmarks(positions=pos1.copy(), handedness="Right", confidence=0.9)]
    
    # Initialize 6 frames
    for _ in range(6):
        recognizer.recognize(lm1)
    
    pos3 = pos1.copy()
    pos3[5] = [0.2, -0.2, 0.0]
    pos3[9] = [0.3, 0.3, 0.0] 
    pos3[13] = [0.3, 0.3, 0.0] 
    pos3[17] = [-0.2, -0.2, 0.0] 
    lm3 = [HandLandmarks(positions=pos3.copy(), handedness="Right", confidence=0.9)]
    
    intent3 = recognizer.recognize(lm3)
    
    assert intent3.gesture_type == GestureType.ACTIVE
    assert intent3.compression_delta > 0
