import numpy as np
from src.input.camera import CameraCapture
from src.config import CameraConfig

def test_camera_mock_injection():
    config = CameraConfig(width=640, height=480, fps=30)
    camera = CameraCapture(config)
    
    # Create a dummy green frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_frame[:, :, 1] = 255  # Green channel
    
    # Inject it
    camera.inject_mock_frame(mock_frame)
    
    # Starting should succeed with mock frame
    assert camera.start() is True
    
    # Get frame should return the injected frame
    frame = camera.get_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    np.testing.assert_array_equal(frame, mock_frame)
    
    camera.stop()

def test_camera_uninitialized():
    config = CameraConfig()
    camera = CameraCapture(config)
    
    # Getting a frame before starting should return None
    assert camera.get_frame() is None

def test_camera_context_manager():
    config = CameraConfig()
    mock_frame = np.ones((100, 100, 3), dtype=np.uint8)
    
    with CameraCapture(config) as camera:
        camera.inject_mock_frame(mock_frame)
        assert camera.start() is True
        assert camera.get_frame() is not None
        
    # After exiting the context, getting frame should return None (is_open = False)
    assert camera.get_frame() is None
