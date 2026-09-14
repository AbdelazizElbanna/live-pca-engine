import numpy as np
from src.config import DEFAULT_CONFIG
from src.interaction.state_manager import StateManager
from src.interaction.gesture_recognizer import InteractionIntent, GestureType

def test_state_manager_initialization():
    from src.interaction.state_manager import InteractionMode
    manager = StateManager(DEFAULT_CONFIG)
    state = manager.get_scene_state()
    
    assert manager.current_mode == InteractionMode.IDLE
    assert state.camera_rotation == (0.0, 0.0)
    assert np.allclose(state.object_rotation_mat, np.eye(3))
    assert state.camera_zoom == 1.0
    assert state.compression_c == 0.0
    assert state.dataset.shape == (DEFAULT_CONFIG.data.n_points, 3)
    assert state.pca_result is not None
    assert state.pca_box is not None

def test_state_manager_idle():
    from src.interaction.state_manager import InteractionMode
    manager = StateManager(DEFAULT_CONFIG)
    assert manager.current_mode == InteractionMode.IDLE
    
    # Rotation intent must have no effect
    intent_rot = InteractionIntent(gesture_type=GestureType.ACTIVE, rotation_delta=(90.0, 45.0))
    manager.apply_intent(intent_rot)
    manager.update(1.0/60.0)
    assert np.allclose(manager.get_scene_state().object_rotation_mat, np.eye(3))
    
    # Zoom intent must have no effect
    intent_zoom = InteractionIntent(gesture_type=GestureType.ACTIVE, zoom_delta=-0.5)
    manager.apply_intent(intent_zoom)
    manager.update(1.0/60.0)
    assert manager.get_scene_state().camera_zoom == 1.0
    
    # Compress intent must have no effect
    intent_comp = InteractionIntent(gesture_type=GestureType.ACTIVE, compression_delta=0.5)
    manager.apply_intent(intent_comp)
    manager.update(1.0/60.0)
    assert manager.get_scene_state().compression_c == 0.0

def test_state_manager_rotate():
    from src.interaction.state_manager import InteractionMode
    manager = StateManager(DEFAULT_CONFIG)
    manager.set_mode(InteractionMode.ROTATION)
    
    intent = InteractionIntent(gesture_type=GestureType.ACTIVE, rotation_delta=(90.0, 45.0))
    manager.apply_intent(intent)
    
    # Update for 1 frame
    manager.update(1.0/60.0)
    
    state = manager.get_scene_state()
    # It adds momentum now, and updates the matrix
    # Trace of the matrix should not be 3.0 (which is Identity)
    assert not np.allclose(state.object_rotation_mat, np.eye(3))

def test_state_manager_zoom():
    manager = StateManager(DEFAULT_CONFIG)
    from src.interaction.state_manager import InteractionMode
    manager.set_mode(InteractionMode.ZOOM)
    
    intent = InteractionIntent(gesture_type=GestureType.ACTIVE, zoom_delta=-0.5)
    manager.apply_intent(intent)
    manager.update(1.0/60.0)
    
    state = manager.get_scene_state()
    assert state.camera_zoom == 0.5
    
    # Test clamping
    intent2 = InteractionIntent(gesture_type=GestureType.ACTIVE, zoom_delta=-1.0)
    manager.apply_intent(intent2)
    manager.update(1.0/60.0)
    state = manager.get_scene_state()
    assert state.camera_zoom >= 0.1 # Should be clamped

def test_state_manager_compress():
    manager = StateManager(DEFAULT_CONFIG)
    from src.interaction.state_manager import InteractionMode
    manager.set_mode(InteractionMode.COMPRESSION)
    
    # Send a positive compression velocity (like hand moving)
    intent = InteractionIntent(gesture_type=GestureType.ACTIVE, compression_delta=0.5)
    manager.apply_intent(intent)
    manager.update(1.0/60.0)
    
    state = manager.get_scene_state()
    # It adds velocity * dt, but velocity is lerped (0.5 * 0.15 = 0.075)
    assert np.allclose(state.compression_c, 0.075)
    
    # Test clamping upper
    intent_clamp = InteractionIntent(gesture_type=GestureType.ACTIVE, compression_delta=1.0)
    manager.apply_intent(intent_clamp)
    for _ in range(30):
        manager.update(1.0/60.0)
    
    state = manager.get_scene_state()
    assert state.compression_c == 1.0 # Clamped at 1.0
    
    # Test clamping lower (reversal)
    intent_reverse = InteractionIntent(gesture_type=GestureType.ACTIVE, compression_delta=-1.5)
    manager.apply_intent(intent_reverse)
    
    for _ in range(30):
        manager.update(1.0/60.0)
        
    state = manager.get_scene_state()
    assert state.compression_c == 0.0 # Clamped at 0.0
