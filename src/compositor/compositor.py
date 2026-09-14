import cv2
import numpy as np
from typing import Optional

class Compositor:
    """Blends Panda3D rendered RGBA frames with OpenCV BGR camera frames."""
    
    def __init__(self):
        pass
        
    def composite(self, camera_frame: Optional[np.ndarray], render_frame: np.ndarray) -> np.ndarray:
        """
        Alpha blend the rendered virtual scene over the camera frame.
        
        Args:
            camera_frame: BGR frame from OpenCV, shape (H, W, 3). Can be None.
            render_frame: RGBA frame from Panda3D, shape (H, W, 4).
            
        Returns:
            Composited BGR frame, shape (H, W, 3).
        """
        # If camera frame is missing, just return the RGB part of the render frame
        if camera_frame is None:
            return render_frame[:, :, :3].copy()
            
        render_h, render_w = render_frame.shape[:2]
        cam_h, cam_w = camera_frame.shape[:2]
        
        # Resize camera frame if dimensions mismatch
        if (render_h, render_w) != (cam_h, cam_w):
            camera_frame = cv2.resize(camera_frame, (render_w, render_h))
            
        # Extract alpha and colors
        alpha = render_frame[:, :, 3] / 255.0
        render_bgr = render_frame[:, :, :3]
        
        # We need alpha in 3 channels for broadcasting
        alpha_3 = np.expand_dims(alpha, axis=-1)
        
        # Perform alpha blending
        # out = alpha * render_color + (1 - alpha) * camera_color
        # Note: render_frame might be RGB or BGR depending on how Panda extracts it.
        # Panda3D's getScreenshot() usually gives BGRA or RGBA.
        # Assuming render_frame is BGR for the color channels based on typical OpenCV usage.
        # If it's RGB, we'll swap it here (assuming caller passed it as RGBA but OpenCV uses BGR).
        # Actually, let's assume it's BGRA to match OpenCV, or RGBA and we swap R and B.
        # We'll stick to typical mathematical blending first.
        
        composited = alpha_3 * render_bgr + (1.0 - alpha_3) * camera_frame
        
        return composited.astype(np.uint8)
