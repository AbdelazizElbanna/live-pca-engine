"""
Aesthetic 2D Hand Skeleton Visualizer for OpenCV Camera Window.

Renders modern, clean, scientific hand tracking landmarks and delicate bone
connections onto the camera frame with high visual fidelity and zero performance overhead.
"""

import cv2
import numpy as np
from typing import List, Any, Sequence

# Standard MediaPipe 21-hand landmark connections
# Matches natural anatomical hand structure cleanly without clutter
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger & palm base
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

FINGERTIP_INDICES = {4, 8, 12, 16, 20}

# Color palette: curated turquoise / cyan / luminous ivory (BGR format)
COLOR_BONE = (220, 215, 45)       # Cyan / turquoise bone links
COLOR_HALO = (235, 205, 40)       # Soft glowing turquoise halo ring
COLOR_CORE = (255, 255, 220)      # Crisp bright ivory / cyan core dot
COLOR_TIP_OUTER = (240, 230, 60)  # Fingertip glowing ring
COLOR_TIP_CORE = (255, 255, 255)  # Pure luminous white fingertip core


def draw_aesthetic_hand_landmarks(
    frame: np.ndarray,
    landmarks_list: Sequence[Any],
    is_square_coords: bool = True
) -> np.ndarray:
    """
    Renders aesthetic hand landmarks and delicate skeleton connections onto the camera frame.
    
    Args:
        frame: OpenCV BGR image (H, W, 3).
        landmarks_list: List of HandLandmarks objects or landmark position arrays.
        is_square_coords: True if landmarks are normalized in 1:1 square space [0, 1] 
                          due to aspect-ratio padding in the tracker.
                          
    Returns:
        The frame with hand landmarks drawn.
    """
    if not landmarks_list or frame is None:
        return frame
        
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame
        
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_left = (size - w) // 2
    
    # 1. Overlay for anti-aliased, semi-translucent glowing skeleton lines and halos
    overlay = frame.copy()
    
    for hand in landmarks_list:
        positions = hand.positions if hasattr(hand, 'positions') else hand
        if positions is None or len(positions) < 21:
            continue
            
        # Convert normalized coordinates to camera frame pixel coordinates
        pixel_pts = []
        for lm in positions:
            if is_square_coords:
                px = int(round(lm[0] * size - pad_left))
                py = int(round(lm[1] * size - pad_top))
            else:
                px = int(round(lm[0] * w))
                py = int(round(lm[1] * h))
            pixel_pts.append((px, py))
            
        # Draw delicate anti-aliased bone connections
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            pt1 = pixel_pts[p1_idx]
            pt2 = pixel_pts[p2_idx]
            cv2.line(overlay, pt1, pt2, COLOR_BONE, 1, lineType=cv2.LINE_AA)
            
        # Draw soft halo rings for joints
        for idx, pt in enumerate(pixel_pts):
            if idx in FINGERTIP_INDICES:
                cv2.circle(overlay, pt, 6, COLOR_TIP_OUTER, 1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(overlay, pt, 4, COLOR_HALO, 1, lineType=cv2.LINE_AA)
                
    # Blend overlay with original frame (70% overlay, 30% original) for subtle glowing lines
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    
    # 2. Draw solid, sharp core dots directly on top for crisp visual precision
    for hand in landmarks_list:
        positions = hand.positions if hasattr(hand, 'positions') else hand
        if positions is None or len(positions) < 21:
            continue
            
        for idx, lm in enumerate(positions):
            if is_square_coords:
                px = int(round(lm[0] * size - pad_left))
                py = int(round(lm[1] * size - pad_top))
            else:
                px = int(round(lm[0] * w))
                py = int(round(lm[1] * h))
            pt = (px, py)
            
            if idx in FINGERTIP_INDICES:
                cv2.circle(frame, pt, 4, COLOR_TIP_CORE, -1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, COLOR_CORE, -1, lineType=cv2.LINE_AA)
                
    return frame
