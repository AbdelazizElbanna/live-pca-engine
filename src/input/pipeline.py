import threading
import time
import copy
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from src.input.camera import CameraCapture
from src.input.hand_tracker import HandTracker, HandLandmarks
from src.config import CameraConfig

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InputSnapshot:
    """
    Immutable snapshot of the latest camera frame and tracking data.
    Used to safely pass data from the input thread to the rendering thread.
    """
    frame: Optional[np.ndarray]
    landmarks: Optional[List[HandLandmarks]] # Allow multiple hands
    timestamp: float

class InputPipeline:
    """
    Manages the camera and MediaPipe in a dedicated background thread.
    Ensures that the rendering loop is never blocked by camera latency.
    """
    
    def __init__(self, camera_config: CameraConfig, max_hands: int = 2):
        self.camera_config = camera_config
        self.max_hands = max_hands
        
        self.camera: Optional[CameraCapture] = None
        self.tracker: Optional[HandTracker] = None
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        
        # UI State for OpenCV Window
        self.ui_state = {
            "text": "|| FROZEN",
            "badge_col": (0, 195, 255),
            "border_col": (0, 160, 220),
            "title": "Hand Tracking Camera [FROZEN] - Space: Resume | 1:Rot | 2:Zoom | 3:Comp"
        }
        self._ui_lock = threading.Lock()
        self._last_key = -1
        
        # Initial empty snapshot
        self._latest_snapshot = InputSnapshot(frame=None, landmarks=None, timestamp=time.perf_counter())
        
    def start(self):
        """Start the background input thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="InputPipelineThread")
        self._thread.start()
        
    def stop(self):
        """Signal the thread to stop and wait for it."""
        self._stop_event.set()
        if self.camera is not None:
            try:
                self.camera.stop()
            except:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            
    def get_latest_snapshot(self) -> InputSnapshot:
        """
        Get the latest immutable input snapshot. Non-blocking.
        Returns immediately, suitable for the main render loop.
        """
        with self._snapshot_lock:
            return self._latest_snapshot
            
    def set_ui_state(self, text: str, badge_col: tuple, border_col: tuple, title: str):
        """Update the badge and title for the OpenCV window."""
        with self._ui_lock:
            self.ui_state["text"] = text
            self.ui_state["badge_col"] = badge_col
            self.ui_state["border_col"] = border_col
            self.ui_state["title"] = title
            
    def get_last_key(self) -> int:
        """Get and clear the last pressed key from OpenCV window."""
        with self._ui_lock:
            k = self._last_key
            self._last_key = -1
            return k
            
    def _run_loop(self):
        """The main loop for the background thread."""
        logger.info("Initializing input pipeline in background thread...")
        
        # Initialize resources in the background so app startup is instant
        self.camera = CameraCapture(self.camera_config)
        # We explicitly request 2 hands now
        self.tracker = HandTracker(max_num_hands=self.max_hands)
        
        if not self.camera.start():
            logger.warning("Camera failed to start in InputPipeline.")
            # We continue anyway, frame will be None
            
        logger.info("Input pipeline initialized.")
        
        while not self._stop_event.is_set():
            try:
                loop_start = time.perf_counter()
                
                # 1. Capture Frame (blocking, depends on camera FPS)
                frame = self.camera.get_frame()
                
                # 2. Run Tracking
                landmarks = None
                if frame is not None and getattr(frame, 'size', 0) > 0 and frame.shape[0] > 0 and frame.shape[1] > 0:
                    res = self.tracker.process_frame(frame)
                    if res is not None:
                        if isinstance(res, list):
                            landmarks = res
                        else:
                            landmarks = [res]
                            
                # 3. Create Immutable Snapshot
                frame_copy = frame.copy() if frame is not None else None
                snapshot = InputSnapshot(
                    frame=frame_copy,
                    landmarks=landmarks,
                    timestamp=time.perf_counter()
                )
                
                # 4. Atomic Swap
                with self._snapshot_lock:
                    self._latest_snapshot = snapshot
                    
                # 5. Display OpenCV Window (Isolating scaling lag from main thread)
                if frame is not None:
                    import cv2
                    from src.compositor.hand_visual import draw_aesthetic_hand_landmarks
                    
                    if not hasattr(self, 'cv_window_created'):
                        cv2.namedWindow("Hand Tracking Camera", cv2.WINDOW_NORMAL)
                        self.cv_window_created = True
                        self._last_title = ""
                        
                    display_frame = frame.copy()
                    if landmarks is not None:
                        draw_aesthetic_hand_landmarks(display_frame, landmarks, is_square_coords=True)
                        
                    with self._ui_lock:
                        b_text = self.ui_state["text"]
                        b_col = self.ui_state["badge_col"]
                        br_col = self.ui_state["border_col"]
                        title = self.ui_state["title"]
                        
                    if title != self._last_title:
                        try:
                            cv2.setWindowTitle("Hand Tracking Camera", title)
                            self._last_title = title
                        except:
                            pass

                    # Draw Badge
                    font = cv2.FONT_HERSHEY_DUPLEX
                    scale, thick = 0.65, 2
                    (tw, th), _ = cv2.getTextSize(b_text, font, scale, thick)
                    bx1, by1 = 12, 12
                    bx2, by2 = bx1 + tw + 20, by1 + th + 16
                    
                    sub = display_frame[by1:by2, bx1:bx2]
                    bg = np.zeros_like(sub)
                    bg[:] = (18, 18, 18)
                    cv2.addWeighted(bg, 0.75, sub, 0.25, 0, sub)
                    display_frame[by1:by2, bx1:bx2] = sub
                    cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), br_col, 1, cv2.LINE_AA)
                    cv2.putText(display_frame, b_text, (bx1 + 10, by1 + th + 7), font, scale, b_col, thick, cv2.LINE_AA)
                    
                    cv2.imshow("Hand Tracking Camera", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        with self._ui_lock:
                            self._last_key = key
                    
                elapsed = time.perf_counter() - loop_start
                target_fps = self.camera_config.fps
                target_time = 1.0 / target_fps if target_fps > 0 else 0.033
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
            except Exception as e:
                if self._stop_event.is_set():
                    break
                import traceback
                print(f"Exception in InputPipeline loop: {e}")
                traceback.print_exc()
                time.sleep(0.5)
                
        # Cleanup
        logger.info("Cleaning up input pipeline...")
        try:
            if self.camera is not None:
                self.camera.stop()
        except:
            pass
        try:
            if self.tracker is not None:
                self.tracker.close()
        except:
            pass
