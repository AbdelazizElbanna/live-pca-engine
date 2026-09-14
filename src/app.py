import cv2
import sys
import numpy as np
from panda3d.core import Texture, GraphicsOutput
from direct.task import Task

from src.config import DEFAULT_CONFIG, AppConfig
from src.renderer.scene_manager import PCASceneManager
from src.renderer.point_cloud import PointCloudVisual
from src.renderer.pc_vectors import PCVectorsVisual
from src.renderer.pca_box_visual import PCABoxVisual
from src.renderer.projection_visual import ProjectionGuidesVisual
from src.renderer.plane_visual import PlaneVisual
from src.renderer.materials import MaterialSystem
from src.input.camera import CameraCapture
from src.input.hand_tracker import HandTracker
from src.input.smoothing import LandmarkSmoother
from src.interaction.gesture_recognizer import GestureRecognizer
from src.interaction.state_manager import StateManager
from src.compositor import Compositor, draw_aesthetic_hand_landmarks
from src.math_core.projection import compute_projection
from src.performance.profiler import Profiler

from src.input.pipeline import InputPipeline

class PCApplication:
    """Main application wiring all components together."""
    
    def __init__(self, config: AppConfig = DEFAULT_CONFIG):
        self.config = config
        
        # 1. Initialize State
        self.state_manager = StateManager(config)
        
        # 2. Initialize Input Pipeline (Asynchronous)
        self.input_pipeline = InputPipeline(config.camera, max_hands=2)
        self.smoother = LandmarkSmoother(config.interaction)
        self.recognizer = GestureRecognizer(config.interaction)
        
        # 3. Initialize Renderer & Compositor
        self.scene = PCASceneManager(config.render)
        self.materials = MaterialSystem(config.visual)
        self.compositor = Compositor()
        self.profiler = Profiler(enabled=config.enable_profiling)
        
        # Setup Visuals
        self.point_cloud = PointCloudVisual(self.scene.points_root, config.visual, max_points=config.data.n_points)
        self.pc_vectors = PCVectorsVisual(self.scene.pcs_root, config.visual)
        self.pca_box_visual = PCABoxVisual(self.scene.box_root, config.visual)
        self.pca_box_visual.node_path.hide()  # Temporarily hidden for visual test
        self.projection_guides = ProjectionGuidesVisual(self.scene.guides_root, config.visual, max_points=config.data.n_points)
        self.plane_visual = PlaneVisual(self.scene.plane_root, config.visual)
        
        # We start the input pipeline in the background
        self.input_pipeline.start()
            
        # Hook into Panda3D's main loop
        self.scene.taskMgr.add(self.main_loop_task, "main_loop_task")
        
        # Setup Panda3D key bindings
        self.scene.accept('escape', self.shutdown)
        self.scene.accept('q', self.shutdown)
        self.scene.accept('p', self._toggle_profiler)
        if self.scene.win:
            self.scene.win.setCloseRequestEvent('close_window_event')
            self.scene.accept('close_window_event', self.shutdown)
        
        # Mode bindings
        self.scene.accept('space', self._toggle_idle)
        self.scene.accept('0', self._set_mode_idle)
        self.scene.accept('1', self._set_mode_rotation)
        self.scene.accept('2', self._set_mode_zoom)
        self.scene.accept('3', self._set_mode_compression)
        self.scene.accept('mouse1', self._cycle_mode_next)
        self.scene.accept('mouse3', self._cycle_mode_prev)
        
        # UI for Mode - sleek bottom status footer away from 3D data
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        self.footer_text = OnscreenText(
            text="",
            pos=(0.0, -0.93), scale=0.042, fg=(0.85, 0.85, 0.85, 0.85),
            align=TextNode.ACenter, shadow=(0, 0, 0, 0.95), shadowOffset=(0.04, 0.04)
        )
        self._set_mode_idle()
        
        self.first_frame = True
        self.last_snapshot_time = 0.0
        self.current_landmarks = []
        
    def _update_ui_mode(self):
        from src.interaction.state_manager import InteractionMode
        from panda3d.core import WindowProperties
        mode = self.state_manager.current_mode
        if mode == InteractionMode.IDLE:
            self.footer_text.setText("[ FROZEN / IDLE ]   Press [Space] or [1] Rotate | [2] Zoom | [3] Compress")
            self.footer_text.setFg((1.0, 0.78, 0.2, 0.95))
            win_title = "Live PCA Engine - [FROZEN] Press Space or 1,2,3"
            cv_title = "Hand Tracking Camera [FROZEN] - Space: Resume | 1:Rot | 2:Zoom | 3:Comp"
        elif mode == InteractionMode.ROTATION:
            self.footer_text.setText("[ ROTATION MODE ]   [Space] Freeze | [1] Rotate | [2] Zoom | [3] Compress")
            self.footer_text.setFg((0.3, 1.0, 0.6, 0.95))
            win_title = "Live PCA Engine - [ROTATION] Press Space to Freeze"
            cv_title = "Hand Tracking Camera [ROTATION] - Space: Freeze | 1:Rot | 2:Zoom | 3:Comp"
        elif mode == InteractionMode.ZOOM:
            self.footer_text.setText("[ ZOOM MODE ]   [Space] Freeze | [1] Rotate | [2] Zoom | [3] Compress")
            self.footer_text.setFg((0.3, 0.85, 1.0, 0.95))
            win_title = "Live PCA Engine - [ZOOM] Press Space to Freeze"
            cv_title = "Hand Tracking Camera [ZOOM] - Space: Freeze | 1:Rot | 2:Zoom | 3:Comp"
        elif mode == InteractionMode.COMPRESSION:
            self.footer_text.setText("[ COMPRESSION MODE ]   [Space] Freeze | [1] Rotate | [2] Zoom | [3] Compress")
            self.footer_text.setFg((0.9, 0.4, 1.0, 0.95))
            win_title = "Live PCA Engine - [COMPRESSION] Press Space to Freeze"
            cv_title = "Hand Tracking Camera [COMPRESSION] - Space: Freeze | 1:Rot | 2:Zoom | 3:Comp"
            
        if self.scene.win:
            props = WindowProperties()
            props.setTitle(win_title)
            self.scene.win.requestProperties(props)
            
        # Update OpenCV window via pipeline
        badge_text = ""
        badge_col = (255, 255, 255)
        border_col = (255, 255, 255)
        
        if mode == InteractionMode.IDLE:
            badge_text = "|| FROZEN"
            badge_col = (0, 195, 255) # Amber
            border_col = (0, 160, 220)
        elif mode == InteractionMode.ROTATION:
            badge_text = "> ROTATE"
            badge_col = (60, 255, 120) # Neon Green
            border_col = (40, 200, 90)
        elif mode == InteractionMode.ZOOM:
            badge_text = "> ZOOM"
            badge_col = (255, 210, 60) # Cyan/Sky
            border_col = (200, 170, 40)
        elif mode == InteractionMode.COMPRESSION:
            badge_text = "> COMPRESS"
            badge_col = (255, 110, 220) # Magenta
            border_col = (200, 80, 180)
            
        self.input_pipeline.set_ui_state(badge_text, badge_col, border_col, cv_title)

    def _set_mode_idle(self):
        from src.interaction.state_manager import InteractionMode
        self.state_manager.set_mode(InteractionMode.IDLE)
        self._update_ui_mode()
        
    def _set_mode_rotation(self):
        from src.interaction.state_manager import InteractionMode
        self.state_manager.set_mode(InteractionMode.ROTATION)
        self._update_ui_mode()
        
    def _set_mode_zoom(self):
        from src.interaction.state_manager import InteractionMode
        self.state_manager.set_mode(InteractionMode.ZOOM)
        self._update_ui_mode()
        
    def _set_mode_compression(self):
        from src.interaction.state_manager import InteractionMode
        self.state_manager.set_mode(InteractionMode.COMPRESSION)
        self._update_ui_mode()

    def _toggle_idle(self):
        from src.interaction.state_manager import InteractionMode
        if self.state_manager.current_mode == InteractionMode.IDLE:
            target = getattr(self.state_manager, 'last_active_mode', InteractionMode.ROTATION)
            if target == InteractionMode.ZOOM:
                self._set_mode_zoom()
            elif target == InteractionMode.COMPRESSION:
                self._set_mode_compression()
            else:
                self._set_mode_rotation()
        else:
            self._set_mode_idle()
        
    def _cycle_mode_next(self):
        from src.interaction.state_manager import InteractionMode
        current = self.state_manager.current_mode
        if current == InteractionMode.IDLE:
            self._set_mode_rotation()
        elif current == InteractionMode.ROTATION:
            self._set_mode_zoom()
        elif current == InteractionMode.ZOOM:
            self._set_mode_compression()
        else:
            self._set_mode_rotation()

    def _cycle_mode_prev(self):
        from src.interaction.state_manager import InteractionMode
        current = self.state_manager.current_mode
        if current == InteractionMode.IDLE:
            self._set_mode_compression()
        elif current == InteractionMode.ROTATION:
            self._set_mode_compression()
        elif current == InteractionMode.ZOOM:
            self._set_mode_rotation()
        else:
            self._set_mode_zoom()
        
    def _toggle_profiler(self):
        self.profiler.enabled = not self.profiler.enabled

    def _update_scene(self, state):
        """Update Panda3D scene graph with current state."""
        # Update Scene Transform using Trackball Matrix
        m3 = state.object_rotation_mat
        
        # Convert numpy 3x3 (which acts on column vectors) to Panda3D Mat4.
        # Panda3D uses row-major vectors (v * M), so we must transpose the numpy matrix!
        from panda3d.core import Mat4
        m4 = Mat4(
            m3[0,0], m3[1,0], m3[2,0], 0.0,
            m3[0,1], m3[1,1], m3[2,1], 0.0,
            m3[0,2], m3[1,2], m3[2,2], 0.0,
            0.0,     0.0,     0.0,     1.0
        )
        self.scene.data_root.setMat(m4)
        
        # Apply true spatial scaling based on zoom state
        self.scene.data_root.setScale(state.camera_zoom)
        
        # Calculate projection based on compression_c
        proj_state = compute_projection(
            data=state.dataset,
            pca_result=state.pca_result,
            pca_box=state.pca_box,
            c=state.compression_c
        )
        
        # Update point cloud positions
        # Simple coloring based on Z position for now
        z_norm = (proj_state.positions[:, 2] - np.min(proj_state.positions[:, 2])) / (np.ptp(proj_state.positions[:, 2]) + 1e-5)
        colors = np.zeros((self.config.data.n_points, 4))
        base_col = self.materials.get_point_color()
        for i in range(self.config.data.n_points):
            colors[i] = [base_col.getX(), base_col.getY() * z_norm[i], base_col.getZ(), base_col.getW()]
            
        # Calculate dynamic point size based on zoom only
        dynamic_size = np.clip(10.0 * state.camera_zoom, 2.0, 20.0)
        self.point_cloud.set_point_size(dynamic_size)
        
        self.point_cloud.update_points(proj_state.positions, colors)
        
        # Update PC vectors (with alpha for PC3)
        self.pc_vectors.update_vectors(state.pca_result, pc3_alpha=proj_state.pc3_alpha)
        
        # Update PCA box
        self.pca_box_visual.update_box(state.pca_box, box_extent_3=proj_state.box_extent_3)
        
        # Update Projection Guides
        # Fade in as c increases from 0 to 0.5, then fade out from 0.5 to 1.0
        c = state.compression_c
        guide_alpha = (c * 2.0) if c < 0.5 else ((1.0 - c) * 2.0)
        guide_color = self.materials.get_guide_color(alpha=guide_alpha)
        
        if guide_alpha > 0.01:
            self.projection_guides.node_path.show()
            self.projection_guides.update_guides(proj_state, base_color=guide_color)
        else:
            self.projection_guides.node_path.hide()
            
        # Update Plane
        self.plane_visual.update_plane(state.pca_box, state.pca_result, alpha=c)

    def main_loop_task(self, task):
        """Panda3D task executing the main loop."""
        try:
            self.profiler.start("total")
            
            # 1. Read input snapshot (non-blocking)
            self.profiler.start("camera")
            snapshot = self.input_pipeline.get_latest_snapshot()
            cam_frame = snapshot.frame
            self.profiler.stop("camera")
            
            # 2. Track & Smooth Hand (Only if new frame arrived)
            self.profiler.start("hand_tracking")
            if snapshot.timestamp != self.last_snapshot_time:
                self.last_snapshot_time = snapshot.timestamp
                
                if snapshot.landmarks:
                    smoothed_landmarks = self.smoother.smooth(snapshot.landmarks)
                else:
                    smoothed_landmarks = self.smoother.smooth([])
                self.current_landmarks = smoothed_landmarks
                self.profiler.stop("hand_tracking")
                    
                # 3. Recognize Gesture & Update State target velocity
                self.profiler.start("interaction")
                intent = self.recognizer.recognize(smoothed_landmarks)
                self.state_manager.apply_intent(intent)
                self.profiler.stop("interaction")
            else:
                self.profiler.stop("hand_tracking")
                
            # Apply physics/momentum using delta time
            from direct.showbase.ShowBaseGlobal import globalClock
            dt = globalClock.getDt()
            self.state_manager.update(dt)
            state = self.state_manager.get_scene_state()
            
            # 4. Handle Keys from OpenCV Window (now isolated in background thread)
            key = self.input_pipeline.get_last_key()
            if key != -1:
                if key == ord(' ') or key == ord('0'):
                    self._toggle_idle()
                elif key == ord('1'):
                    self._set_mode_rotation()
                elif key == ord('2'):
                    self._set_mode_zoom()
                elif key == ord('3'):
                    self._set_mode_compression()
                elif key == ord('q') or key == 27:
                    self.shutdown()
                    return Task.done
                
            # 5. Update Scene Graphics
            self.profiler.start("scene_update")
            self._update_scene(state)
            self.profiler.stop("scene_update")
            
            self.profiler.stop("total")
            return Task.cont
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.shutdown()
            return Task.done

    def shutdown(self):
        """Cleanly shutdown resources."""
        print("Shutting down...")
        self.input_pipeline.stop()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        try:
            self.scene.userExit()
        except:
            import sys
            sys.exit(0)

    def run(self):
        """Start the application loop."""
        print("Starting Interactive Spatial PCA Visualization...")
        # Since we use Panda3D's event loop, we call run() on the ShowBase instance
        self.scene.run()
