import sys
from panda3d.core import (
    WindowProperties,
    AmbientLight,
    DirectionalLight,
    Vec4,
    LColor
)
from direct.showbase.ShowBase import ShowBase
from src.config import RenderConfig

class PCASceneManager(ShowBase):
    """
    Panda3D application shell managing the scene graph, lighting, camera, and window.
    """
    def __init__(self, config: RenderConfig):
        super().__init__()
        
        self.app_config = config
        
        # Configure window
        props = WindowProperties()
        props.setSize(config.window_width, config.window_height)
        props.setTitle("Interactive Spatial PCA Visualization")
        self.win.requestProperties(props)
        
        # Set background color to solid black
        self.setBackgroundColor(LColor(0.0, 0.0, 0.0, 1.0))
        
        # Setup Scene Graph root nodes
        # Organized by visual hierarchy
        self.data_root = self.render.attachNewNode("data_root")
        
        self.points_root = self.data_root.attachNewNode("points_root")
        self.pcs_root = self.data_root.attachNewNode("pcs_root")
        self.box_root = self.data_root.attachNewNode("box_root")
        self.plane_root = self.data_root.attachNewNode("plane_root")
        self.guides_root = self.data_root.attachNewNode("guides_root")
        
        self._setup_lighting()
        self._setup_camera()
        
    def _setup_lighting(self):
        """Configure clean, scientific lighting."""
        # Ambient light (boosted slightly to prevent pure black shadows)
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.6, 0.6, 0.6, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)
        
        # Directional light (key light) - Attached to CAMERA!
        # This ensures the side of the object facing the user is ALWAYS brightly lit.
        directional = DirectionalLight("directional")
        directional.setColor(Vec4(0.7, 0.7, 0.7, 1.0))
        directional_np = self.camera.attachNewNode(directional)
        # Point straight ahead from the camera
        directional_np.setHpr(0, 0, 0)
        self.render.setLight(directional_np)
        
        # Fill light - Also attached to CAMERA
        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.4, 0.4, 0.45, 1.0))
        fill_np = self.camera.attachNewNode(fill)
        # Point slightly from the side to give some 3D depth perception
        fill_np.setHpr(45, 0, 0)
        self.render.setLight(fill_np)
        
    def _setup_camera(self):
        """Position camera to view the origin."""
        # Disable default mouse control so we can control it via hand gestures
        self.disableMouse()
        
        # Position camera looking at origin from a reasonable distance
        self.camera.setPos(0, -30, 10)
        self.camera.lookAt(0, 0, 0)

if __name__ == "__main__":
    print("Starting SceneManager test (will auto-close in 1s)...")
    config = RenderConfig()
    app = PCASceneManager(config)
    
    # Auto-close task for automated validation
    def stop_task(task):
        print("Closing SceneManager...")
        sys.exit(0)
        return task.done
        
    app.taskMgr.doMethodLater(1.0, stop_task, "StopTask")
    app.run()
