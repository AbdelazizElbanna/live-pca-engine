# System Architecture & Data Flow

> "Decouple the slow physical world from the fast virtual world."

Real-time computer vision applications frequently suffer from a classic flaw: if the webcam slows down or drops a frame, the entire 3D rendering pipeline stutters. 

The **Live PCA Engine** solves this with a **decoupled, multi-threaded dual-loop architecture**.

---

## 1. The Dual-Loop Threading Model

The application operates two concurrent loops connected through an atomic, lock-protected memory bridge:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   THREAD 1: INPUT PIPELINE (Daemon)                      │
│                                                                          │
│   CameraCapture.read() ──▶ HandTracker.process() ──▶ InputSnapshot()     │
│   (Blocks at ~30 FPS)       (MediaPipe Tasks v1)      (Thread-safe swap) │
└────────────────────────────────────────┬─────────────────────────────────┘
                                         │
                                [Atomic Snapshot]
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   THREAD 2: MAIN ENGINE & RENDERER                       │
│                                                                          │
│   Panda3D TaskMgr ──▶ GestureRecognizer ──▶ StateManager (Physics)       │
│   (Locked @ 60 FPS)   (Intent Extraction)   (Trackball & Lerp)           │
│                             │                                            │
│                             ▼                                            │
│                       Math Core & PCA ──▶ Panda3D Scene Graph            │
│                       (Vectorized)        (GPU Vertex Buffers)           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters
- If MediaPipe spends 25ms analyzing a complex hand posture, **Panda3D never stutters**.
- Panda3D continues applying smooth momentum, friction, and camera transitions every frame.
- Frame updates are passed via an immutable, frozen dataclass:

```python
@dataclass(frozen=True)
class InputSnapshot:
    frame: Optional[np.ndarray]
    landmarks: Optional[List[HandLandmarks]]
    timestamp: float
```

The main render thread grabs the snapshot non-blockingly via `get_latest_snapshot()`. If no new camera frame arrived, the render thread skips redundant tracking computations and smoothly integrates momentum physics.

---

## 2. The 5 Explicit Coordinate Spaces

A frequent source of bugs in spatial AR applications is coordinate cross-contamination (e.g., trying to use camera pixel coordinates directly in a 3D scene). 

The engine enforces **5 strictly bounded coordinate spaces**:

```
 ┌─────────────────┐       ┌─────────────────┐       ┌──────────────────┐
 │  1. Image Space │ ────▶ │2. Tracking Space│ ────▶ │3.Interaction Space│
 │  (Camera Pixels)│       │ (MediaPipe Box) │       │ (Semantic Deltas)│
 └─────────────────┘       └─────────────────┘       └─────────┬────────┘
                                                               │
                                                               ▼
                           ┌─────────────────┐       ┌──────────────────┐
                           │   5. PCA Space  │ ◀──── │ 4. Virtual Scene │
                           │(Eigen Basis [V])│       │ (Panda3D World)  │
                           └─────────────────┘       └──────────────────┘
```

### Space Definitions & Ownership

| Coordinate Space | Domain / Dimensions | Owner | Description |
|---|---|---|---|
| **1. Camera / Image Space** | $(0, 0) \dots (1280, 720)$ pixels | [`src/input/camera.py`](../src/input/camera.py) | Raw 2D BGR frame captured from OpenCV webcam. |
| **2. Tracking Space** | $[0, 1] \times [0, 1] \times [-1, 1]$ | [`src/input/hand_tracker.py`](../src/input/hand_tracker.py) | Normalized landmark positions inside the square bounding box. |
| **3. Interaction Space** | Velocity deltas: $(\Delta\text{yaw}, \Delta\text{pitch}, \Delta\text{zoom}, \Delta c)$ | [`src/interaction/gesture_recognizer.py`](../src/interaction/gesture_recognizer.py) | Semantic user intents, stripped of hand coordinates. |
| **4. Virtual Scene Space** | $(X, Y, Z) \in \mathbb{R}^3$ meters | [`src/renderer/scene_manager.py`](../src/renderer/scene_manager.py) | Panda3D 3D world space. Camera at $(0, -30, 10)$ looking at $(0, 0, 0)$. |
| **5. PCA Space** | Principal coordinates $Z = X_c V$ | [`src/math_core/`](../src/math_core/) | Intrinsic subspace aligned with the eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$. |

---

## 3. Modular Separation (LEGO Principle)

Every module in `src/` can be isolated, instantiated, and unit-tested without its neighbors:

1. **Math Core:** Zero dependencies on Panda3D, OpenCV, or MediaPipe. Tested purely with NumPy.
2. **Input Pipeline:** Supports mock camera frame injection (`inject_mock_frame`) for testing environments without physical webcams.
3. **Gesture Recognizer:** Tested against static JSON landmark dumps (`tests/fixtures/sample_hand_landmarks.json`).
4. **Renderer:** Manages its own nodes and transforms, completely agnostic to how the rotation matrix or compression factor was calculated.
