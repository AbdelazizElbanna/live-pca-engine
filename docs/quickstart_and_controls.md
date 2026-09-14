# Quickstart & Controls Cheat Sheet

> "Zero to presenting in 60 seconds."

This guide contains everything you need to set up, configure, and operate the **Live PCA Engine**.

---

## 1. Environment & Installation

### Requirements
- **Python:** 3.10 or 3.11 (Python 3.11 recommended)
- **Webcam:** Standard USB webcam or built-in laptop camera (720p or 1080p)
- **Dependencies:** Listed in [requirements.txt](../requirements.txt) and [environment.yml](../environment.yml)

### Option A: Using Conda (Recommended)

```bash
# 1. Create a clean environment with Python 3.11
conda create -n live-pca python=3.11 -y
conda activate live-pca

# 2. Install required packages
pip install -r requirements.txt
```

Alternatively, you can build directly from the Conda environment file:
```bash
conda env create -f environment.yml
conda activate scientific-visualization
```

### Option B: Using Python Virtual Environment (venv)

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Verification
Verify that core packages import without errors:
```bash
python -c "import numpy, cv2, mediapipe, panda3d; print('All dependencies successfully installed!')"
```

---

## 2. Launching the Application

### Standard Launch
```bash
python main.py
```

### Dedicated GPU Launch (Linux / Hybrid GPUs)
If you are running on a Linux machine with dual GPUs (integrated Intel/AMD + dedicated NVIDIA), run via the GPU script:
```bash
./run_gpu.sh
```

---

## 3. Controls Cheat Sheet

### Keyboard & Mouse Shortcuts

You can trigger any mode directly from either the Panda3D 3D window or the OpenCV camera window:

| Key / Input | Action | What It Does |
|---|---|---|
| **`[Space]`** | **Toggle Freeze / Resume** | Instantly freezes the virtual world so you can gesture freely while speaking. |
| **`[1]`** | **Rotation Mode** | Enables trackball hand rotation. |
| **`[2]`** | **Zoom Mode** | Enables depth-based virtual scaling. |
| **`[3]`** | **Compression Mode** | Enables physical dimensionality reduction into 2D. |
| **`[P]`** | **Toggle Profiler** | Shows real-time FPS and millisecond latency for each pipeline stage. |
| **`[Q]`** or **`[Esc]`** | **Clean Exit** | Gracefully closes all threads, cameras, and windows. |
| **Left Click** | **Cycle Mode Next** | Cycles forward: Rotate -> Zoom -> Compress. |
| **Right Click** | **Cycle Mode Prev** | Cycles backward through modes. |

---

## 4. Presenting Like a Pro (Hand Gestures)

The engine tracks your **physical right hand** in selfie mirror mode.

```
       1. ROTATE                        2. ZOOM                      3. COMPRESS
   (Back of hand to camera)       (Move closer / further)       (Push forward / pull back)
         ──────>                         ───> <───                         ───>
      Swiping motions               Closer  = Bigger                Push = Flatten to 2D
    ratchet the 3D cloud.           Further = Smaller               Pull = Restore to 3D
```

### The Ratchet Move (Rotating Effortlessly)
1. Face the **back of your right hand** toward the webcam (so your palm faces your face).
2. Swipe left or right. The 3D cloud spins smoothly with momentum.
3. To reposition your hand without spinning the cloud backward: **flip your hand so your palm faces the webcam**, move your hand back to center, and flip it back.

### Zooming Smoothly
1. Press `[2]` on the keyboard to enter Zoom Mode.
2. Move your hand closer to the camera: the point cloud expands to fill the room.
3. Pull your hand back: the point cloud contracts to a compact core.
4. *Tip:* Because the engine uses the rigid metacarpal bone metric, curling or stretching your fingers will not disrupt the zoom.

### Compression: Pressing 3D into 2D
1. Press `[3]` on the keyboard to enter Compression Mode.
2. Notice the yellow PC3 arrow (the axis of smallest variance).
3. Push your hand firmly toward the webcam:
   - The bounding box collapses.
   - The yellow PC3 arrow shrinks.
   - Dotted trajectory lines emerge, guiding each 3D point along its optimal projection path.
   - The point cloud flattens into a razor-thin 2D slice on the glowing projection plane.
4. The magnetic spring automatically snaps cleanly into the $c = 1.000$ subspace.
5. Press `[1]` to rotate the new 2D slice and show your audience that variance along the third dimension has been eliminated.

---

## 5. Configuration & Troubleshooting

All system defaults are configured in [`src/config.py`](../src/config.py):

```python
# Change camera device (if you have multiple webcams):
device_id: int = 0  # Change to 1 or 2 if needed

# Change dataset size:
n_points: int = 480  # Increase for denser clouds, decrease for ultra-lightweight CPUs

# Change ellipsoid variance structure:
axis_lengths: Tuple[float, float, float] = (5.0, 3.0, 1.0)
```

### Running the Test Suite
To verify the math and pipeline integrity at any time:
```bash
pytest
```
All 30 tests should pass cleanly in ~3 seconds.
