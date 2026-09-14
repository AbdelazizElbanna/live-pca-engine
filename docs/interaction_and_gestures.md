# Spatial Control, Hand Tracking & Gestures

> "Natural physical interaction requires two things: rock-solid tracking stability and an intuitive clutching mechanism."

Controlling a 3D mathematical visualization using hand gestures in front of a webcam often suffers from jitter, accidental inputs, and awkward hand positioning. This guide explains how the **Live PCA Engine** solves these problems with custom tracking algorithms and state machines.

---

## 1. Hand Tracking: MediaPipe Tasks API v1.0.1

The engine uses Google's latest **MediaPipe Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker`). 

### The 256×256 Square Fix (Eliminating the C++ Core Dump)
When running MediaPipe's HandLandmarker in `IMAGE` mode on widescreen inputs (e.g. 16:9 / 720p), MediaPipe's internal C++ ROI-tracking graph frequently triggers a segmentation fault / core dump due to aspect ratio distortion.

In [`src/input/hand_tracker.py`](../src/input/hand_tracker.py), we implemented an bulletproof fix:
1. **Square Padding:** We pad the 1280×720 image with black borders into a 1280×1280 square.
2. **Native Resizing:** We resize that square directly to **256×256**, matching the exact native neural net tensor size expected by `hand_landmarker.task`.
3. **Coordinate Unpadding:** Landmarks extracted in normalized square coordinates are mapped back to widescreen pixels using the known padding offset.

```
  1280x720 (16:9)            1280x1280 Square            256x256 Tensor
┌─────────────────┐        ┌──────────────────┐        ┌─────────┐
│                 │        │░░░░░░░░░░░░░░░░░░│        │░░░░░░░░░│
│  Camera Frame   │ ─────▶ │   Camera Frame   │ ─────▶ │ Model   │ ──▶ 21 3D Landmarks
│                 │        │░░░░░░░░░░░░░░░░░░│        │░░░░░░░░░│     (Rock Solid)
└─────────────────┘        └──────────────────┘        └─────────┘
```

### Chirality & Mirror Inversion
The camera feed is flipped horizontally (`cv2.flip(frame, 1)`) so it functions like a natural mirror.
- In mirror reflection, chirality is inverted: MediaPipe sees your physical **right hand** on the left side of the frame and labels it as **"Left"**.
- The engine explicitly inverts this check:
  ```python
  is_physical_right = (handedness_info.category_name == "Left")
  ```
- The engine **exclusively tracks your physical right hand**, completely ignoring your left hand or anyone walking behind you.

---

## 2. Landmark Smoothing (EMA Filter)

Raw optical tracking exhibits high-frequency microscopic jitter. We apply an Exponential Moving Average (EMA) smoother in [`src/input/smoothing.py`](../src/input/smoothing.py):

$$\mathbf{p}_{smooth}(t) = \alpha \cdot \mathbf{p}_{smooth}(t - 1) + (1 - \alpha) \cdot \mathbf{p}_{raw}(t)$$

With $\alpha = 0.3$, hand jitter disappears while maintaining snappy responsiveness with zero noticeable lag.

---

## 3. The 4 Interaction Modes

To keep interactions simple and reliable, the application operates in discrete, unambiguous modes:

```
  [Space] IDLE / FROZEN ──▶ Safe mode for presenting and talking.
  [1]     ROTATION      ──▶ Spin the 3D data distribution via trackball dragging.
  [2]     ZOOM          ──▶ Expand or shrink the point cloud in virtual space.
  [3]     COMPRESSION   ──▶ Physically press down to flatten 3D into 2D.
```

You can toggle modes using **Keys `[1]`, `[2]`, `[3]`** or cycle them with mouse clicks.

---

## 4. Deep-Dive: The Ratchet / Clutch Mechanism

When rotating a 3D object on a screen, what happens when your hand reaches the edge of your camera frame?
If you move your hand back to the center, a naive system rotates the object backward, cancelling your movement!

We invented a **physical clutching ratchet** in [`src/interaction/gesture_recognizer.py`](../src/interaction/gesture_recognizer.py):

```
       USER SEES HER PALM                     USER SEES BACK OF HAND
    (Back of hand faces camera)                (Palm faces camera)
    
           ┌──────────┐                            ┌──────────┐
           │  CLUTCH  │                            │  PAUSE   │
           │  ENGAGED │                            │DISENGAGED│
           └────┬─────┘                            └────┬─────┘
                │                                       │
                ▼                                       ▼
        Object Rotates                          Object Stays Still
      (Trackball Active)                     (Reset hand to center)
```

### Mathematical Formulation
We compute the 2D cross-product of the index finger base vector ($\mathbf{v}_{index} = \text{MCP}_5 - \text{Wrist}_0$) and the pinky base vector ($\mathbf{v}_{pinky} = \text{MCP}_{17} - \text{Wrist}_0$):

$$\text{cross}_z = v_{index}^x \cdot v_{pinky}^y - v_{index}^y \cdot v_{pinky}^x$$

In a mirrored camera feed:
- When $\text{cross}_z > 0$: The back of your hand is facing the camera (you see your palm). **Rotation is active.**
- When $\text{cross}_z \le 0$: The palm of your hand faces the camera. **Rotation is clutched/disconnected.**

**The Experience:** You swipe right to spin the point cloud, turn your hand slightly toward the camera to bring your arm back, and swipe again. It feels exactly like using a mechanical ratchet wrench!

---

## 5. Trackball Math & Re-Orthogonalization

Rather than maintaining Euler angles (which suffer from gimbal lock), [`src/interaction/state_manager.py`](../src/interaction/state_manager.py) accumulates rotation directly in a $3 \times 3$ orthonormal matrix $\mathbf{R}_{obj} \in SO(3)$:

$$\mathbf{R}_{obj}(t) = \mathbf{R}_z(-\Delta\text{yaw}) \cdot \mathbf{R}_x(-\Delta\text{pitch}) \cdot \mathbf{R}_{obj}(t - 1)$$

### SVD Re-Orthogonalization
Over thousands of frames, floating-point roundoff can cause the rotation matrix to warp or scale. Every frame, we re-orthogonalize via Singular Value Decomposition:

$$\mathbf{U}, \boldsymbol{\Sigma}, \mathbf{V}^T = \text{SVD}(\mathbf{R}_{obj}), \qquad \mathbf{R}_{clean} = \mathbf{U} \mathbf{V}^T$$

This ensures the transformation matrix remains a mathematically pure rotation forever.

---

## 6. Depth Zoom & Robust Palm Metric

Depth estimation from a single 2D webcam is notoriously noisy. Most apps estimate hand distance using the distance between index tip and wrist, but if you bend your fingers, the app zooms uncontrollably!

We developed an **invariant palm metric**:
1. Palm width: $w = \|\text{MCP}_5 - \text{MCP}_{17}\|$ (Index MCP to Pinky MCP)
2. Palm height: $h = \|\text{Wrist}_0 - \text{MCP}_9\|$ (Wrist to Middle MCP)
3. Hand size metric: $s = \max(w, h)$

Because this takes the maximum between the perpendicular width and height of the rigid metacarpal bones, **tilting, pitching, or curling your fingers will not distort the zoom level**.

---

## 7. Compression Momentum & Magnetic Extreme Springs

In Compression Mode, your hand velocity along the smallest variance axis drives the compression state $c \in [0, 1]$.

To make presentations effortless:
1. **Velocity Lerping:** $v_c \leftarrow v_c + (v_{target} - v_c) \cdot 0.15$ removes all hand tremors.
2. **Magnetic Extreme Springs:** If you push the cloud and let go around $c = 0.7$, an artificial spring pulls $c$ smoothly to $1.0$. If you pull back to $c = 0.2$, the spring restores it to $0.0$.
3. **Tangent Snapping:** When $|1.0 - c| < 0.005$, the engine snaps cleanly to $c = 1.000$, guaranteeing that points lie exactly on the 2D plane with zero mathematical error.
