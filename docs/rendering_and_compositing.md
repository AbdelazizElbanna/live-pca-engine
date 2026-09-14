# 3D Spatial Rendering & Compositing

> "Clean, modern, scientific. No garish neon gamer HUDs; pure mathematical elegance."

The visual presentation of the **Live PCA Engine** is built on **Panda3D** for hardware-accelerated 3D spatial graphics, paired with an **OpenCV** real-time computer vision compositor.

---

## 1. The Panda3D Scene Graph Hierarchy

The 3D world is organized into a clean, parent-child scene graph under `self.data_root`:

```
                 Panda3D [render] Root
                           │
                 ┌─────────┴─────────┐
                 │    Camera Node    │ ◄── Key & Fill Lights attached here
                 └───────────────────┘
                           │
                 ┌───────────────────┐
                 │     data_root     │ ◄── Receives Trackball Mat4 & Zoom Scale
                 └─────────┬─────────┘
       ┌──────────────┬────┴─────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼              ▼
  points_root      pcs_root       box_root      plane_root    guides_root
 (Point Cloud)   (PC Vectors)    (Wireframe)    (2D Plane)   (Trajectories)
```

### Camera-Attached Lighting
In scientific visualizations, orbiting around a dataset often places the camera in the shadows. We solve this by attaching the main lighting rig directly to the **Camera Node**:
- **Key Light:** Directional light pointing directly along camera forward `(0, 0, 0)` with intensity `(0.7, 0.7, 0.7)`.
- **Fill Light:** Offset at `45°` yaw with intensity `(0.4, 0.4, 0.45)` for subtle 3D depth shading.
- **Ambient Light:** Soft baseline light `(0.6, 0.6, 0.6)` ensuring no point ever plunges into pure pitch-black shadow.

---

## 2. Point Cloud Visual (`GeomPoints` & Point Sprites)

Implemented in [`src/renderer/point_cloud.py`](../src/renderer/point_cloud.py).

### Hardware Point Sprites
Rather than creating expensive 3D sphere meshes for hundreds of points, the engine renders point primitives using hardware point sprites:
- **Texture:** Mapped with `assets/sphere.png` using `TexGenAttrib.MPointSprite`.
- **Transparency:** `TransparencyAttrib.MAlpha` with `setDepthWrite(True)` and `setDepthTest(True)`.
- **Dynamic Sizing:** As the user zooms in or out, point thickness dynamically scales between 2 and 20 pixels:
  $$\text{size} = \text{clamp}(10.0 \times \text{zoom}, 2.0, 20.0)$$
- **Zero-Allocation Updates:** Point positions are streamed directly into the existing GPU vertex array buffer via `GeomVertexWriter(vdata, "vertex")`, completely avoiding CPU garbage collection spikes.

---

## 3. Principal Component Vectors (Eigen Arrows)

Implemented in [`src/renderer/pc_vectors.py`](../src/renderer/pc_vectors.py).

```
   ▲ PC1 (Magenta, λ₁) ── High Variance Direction
   │
   │       ▲ PC2 (Orange, λ₂) ── Secondary Variance
   │      /
   │     /
   │    / 
   │   /
   └───┼───────────▶ PC3 (Yellow, λ₃) ── Smallest Variance (Collapses!)
    Centroid μ
```

### Mathematical Scale Convention
- Vector directions are the exact orthonormal columns of the eigenvector matrix $\mathbf{V}$.
- The length of vector $j$ is scaled to:
  $$\text{length}_j = \sqrt{\lambda_j}$$
  This represents exactly **one standard deviation** ($\sigma_j$) along that principal direction.
- During compression ($c > 0$), PC3's length dynamically contracts by $(1 - c)$ until it disappears into the 2D plane at $c = 1.0$.

---

## 4. The 2D Projection Plane & Depth Sorting

Implemented in [`src/renderer/plane_visual.py`](../src/renderer/plane_visual.py).

One of the trickiest graphical problems in scientific 3D rendering is cross-sectional slicing: you want the projection plane to be semi-transparent, but you still want points on the near side of the plane to properly occlude it, and points on the far side to show behind it.

### The Fixed Bin Ordering Solution
Panda3D manages render ordering through render bins:
1. **Point Cloud:** Placed in `fixed` bin with priority `10`. Depth writes are enabled.
2. **Projection Plane:** Placed in `fixed` bin with priority `30`. Depth writes are disabled (`setDepthWrite(False)`), but depth testing is enabled (`setDepthTest(True)`).

**Result:** The point cloud renders first and writes true depth values to the Z-buffer. When the semi-transparent plane renders second, points in front stay in front, points behind are seen through the tinted glass, and points cutting through the plane form a razor-sharp, mathematically accurate cross-section!

---

## 5. The HUD & OpenCV Tracking Window

The secondary window provides continuous feedback without cluttering the 3D viewport.

```
┌─────────────────────────────────────────────────────────────┐
│ [ > ROTATE ]  Hand Tracking Camera              30.0 FPS    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│         (•)──(•)──(•)──(•)                                  │
│          │    │    │    │    (Fingertips with White Cores)  │
│         (•)──(•)──(•)──(•)                                  │
│           \   │    │   /     (Anti-Aliased Cyan Bones)      │
│            \  │    │  /                                     │
│             (•)──(•)         (Metacarpal Halos)             │
│                 \                                           │
│                 (•) Wrist                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Aesthetic Skeleton Visualizer
Implemented in [`src/compositor/hand_visual.py`](../src/compositor/hand_visual.py):
- **Bones:** Anti-aliased lines in curated cyan/turquoise `(220, 215, 45)`.
- **Joints:** Semi-transparent halo rings with bright ivory cores `(255, 255, 220)`.
- **Fingertips:** Pure luminous white dots `(255, 255, 255)`.
- **Status Badge:** Frosted-glass darkened rectangle displaying current mode (`[ > ROTATE ]`, `[ > ZOOM ]`, `[ > COMPRESS ]`, or `[ || FROZEN ]`).
