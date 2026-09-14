# Live PCA Engine — Interactive Spatial System

> **Turning abstract linear algebra into a tangible, physical 3D reality.**

Welcome to the **Live PCA Engine** documentation! 

Most people learn Principal Component Analysis (PCA) through dry covariance formulas, eigenvector proofs, or flat 2D scatter plots in Jupyter notebooks. This engine takes a radically different approach: **Physical Linear Algebra**.

You stand in front of your webcam, reach your hand into a floating 3D Gaussian cloud, rotate it like a physical sphere, inspect the emerging orthogonal eigenvectors, and physically push down on the dimension with the smallest variance to watch 3D points collapse smoothly into an optimal 2D projection subspace in real time.

---

## The Big Picture

Here is how the entire system communicates in a single, non-blocking pipeline:

```
                  ┌────────────────────────────────────────┐
                  │             WEBCAM FEED                │
                  │        (720p @ 30 FPS / BGR)           │
                  └──────────────────┬─────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │              BACKGROUND INPUT PIPELINE                 │
        │                                                        │
        │  [MediaPipe Tasks v1] ──▶ 256x256 Square ROI Fix       │
        │  [Landmark Smoother]  ──▶ Exponential Moving Average   │
        │  [Atomic Snapshot]    ──▶ Non-blocking thread boundary │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │            INTERACTION & STATE ENGINE                  │
        │                                                        │
        │  [Clutch Mechanism]   ──▶ Back-of-hand ratchet check   │
        │  [Trackball Rotation] ──▶ World-space SVD rotation mat │
        │  [Momentum & Springs] ──▶ Physics lerp & snapping      │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │                  3D SPATIAL ENGINE                     │
        │                                                        │
        │  [Panda3D Scene]      ──▶ Dynamic Point Sprites (GPU)  │
        │  [Math Core]          ──▶ Continuous Projection p_i(c) │
        │  [Eigen Arrows]       ──▶ Scaled by sqrt(eigenvalue)   │
        │  [Projection Plane]   ──▶ Accurate Depth Buffer Slicing│
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                  ┌────────────────────────────────────────┐
                  │            LIVE DUAL VIEW              │
                  │  • Main: Panda3D 3D Spatial Window     │
                  │  • HUD: OpenCV Hand Tracker & Badges   │
                  └────────────────────────────────────────┘
```

---

## Documentation Map

Explore each facet of the system below:

| Guide | What's Inside? | Why You'll Care |
|---|---|---|
| [**Math Engine**](math_engine.md) | Eigendecomposition, sign stabilization, PCA box extents, and the continuous parameterization formula $p_i(c)$. | The mathematical proof that this is 100% rigorous linear algebra, not an animation gimmick. |
| [**Architecture**](architecture.md) | Asynchronous dual-loop architecture, 5 isolated coordinate spaces, and thread-safe snapshots. | How we guarantee a buttery 30+ FPS without webcam latency blocking the 3D renderer. |
| [**Interaction & Gestures**](interaction_and_gestures.md) | The C++ core-dump fix, selfie mirror chirality, the ratchet/clutch mechanism, and trackball physics. | How 1-hand gestures control rotation, zoom, and dimensionality compression without false triggers. |
| [**Rendering & Compositing**](rendering_and_compositing.md) | Panda3D scene graph, dynamic point sprites, depth-buffer sorting (bins 10 vs 30), and arrow scaling. | The secrets behind the clean, scientific, holographic turquoise aesthetic. |
| [**Quickstart & Controls**](quickstart_and_controls.md) | Environment setup, launch scripts, full keyboard cheatsheet, and gesture presentation guide. | Get up and running in under 2 minutes. |

---

## Key Design Principles

1. **Truth Over Visuals:** Every point position, arrow length, and box dimension represents an exact linear algebra calculation. Nothing is faked with canned animations.
2. **Velocity-Driven, Not Time-Driven:** If your hand stops, the projection freezes at that exact compression state $c$. If your hand reverses, the projection reverses. You control the mathematics.
3. **Decoupled Isolation:** The math core knows nothing about Panda3D. The gesture recognizer knows nothing about camera hardware. Everything can be tested headlessly with mocks.
