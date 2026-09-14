# Mathematical Engine & PCA Theory

> "Linear algebra is the mathematics of space, and PCA is how we find the truest directions in that space."

The **Live PCA Engine** does not treat Principal Component Analysis as an approximation or a visual trick. Every coordinate, vector length, and bounding plane in the engine is calculated through rigorous linear algebra.

---

## 1. Synthetic Data Generation

To guarantee that the demonstration is educational and reproducible, the engine initializes a deterministic 3D Gaussian distribution shaped like an elongated tri-axial ellipsoid.

### Mathematical Definition
Given a random seed (default: `42`), we sample $N$ independent standard Gaussian vectors:

$$\mathbf{u}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3), \quad i = 1, \dots, N$$

We stretch each axis by configured semi-axis lengths $(a_1, a_2, a_3) = (5.0, 3.0, 1.0)$:

$$\mathbf{s}_i = \mathbf{u}_i \odot \begin{bmatrix} 5.0 \\ 3.0 \\ 1.0 \end{bmatrix}$$

Because $a_1 > a_2 > a_3$, the dataset is guaranteed to have distinct variances across its principal directions, preventing degenerate or spherical distributions where eigenvectors are undefined.

### Correlation Rotation
To ensure the principal axes do not align trivially with the camera's world axes ($X, Y, Z$), we apply fixed 3D rotations:
- Pitch $\theta_x = 45^\circ$
- Yaw $\theta_y = 60^\circ$

$$\mathbf{R} = \mathbf{R}_y(60^\circ) \mathbf{R}_x(45^\circ)$$
$$\mathbf{X}_{raw} = \mathbf{S} \mathbf{R}^T + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2)$$

This produces an inclined, elongated 3D cloud in space ready for PCA analysis.

---

## 2. The PCA Computation Pipeline

The core PCA implementation lives in [`src/math_core/pca_engine.py`](../src/math_core/pca_engine.py). It executes 5 deterministic steps:

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Raw Points X   │ ──▶ │ Mean Centering X_c  │ ──▶ │ Covariance Matrix C  │
│     (N × 3)     │     │    X - μ            │     │ (1 / N-1) * X_c^T X_c│
└─────────────────┘     └─────────────────────┘     └──────────┬───────────┘
                                                               │
                                                               ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ Sorted & Signed │ ◀── │ Descending Sort     │ ◀── │ Symmetric Eig (eigh) │
│ Eigenvectors V  │     │ λ₁ ≥ λ₂ ≥ λ₃        │     │ C v_j = λ_j v_j      │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
```

### Step 1: Mean Centering
Compute the empirical centroid $\boldsymbol{\mu} \in \mathbb{R}^3$:

$$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i, \qquad \mathbf{X}_c = \mathbf{X} - \mathbf{1} \boldsymbol{\mu}^T$$

### Step 2: Sample Covariance Matrix
Because $\mathbf{X}_c$ is centered, the sample covariance matrix $\mathbf{C} \in \mathbb{R}^{3 \times 3}$ is:

$$\mathbf{C} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c$$

### Step 3: Symmetric Eigendecomposition (`np.linalg.eigh`)
Because $\mathbf{C}$ is symmetric ($\mathbf{C} = \mathbf{C}^T$) and positive semi-definite by construction, the engine strictly uses `np.linalg.eigh`:
- Guarantees **pure real eigenvalues** (no complex numerical residues).
- Guarantees **strictly orthonormal eigenvectors** ($\mathbf{V}^T \mathbf{V} = \mathbf{I}$).
- Numerically superior and faster than generic `np.linalg.eig`.

### Step 4: Descending Sort
Eigenvalues returned by `eigh` are sorted in ascending order. We invert the indexing to ensure:

$$\lambda_1 \ge \lambda_2 \ge \lambda_3$$

The eigenvectors are rearranged column-wise to match: $\mathbf{V} = [\mathbf{v}_1 \quad \mathbf{v}_2 \quad \mathbf{v}_3]$.

### Step 5: Deterministic Sign Convention
Eigendecomposition has inherent sign ambiguity: if $\mathbf{v}$ is an eigenvector, $-\mathbf{v}$ is equally valid. Without stabilization, eigenvectors would randomly flip $180^\circ$ between frames, causing jitter.

**The Rule:** For each column $j$, we identify the element with the maximum absolute magnitude and force its sign to be positive:

$$k = \arg\max_r |V_{rj}|, \qquad \text{if } V_{kj} < 0 \implies \mathbf{v}_j \leftarrow -\mathbf{v}_j$$

This matches `scikit-learn`'s convention and guarantees 100% visual stability.

---

## 3. The PCA Bounding Box

Unlike typical 3D game engines that use Axis-Aligned Bounding Boxes (AABB) or Oriented Bounding Boxes (OBB) fit through heuristics, our PCA Box is **purely mathematical**.

1. Project all centered points onto the principal basis:
   $$\mathbf{Z} = \mathbf{X}_c \mathbf{V} \quad \in \mathbb{R}^{N \times 3}$$
2. Compute the min and max coordinates along each principal component:
   $$\mathbf{z}_{min} = \min_{row} \mathbf{Z}, \qquad \mathbf{z}_{max} = \max_{row} \mathbf{Z}$$
3. Extents along PC1, PC2, PC3:
   $$\mathbf{L} = \mathbf{z}_{max} - \mathbf{z}_{min}$$
4. Box center in PCA space:
   $$\mathbf{c}_{pca} = \frac{\mathbf{z}_{min} + \mathbf{z}_{max}}{2}$$
5. Transform box center back to world space:
   $$\mathbf{c}_{world} = \boldsymbol{\mu} + \mathbf{V} \mathbf{c}_{pca}$$

The box tightly encases the distribution and is perfectly aligned with the eigenvector directions.

---

## 4. Continuous Compression & Projection ($c \in [0, 1]$)

The educational climax of the engine is the continuous flattening of the 3D cloud onto the 2D subspace defined by $\mathbf{v}_1$ and $\mathbf{v}_2$.

### The Parameterized Trajectory
Let $c \in [0.0, 1.0]$ be the compression state controlled by the presenter's hand.
For each point $\mathbf{x}_i$:

$$\mathbf{z}_i = \mathbf{V}^T (\mathbf{x}_i - \boldsymbol{\mu}) = [z_{i1}, z_{i2}, z_{i3}]^T$$

The target projection $\mathbf{p}_i^*$ onto the optimal 2D plane (discarding the smallest variance direction $\mathbf{v}_3$) is:

$$\mathbf{p}_i^* = \boldsymbol{\mu} + z_{i1} \mathbf{v}_1 + z_{i2} \mathbf{v}_2$$

The dynamic position at state $c$ is a continuous linear homotopy:

$$\mathbf{p}_i(c) = (1 - c) \mathbf{x}_i + c \mathbf{p}_i^*$$

```
   Original 3D Point x_i
          \
           \  Trajectory (Trajectory Guides fade in)
            \
             ▼
        p_i(c)  <── Hand velocity controls progress c in [0, 1]
             │
             │
             ▼
        Target 2D Subspace Point p_i* (c = 1.0)
   ══════════════════════════════════════════ [Plane formed by v_1 & v_2]
```

### Visual Synchronizations at State $c$
As $c$ advances from $0 \to 1$:
- **PC3 Vector Length:** $\text{length}_3(c) = \sqrt{\lambda_3} \cdot (1 - c)$. It shrinks into nothingness at $c=1.0$.
- **PCA Box Depth:** $L_3(c) = L_3 \cdot (1 - c)$. The 3D box flattens into a 2D rectangle.
- **Projection Guides:** Opacity peaks at $c = 0.5$ ($\alpha = 2c$ for $c \le 0.5$, $\alpha = 2(1-c)$ for $c > 0.5$), then fades to zero once projection is complete.
- **2D Plane:** Opacity increases proportionally to $c$, illuminating the final subspace.
