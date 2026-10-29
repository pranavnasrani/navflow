# Interactive 4-Monitor Navier-Stokes CFD Exhibition System
**Math Section — STEM Exhibition**
*Aerodynamics, Navier-Stokes PDEs & Sustainability*

---

## 🚀 Quick Start Guide

### 1. Requirements & Setup
All dependencies run natively in Python 3.12:
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Running the Exhibition Application
```bash
# Development Mode (Opens all 4 synchronized windows on your current screen)
python3 app/main.py --dev-mode

# Exhibition Mode (Distributes 4 fullscreen windows across 4 physical monitors)
python3 app/main.py --monitor-mode

# Single Object Mode (Focused view on one object)
python3 app/main.py --object airplane   # Options: airplane, f1_car, building, human
```

---

## 🎮 Macropad, Keyboard & iPad Controls

Every visitor input drives all 4 monitors at once. There are two controls: the **smoke wand** (10 heights) and the **wind-direction knob** (9 solved directions per object, blended in between).

| Control | Macropad / keyboard input | Action |
| :--- | :--- | :--- |
| **Smoke wand height** | Keys `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `0` | Jumps the smoke wand to height 1 to 10; streamlines are traced from a rake of seeds at that height |
| **Wand nudge** | `Up` / `Down` arrow keys, or `W` / `S` | Steps the smoke wand up or down one height |
| **Wind-direction knob** | `Right` / `Left` arrow keys, `+` / `-` (also `=` / `_`), `D` / `A`, `]` / `[` | Turns the knob through the object's wind directions, with 0606 linear interpolation between solved angles |
| **Mouse wheel** | Scroll up / down | Turns the wind-direction knob smoothly (one notch = one detent) |
| **View** | `Tab` | Cycles the three views: surface pressure, dominant Navier-Stokes term, turbulence k |
| **iPad remote** | Open `http://<lan-ip>:8765/` in Safari | Touch page with the same wand, knob and view controls. The app prints the exact URL in the terminal and on each window (pass `--no-remote` to turn it off). The iPad must be on the same Wi-Fi or phone hotspot |

What the knob means differs per object: airplane = angle of attack, F1 car = crosswind (yaw), building = wind bearing, human = body angle to the wind.

---

## 📐 Mathematical Concepts (IGCSE Additional Mathematics 0606)

This project connects everyday IGCSE Additional Mathematics calculus and vectors directly to the Navier-Stokes equations:

### 1. 3D Velocity Vectors & Rates of Change
At every point in the wind tunnel $(x, y, z)$, air moves with a 3D velocity vector:
$$\mathbf{v} = \begin{pmatrix} u \\ v \\ w \end{pmatrix}, \quad \text{Speed } |\mathbf{v}| = \sqrt{u^2 + v^2 + w^2}$$

### 2. Incompressibility (Mass Conservation)
Air entering a tiny grid cube must equal air leaving it. Rates of change add to zero:
$$\frac{du}{dx} + \frac{dv}{dy} + \frac{dw}{dz} = 0$$

### 3. Newton's Second Law for Fluids ($m \cdot a = \sum \text{Forces}$)
$$\rho \underbrace{\left(\frac{du}{dt} + u\frac{du}{dx} + v\frac{du}{dy} + w\frac{du}{dz}\right)}_{\text{Chain Rule / Connected Rates of Change}} = \underbrace{-\frac{dp}{dx}}_{\text{Pressure Push}} + \underbrace{\mu \left(\frac{d^2u}{dx^2} + \frac{d^2u}{dy^2} + \frac{d^2u}{dz^2}\right)}_{\text{Air Friction (Second Derivatives)}}$$

### 4. The Smoke Wand and the Wind-Direction Knob
**Smoke-wand heights are an arithmetic progression.** The 10 wand heights run from $z_{\text{low}}$ to $z_{\text{high}}$ in equal steps:
$$z_k = z_{\text{low}} + (k - 1)\,d, \qquad d = \frac{z_{\text{high}} - z_{\text{low}}}{9}, \qquad k = 1, \dots, 10$$
At each height a row of seed points across the flow (a *rake*) is released upstream, and every streamline follows the air: it is the curve whose tangent at each point is the velocity vector, $\frac{dx}{dt} = u,\ \frac{dy}{dt} = v,\ \frac{dz}{dt} = w$.

**The knob is a straight-line map (linear function).** A knob position $k \in [0, 1]$ gives a wind direction
$$\theta = \theta_{\min} + k\,(\theta_{\max} - \theta_{\min})$$

**Linear interpolation between solved angles.** OpenFOAM solved 9 directions per object. For an angle $\theta$ between two solved neighbours $\theta_1 < \theta < \theta_2$:
$$t = \frac{\theta - \theta_1}{\theta_2 - \theta_1} \in [0, 1], \qquad \mathbf{y}_{\text{live}} = (1 - t)\,\mathbf{y}_1 + t\,\mathbf{y}_2$$
This blends surface pressure, forces, fuel burn and the Navier-Stokes terms smoothly while the knob turns. (Streamlines cannot be blended point by point, so they come from the nearer solved angle.)

**Rotating the wind vector.** The wind tunnel is fixed and the wind always blows along $+x$; to test another direction the *model* is rotated by $R(\theta)$ and re-meshed. To draw the result on the un-rotated model, the wind vector is rotated the other way, and since a rotation matrix is orthogonal, $R^{-1} = R^{T}$. For a rotation about the $y$-axis (airplane angle of attack):
$$\mathbf{w}_{\text{model}} = R_y(\theta)^{T}\begin{pmatrix}1\\0\\0\end{pmatrix} = \begin{pmatrix}\cos\theta\\0\\ \sin\theta\end{pmatrix}, \qquad R_y(\theta) = \begin{pmatrix}\cos\theta & 0 & \sin\theta\\ 0 & 1 & 0\\ -\sin\theta & 0 & \cos\theta\end{pmatrix}$$
(about the $z$-axis the vector is $(\cos\theta, -\sin\theta, 0)$). Between two solved angles the two unit vectors are blended with the same linear interpolation and then divided by their length so the arrow stays a unit vector.

### 5. Sustainability & The Cubic Power Law ($V^3$)
Aerodynamic drag power grows with the **cube of wind speed**:
$$P_{\text{drag}} = F_{\text{drag}} \times V = \frac{1}{2} \rho C_d A_{\text{front}} V^3$$
Demonstrating why streamlining cutting frontal area or drag coefficient saves massive fuel and $\text{CO}_2$ emissions!

---

## 🛠️ Regenerating CFD States & Using Custom 3D Models

### Using Custom 3D Models (.stl, .obj, .ply, .glb, .gltf, .blend, .fbx)
Put each download in its own folder, one file per object:
- `assets/models/incoming/airplane/`
- `assets/models/incoming/f1_car/`
- `assets/models/incoming/building/`
- `assets/models/incoming/human/`

(or pass the file with `--src <file>`). `prepare_model.py` writes the tunnel-ready result to `assets/models/<object>/model.stl`.
.blend, .fbx, .glb and .gltf files are exported by Blender first (`brew install --cask blender`). Display models are made of
many open parts, so add `--wrap 500` to shrink-wrap them into one closed skin (cube size = longest side / 500).
The current models: Singapore Airlines A380 (`--wrap 500 --drop gears`, landing gear left out for cruise),
Red Bull RB22 (`--wrap 500`), Burj Khalifa (`--wrap 400`).

Then prepare, simulate and import (see `CLAUDE.md` for the full command list):
```bash
./.venv/bin/python solver/prepare_model.py --object airplane --wrap 500 --drop gears   # check the preview.png it writes
./.venv/bin/python solver/run_openfoam.py run --objects airplane --preview   # quick 3-direction check
./.venv/bin/python solver/run_openfoam.py run --objects airplane              # all 9 directions
./.venv/bin/python solver/import_results.py                                   # -> assets/cache/*.npz
```

### OpenFOAM
`solver/run_openfoam.py` runs OpenFOAM in Docker (resumable queue, per machine). `scripts/sync_to_mini.sh` and `scripts/pull_from_mini.sh` share the work with a second Mac. The previous homemade-solver exhibit is the git tag `v1-fallback`.
