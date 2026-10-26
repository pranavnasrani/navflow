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

## 🎮 Macropad & Hardware Controls

| Control | Macropad / Keyboard Input | Action |
| :--- | :--- | :--- |
| **Z-Axis Height** | Keys `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `0` | Immediately jumps to discrete Z-height Step 1 to 10 across all 4 monitors |
| **Z-Step Nudge** | `Up` / `Down` Arrow keys, or `W` / `S` | Steps the circular pipe nozzle up or down |
| **Wind Speed Knob** | `Right` / `Left` Arrow keys, or `+` / `-`, `D` / `A` | Twists velocity rotary knob across 10 to 50 m/s with 0606 linear interpolation |
| **Mouse Wheel** | Scroll Up / Down | Turns velocity knob smoothly |

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

### 4. Circular Pipe Nozzle Geometry
The movable water-pipe nozzle on the inlet wall ($x = 0$) at height $z_k$ uses the **circle equation**:
$$(y - 0.5)^2 + (z - z_k)^2 \le R_{\text{pipe}}^2$$
Points inside the circle shoot air forward at $V_{\text{knob}}$; all other inlet points are closed.

### 5. Sustainability & The Cubic Power Law ($V^3$)
Aerodynamic drag power grows with the **cube of wind speed**:
$$P_{\text{drag}} = F_{\text{drag}} \times V = \frac{1}{2} \rho C_d A_{\text{front}} V^3$$
Demonstrating why streamlining cutting frontal area or drag coefficient saves massive fuel and $\text{CO}_2$ emissions!

---

## 🛠️ Regenerating CFD States & Using Custom 3D Models

### Using Custom 3D Models (.stl, .obj, .ply, .glb, .gltf)
Put each download in its own folder, one file per object:
- `assets/models/incoming/airplane/`
- `assets/models/incoming/f1_car/`
- `assets/models/incoming/building/`
- `assets/models/incoming/human/`

(or pass the file with `--src <file>`). `prepare_model.py` writes the tunnel-ready result to `assets/models/<object>/model.stl`.

Then prepare, simulate and import (see `CLAUDE.md` for the full command list):
```bash
./.venv/bin/python solver/prepare_model.py --object airplane      # (or add --src <file>) check the preview.png it writes
./.venv/bin/python solver/run_openfoam.py run --objects airplane --preview   # quick 3-direction check
./.venv/bin/python solver/run_openfoam.py run --objects airplane              # all 9 directions
./.venv/bin/python solver/import_results.py                                   # -> assets/cache/*.npz
```

### OpenFOAM
`solver/run_openfoam.py` runs OpenFOAM in Docker (resumable queue, per machine). `scripts/sync_to_mini.sh` and `scripts/pull_from_mini.sh` share the work with a second Mac. The previous homemade-solver exhibit is the git tag `v1-fallback`.
