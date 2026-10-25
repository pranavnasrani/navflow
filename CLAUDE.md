# CLAUDE.md — Agent & Developer Guide

This repository contains an interactive, native 4-monitor 3D Computational Fluid Dynamics (CFD) simulation system for the **Math section of a STEM Exhibition**. It demonstrates the power of the **Navier-Stokes equations** applied to aerodynamics and sustainability.

---

## 🧭 Project Mission & Exhibition Context

1. **The Math Section Focus**:
   - Demonstrates the **3D Incompressible Navier-Stokes Equations**:
     $$\nabla \cdot \mathbf{v} = 0 \quad \text{(Mass Conservation)}$$
     $$\rho \left(\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v}\right) = -\nabla p + \mu \nabla^2 \mathbf{v} \quad \text{(Newton's 2nd Law for Fluids)}$$
   - Explained and documented using **IGCSE Additional Mathematics (0606)** concepts: 3D velocity vectors, rates of change & connected rates of change (chain rule), small increments ($\delta u / \delta x$), second derivatives for viscosity/friction, circle coordinate geometry for the pipe nozzle, and linear interpolation for the velocity knob.

2. **The Sustainability Link**:
   - **Aerospace (Monitor 1 - Airplane)**: Lift-to-drag ratio optimization, reducing induced drag and jet fuel / $\text{CO}_2$ emissions per flight hour.
   - **Motorsport & Automotive (Monitor 2 - F1 Car)**: Cutting frontal surface area ($A_{\text{front}}$) and managing ground effect downforce to slash the cubic aerodynamic drag power:
     $$P_{\text{drag}} = F_{\text{drag}} \cdot V = \frac{1}{2} \rho C_d A_{\text{front}} V^3$$
   - **Sustainable Architecture (Monitor 3 - Skyscraper)**: High-altitude wind loading, mitigating vortex shedding, and reducing structural material and HVAC wind infiltration.
   - **Benchmark Bluff Body (Monitor 4 - Human)**: Shows severe flow separation, massive turbulent wake eddies, and high drag of unstreamlined shapes compared to engineered geometries.

3. **Interactive Hardware Setup**:
   - **4 Independent Monitors**: One monitor window per object.
   - **1 USB HID Macropad**: Configured as a USB keyboard sending keystrokes that control all 4 monitors simultaneously:
     - **10 Buttons**: Discrete Z-height steps (1 through 10) moving a circular water-pipe inlet nozzle vertically.
     - **1 Rotary Knob**: Wind velocity adjustment across 10–50 m/s with continuous 0606 linear interpolation.

4. **Zero-Latency Architecture (<1 ms switching)**:
   - Live 3D CFD solving is too slow for 60fps joystick feedback.
   - We **pre-compute all 50 states per object** (10 Z-heights × 5 wind speeds = 50 states, 200 states total) and package them into compact `.npz` files in `assets/cache/`.
   - At runtime, `app/state_manager.py` loads the states into RAM. Pressing any button or turning the knob updates all 4 screens in **under 0.4 milliseconds**!

---

## 🛠️ Environment & Development Commands

### Python Environment
- Python version: **3.12+**
- Virtual environment: `.venv/`
- Dependencies in `requirements.txt`: `PySide6`, `pyvista`, `pyvistaqt`, `vtk`, `trimesh`, `rtree`, `numpy`, `scipy`, `matplotlib`.

### Common Commands

```bash
# Run the integration test suite (verifies <1ms RAM lookup, 0606 linear interpolation, solver divergence)
./.venv/bin/python3 tests/test_system.py

# Launch development mode (arranges all 4 monitor windows in a 2x2 grid on current screen)
./run_exhibition.sh
# or:
./.venv/bin/python3 app/main.py --dev-mode

# Launch production exhibition mode (spans 4 fullscreen windows across 4 physical monitors)
./run_exhibition.sh --monitor-mode
# or:
./.venv/bin/python3 app/main.py --monitor-mode

# Launch focused single-object window
./.venv/bin/python3 app/main.py --object airplane   # Options: airplane, f1_car, building, human

# Regenerate starter 3D models (.stl)
./.venv/bin/python3 assets/generate_starter_models.py

# Precompute CFD states (full 50 states x 4 objects)
./.venv/bin/python3 solver/batch_precompute.py

# Quick precomputation test (3 heights x 2 speeds)
./.venv/bin/python3 solver/batch_precompute.py --quick
```

---

## 📁 Repository Structure

```text
Exhibition/
├── CLAUDE.md                       # This guide for Claude Code and coding agents
├── README.md                       # General project documentation & math guide
├── run_exhibition.sh               # Convenience launch script
├── requirements.txt                # Python dependencies
├── config/
│   └── exhibition_config.json      # Master config: wind tunnel geometry, objects, keys, solver settings
├── assets/
│   ├── generate_starter_models.py  # Procedural watertight meshes (airplane, f1_car, building, human)
│   ├── models/                     # Custom .stl / .obj models can be dropped here
│   │   ├── airplane/model.stl
│   │   ├── f1_car/model.stl
│   │   ├── building/model.stl
│   │   └── human/model.stl
│   └── cache/                      # Precomputed .npz archives (450KB each, loaded into RAM)
│       ├── airplane_states.npz
│       ├── f1_car_states.npz
│       ├── building_states.npz
│       └── human_states.npz
├── solver/
│   ├── navier_stokes_3d.py         # From-scratch 3D Chorin Projection finite-difference solver
│   ├── mesh_voxelizer.py           # Voxelizes .stl/.obj meshes & extracts surface normals
│   ├── batch_precompute.py         # Solves parameter grid and generates .npz bundles
│   ├── vtk_importer.py             # Imports ANSYS Fluent & OpenFOAM .vtk/.vtu exports
│   └── openfoam_bridge.py          # Docker runner for OpenFOAM cases
├── app/
│   ├── main.py                     # Master Qt application launcher
│   ├── state_manager.py            # Zero-latency RAM cache manager & linear interpolation
│   ├── monitor_window.py           # Native PySide6 + PyVista 3D window & dark-mode HUD
│   └── input_controller.py         # Global USB HID Macropad & keyboard event dispatcher
└── tests/
    └── test_system.py              # Automated test suite
```

---

## 🧩 Key Architecture Details & Agent Guidelines

1. **Strict Native Requirement**:
   - The user has explicitly specified this must be a **100% native desktop application** (PySide6 + PyVista / VTK with Metal/OpenGL). Do **NOT** replace with or convert into a web app.

2. **Custom 3D Models**:
   - The system is completely model-agnostic. Anyone can drop new `.stl` or `.obj` files into `assets/models/<object_id>/model.stl`.
   - `solver/mesh_voxelizer.py` automatically normalizes, scales, and centers incoming meshes inside the wind tunnel test box $[0.6, 1.4] \times [0.2, 0.8] \times [0.1, 0.9]$.
   - After updating models, re-run `./.venv/bin/python3 solver/batch_precompute.py`.

3. **ANSYS Fluent & OpenFOAM Workflow**:
   - If importing ANSYS Fluent simulations from another computer: export as standard `.vtk` or `.vtu` containing `Velocity` (or `U`) and `Pressure` (or `p`), and run `solver/vtk_importer.py`.
   - If using OpenFOAM: `solver/openfoam_bridge.py` automates `blockMesh`, `simpleFoam`, and `foamToVTK` inside Docker.

4. **Input Handling & Keybindings**:
   - Defined in `config/exhibition_config.json` and handled by `app/input_controller.py`.
   - Number keys `1`–`9`, `0`: Direct Z-step jumps (1 to 10).
   - `Up` / `Down` or `W` / `S`: Step Z-height up/down.
   - `Right` / `Left` or `+` / `-` or Mouse Wheel: Turn velocity knob.

5. **Linear Interpolation Formula (0606)**:
   - When the knob is rotated to a speed $V$ between discrete precomputed speeds $V_1$ and $V_2$, `app/state_manager.py` computes:
     $$t = \frac{V - V_1}{V_2 - V_1} \in [0, 1]$$
     $$\mathbf{y}_{\text{live}} = (1 - t)\mathbf{y}_1 + t\mathbf{y}_2$$
   - This smoothly blends surface pressures, forces, fuel consumption, and telemetry in real time.

6. **Preserving Documentation & Pedagogical Integrity**:
   - When modifying solver or app code, preserve comments explaining how the code maps to IGCSE 0606 calculus and kinematics so students can explain every line to exhibition judges.
