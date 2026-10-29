# CLAUDE.md — Agent & Developer Guide

This repository contains an interactive, native 4-monitor 3D Computational Fluid Dynamics (CFD) simulation system for the **Math section of a STEM Exhibition**. It demonstrates the power of the **Navier-Stokes equations** applied to aerodynamics and sustainability.

---

## 🧭 Project Mission & Exhibition Context

1. **The Math Section Focus**:
   - Demonstrates the **3D Incompressible Navier-Stokes Equations**:
     $$\nabla \cdot \mathbf{v} = 0 \quad \text{(Mass Conservation)}$$
     $$\rho \left(\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v}\right) = -\nabla p + \mu \nabla^2 \mathbf{v} \quad \text{(Newton's 2nd Law for Fluids)}$$
   - Explained and documented using **IGCSE Additional Mathematics (0606)** concepts: 3D velocity vectors, rates of change & connected rates of change (chain rule), small increments ($\delta u / \delta x$), second derivatives for viscosity/friction, coordinate geometry for the smoke-wand rake, and linear interpolation for the wind-direction knob.

2. **The Sustainability Link**:
   - **Aerospace (Monitor 1 - Airplane)**: Lift-to-drag ratio optimization, reducing induced drag and jet fuel / $\text{CO}_2$ emissions per flight hour.
   - **Motorsport & Automotive (Monitor 2 - F1 Car)**: Cutting frontal surface area ($A_{\text{front}}$) and managing ground effect downforce to slash the cubic aerodynamic drag power:
     $$P_{\text{drag}} = F_{\text{drag}} \cdot V = \frac{1}{2} \rho C_d A_{\text{front}} V^3$$
   - **Sustainable Architecture (Monitor 3 - Skyscraper)**: High-altitude wind loading, mitigating vortex shedding, and reducing structural material and HVAC wind infiltration.
   - **Benchmark Bluff Body (Monitor 4 - Human)**: Shows severe flow separation, massive turbulent wake eddies, and high drag of unstreamlined shapes compared to engineered geometries.

3. **Interactive Hardware Setup**:
   - **4 Independent Monitors**: One monitor window per object.
   - **1 USB HID Macropad** (keyboard keystrokes) controlling all 4 monitors at once, plus an **iPad remote** (touch page served by the app over local HTTP; the display itself stays native):
     - **10 Buttons**: the **smoke wand** height (10 steps), a rake of streamline seeds you move through the flow.
     - **1 Rotary Knob**: **wind direction** across 9 solved directions per object, blended by 0606 linear interpolation. The direction is a different quantity per object (airplane = angle of attack, F1 car = crosswind yaw, building = wind bearing, human = body angle).
     - **Tab**: cycle the view (surface pressure / dominant Navier-Stokes term / turbulence k).
   - Each object runs at **one fixed speed** (airplane 102 m/s, F1 car 40, building 50 at roof height, human 10).

4. **Zero-Latency Architecture (<1 ms switching)**:
   - Results come from real **OpenFOAM** (simpleFoam, k-omega SST) runs, not from a live solver. Each object is solved for 9 directions; all 10 wand heights per direction are streamlines traced from that one solution.
   - `solver/import_results.py` packs each object into `assets/cache/<object>_states.npz` (cache format v2).
   - At runtime `app/state_manager.py` loads the cache into RAM. The state lookup (wand, knob or iPad input -> the numbers, pressure and streamlines to draw) takes **under 1 ms**; drawing the frame is separate.

---

## 🛠️ Environment & Development Commands

### Python Environment
- Python **3.12+**, virtual environment `.venv/`, dependencies in `requirements.txt` (`PySide6`, `pyvista`, `pyvistaqt`, `vtk`, `trimesh`, `rtree`, `numpy`, `scipy`, `matplotlib`).
- OpenFOAM runs need **Docker** (image `opencfd/openfoam-default:2512`, volume `navflow-cases`, 4 MPI processes): Docker Desktop on the laptop, **Colima** on the Mac mini (8 CPU / 12 GB; after a reboot run `colima start` there). The app itself does not need Docker.
- Model conversion needs **Blender** (`brew install --cask blender`) for .blend/.fbx/.glb files.

### Common Commands

```bash
# Tests (the suite is expected to pass without Docker; tests marked `docker` need Docker Desktop)
./.venv/bin/python -m pytest -q -m "not docker"

# Launch (dev: 2x2 grid on this screen; --monitor-mode: 4 fullscreen windows on 4 monitors)
./run_exhibition.sh                      # = app/main.py --dev-mode
./run_exhibition.sh --monitor-mode
./.venv/bin/python app/main.py --object airplane   # airplane | f1_car | building | human
./.venv/bin/python app/main.py --no-remote         # skip the iPad remote

# 1. Prepare a downloaded model (STL/OBJ/PLY, or GLB/GLTF/BLEND/FBX via headless Blender) -> assets/models/<id>/model.stl + preview.png
#    --wrap RES shrink-wraps display models made of open parts (cube = longest side / RES); --drop skips Blender objects by name prefix.
#    Current models: airplane --wrap 500 --drop gears, f1_car --wrap 500, building --wrap 400 (sources in assets/models/incoming/, git-ignored).
#    Check preview.png: nose faces the wind (+x), up is up.
./.venv/bin/python solver/prepare_model.py --object airplane --src ~/Downloads/plane.glb

# 2. Run OpenFOAM (resumable; Ctrl-C or `stop` saves and exits; same command resumes)
./.venv/bin/python solver/run_openfoam.py test --object airplane --direction-index 0 --iterations 60 --coarse   # pipeline check
./.venv/bin/python solver/run_openfoam.py run --objects airplane f1_car building human --preview   # 3 directions, cap 500, runs/preview
./.venv/bin/python solver/run_openfoam.py run --objects airplane f1_car building human             # all 9 directions, cap 1000, runs/full
./.venv/bin/python solver/run_openfoam.py run --objects airplane --retry-failed   # failed runs are skipped until reset to waiting
./.venv/bin/python solver/run_openfoam.py status [--machine mini]   # one row per run (all machines unless --machine)
./.venv/bin/python solver/run_openfoam.py stop       # ask the running batch to save and stop

# 3. Import finished runs into the display caches (safe to re-run as more runs finish;
#    directions not yet run are filled from the nearest finished one, and the report says which)
./.venv/bin/python solver/import_results.py [--root runs/full] [--objects airplane] [--out-dir assets/cache]
```

### iPad remote
`app/main.py` serves a touch page on `controls.remote_port` (8765) and prints the URL (`http://<lan-ip>:8765/`) in the terminal and on each window. Open it in Safari on any device on the same Wi-Fi or phone hotspot. If the port is busy the app prints a warning and runs without the remote. The server lives in `app/remote_server.py`; the page is the single file `app/remote/index.html`.

### Two machines (laptop + Mac mini)
Each machine keeps its own queue file `runs/full/queue_<machine>.json`, chosen with `--machine` (default `laptop`). **On the mini always pass `--machine mini`** (for `run`, `stop` and `status`), e.g. `run_openfoam.py run --objects building human --machine mini`; `pull_from_mini.sh` also refuses to copy `queue_laptop.json`, so the laptop's queue is never overwritten.
```bash
./scripts/sync_to_mini.sh     # rsync the project to $MINI_HOST (default the mini's Tailscale IP 100.114.120.115), excluding .venv, runs, .git
./scripts/pull_from_mini.sh   # rsync the mini's runs/full/ back into runs/full/, then run import_results.py
```

### Old exhibit
The previous design (homemade finite-difference solver, pipe nozzle, wind-speed knob, 50 states per object) is preserved at the git tag **`v1-fallback`**: `git checkout v1-fallback`. The current app refuses to start on a v1 cache and prints this hint.

---

## 📁 Repository Structure

```text
navflow/
├── CLAUDE.md                       # This guide
├── README.md                       # General project documentation & math guide
├── run_exhibition.sh               # Launch script (passes flags through to app/main.py)
├── requirements.txt
├── pytest.ini
├── config/
│   └── exhibition_config.json      # openfoam settings, per-object cfd blocks, controls (remote_port, knob_step)
├── assets/
│   ├── generate_starter_models.py  # Procedural placeholder meshes
│   ├── models/<id>/model.stl       # Prepared, tunnel-ready models (+ preview.png)
│   └── cache/<id>_states.npz       # Format-v2 result caches loaded into RAM
├── scripts/
│   ├── sync_to_mini.sh             # rsync project -> Mac mini
│   └── pull_from_mini.sh           # rsync runs/full from the Mac mini
├── solver/
│   ├── prepare_model.py            # Model -> closed, scaled, oriented STL + preview (--wrap, Blender import)
│   ├── blender_export.py           # Run inside headless Blender: .blend/.fbx/.glb -> one joined STL
│   ├── run_openfoam.py             # CLI: run / stop / status / test
│   ├── import_results.py           # CLI: finished runs -> assets/cache/*.npz
│   ├── navier_stokes_3d.py         # Small from-scratch Chorin solver, kept for teaching and its test
│   └── foam/                       # settings, geometry, case_writer (+templates/), convergence, queue,
│                                   # runner, importer, ns_terms, sustainability
├── app/
│   ├── main.py                     # Qt launcher (checks caches, starts the iPad remote)
│   ├── cache_format.py             # Cache format v2 read/write
│   ├── state_manager.py            # RAM cache + linear interpolation
│   ├── input_controller.py         # Keyboard / macropad -> (wand, knob, view) state
│   ├── remote_server.py            # iPad remote HTTP server; app/remote/index.html is the page
│   ├── monitor_window.py           # Native PySide6 + PyVista window (three views, N-S bars, SDG widgets)
│   └── hud_widgets.py              # HUD building blocks
├── tests/                          # pytest suite
└── runs/                           # OpenFOAM case results (git-ignored): preview/, full/, test/
```

---

## 🧩 Key Architecture Details & Agent Guidelines

1. **Strict Native Requirement**:
   - The display is a **100% native desktop application** (PySide6 + PyVista / VTK with Metal/OpenGL). Do **NOT** replace it with or convert it into a web app. The iPad page is only a remote control.

2. **Models and directions**:
   - Models are prepared with `solver/prepare_model.py` (base rotation, scale and placement are applied once, when it writes `model.stl`; everything that reloads that file uses `geometry.load_prepared`, which does not rotate again). The model is then **rotated and re-meshed for every direction**; the tunnel stays axis-aligned with wind along +x. Per-object speed, real size, direction list, floor type and widgets are in `config/exhibition_config.json`.
   - Fuel/CO2: each object is compared with its own worst solved direction ("saved"), except the airplane, which has a cruise `mass_kg`: level flight means drag = W / (L/D), and it is compared with its best-L/D angle ("extra"). The airplane also meshes two levels finer (`surface_refinement` [6, 7], ~1.9 M cells).
   - Air is sea-level for all runs (rho = 1.225 kg/m^3, nu = 1.5e-5 m^2/s). Convergence: stop when drag and lift each change by <0.2% over the last 50 iterations (min 200, cap 1000); a run that hits the cap exports 100-iteration fieldAverage fields and is marked `flagged`.

3. **Cache format v2** (`app/cache_format.py`, `assets/cache/<obj>_states.npz`):
   - One file per object, `format_version = 2`. 9 direction states `d0..d8`, each with surface pressure, Navier-Stokes terms, metrics, and 10 wand streamline sets `d<i>_w<j>_*` (lines, turbulent k, dominant term). The model mesh is stored once in its own frame.
   - v1 caches (keys `z_steps`, `wind_speeds`) are rejected by `read_cache`; `app/main.py` prints the hint and exits 1.

4. **Input handling & keybindings** (hard-coded in `app/input_controller.py`; only `controls.remote_port` and `controls.knob_step` come from `config/exhibition_config.json`):
   - `1`-`9`, `0`: smoke wand to height 1 to 10.
   - `Up` / `Down` or `W` / `S`: step the wand up/down.
   - Wind-direction knob, clockwise (next direction): `Right`, `D`, `+`, `=`, `]`. Anticlockwise: `Left`, `A`, `-`, `_`, `[`. The mouse wheel also turns it (one notch = one detent of `knob_step`).
   - `Tab`: cycle view. `Space`: simple / detailed instrument strip (starts simple). `R`: reset every camera.
   - Camera: drag to turn; trackpad pinch or Cmd + scroll to zoom (plain scroll stays the knob). There is no key to quit.
   - Smoke shows at all 10 wand heights: the selected height bright, the other nine faint (`ObjectStateManager.wand_lines`).

5. **Linear interpolation (0606)**:
   - The knob maps to a direction by a straight line, d = d_min + knob (d_max - d_min). Between solved directions d1 < d < d2, `app/state_manager.py` computes t = (d - d1)/(d2 - d1) and y = (1 - t) y1 + t y2, blending surface pressure, N-S terms and telemetry live.

6. **Preserving Documentation & Pedagogical Integrity**:
   - When modifying solver or app code, preserve and extend comments explaining how the code maps to IGCSE 0606 maths (linear interpolation, V^2 and V^3 scaling, cube roots, the chain rule, interval bisection for the run order) so students can explain every line to exhibition judges.
