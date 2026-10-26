# NavFlow: OpenFOAM Pipeline Design

**Date:** Monday 26 October 2026 (revised after Sameer's review; approved by Isaac 26 Oct) · **Exhibition:** Saturday 31 October 2026
**Owner:** Isaac · **Reviewers:** Sameer, Pranav

---

## 1. What this is and why

NavFlow is our 4-monitor wind-tunnel exhibit for the Math section. It has four objects (airplane, F1 car, skyscraper, human), one per monitor. Visitors change the wind and the screens react instantly, because every setting is solved in advance and the app only switches between stored results.

The repo already works end to end, but it uses a small home-made solver on a coarse grid. **This project replaces those results with real OpenFOAM simulations**, which give proper turbulence, wakes and vortices, the "dirty air" look, while keeping the instant switching. The exhibit is about the **Navier-Stokes equations**, so the screens must show how each change the visitor makes affects the terms of that equation.

**What success looks like on Saturday:** all four monitors show flow that looks believable and convincing (wakes behind the human and building, smooth flow over the plane, dirty air behind the F1 car). It responds instantly to the macropad and the iPad, and we can explain every step to a judge.

## 2. Goals, non-goals, anti-goals

**Goals**
- Real OpenFOAM results (simpleFoam + k-ω SST) for all 4 objects × 9 wind directions = **36 runs**.
- A proper wind tunnel: wind enters across the **whole inlet face**, not from a small nozzle.
- A fully automatic pipeline: drop in an STL, and the tunnel size, mesh, boundary layers and boundary conditions are all set up without hand-editing.
- Make the Navier-Stokes terms visible and live on screen (§5.7).
- Sustainability widgets tied to the UN SDGs (§5.8).
- An iPad remote as well as the macropad (§5.9).
- Runs that can be paused, resumed and stopped without losing work (§5.5).
- Use both Macs, and find out early if something looks wrong, not after a 2-day run.

**Non-goals**
- Live simulation during the exhibition. Everything is precomputed.
- Speed as a simulated control. Each object has one fixed speed (§4). The screens can still show what the forces would be at other speeds (§8).
- Supersonic or compressible flow (see the plane speed in §4).
- The two-building Venturi / downdraft setup (dropped by Isaac, 26 Oct).
- Simulating earthquakes or tsunamis. They aren't airflow. At most a static info card.
- Time-accurate (unsteady) vortex shedding. Steady simpleFoam shows the *average* wake, which is normal for this kind of run.
- Turning the exhibit into a web app. The display stays native PySide6 + PyVista (CLAUDE.md rule). The iPad page is only a remote control.

**Anti-goals (would count as failure even if it "works")**
- Running for 2+ days and only then seeing it looks bad.
- Saturday arrives with an exhibit that has nothing to show. The current results stay as a fallback until the new ones are checked.
- Any switching lag. Moving a control must stay under 1 ms, from the macropad or the iPad.
- Explaining it to judges wrongly (see §8).

## 3. Constraints

- **Deadline:** done by Thursday night, rehearse Friday, show Saturday.
- **Hardware:** Isaac's MacBook Pro: Apple M4, 4 performance + 6 efficiency cores, 16 GB (checked 26 Oct). Mac mini: assumed the same M4; Isaac to check with `sysctl -n machdep.cpu.brand_string hw.perflevel0.physicalcpu hw.memsize`. Each run uses the 4 performance cores.
- **Software:** OpenFOAM in Docker (or the native macOS build, if it tests faster, §5.1).
- **App:** 4 windows. The cache format in `assets/cache/<object>_states.npz`, read by `app/state_manager.py`, changes (direction instead of speed) but keeps the instant RAM lookup and the 0606 linear interpolation.
- **Comments in code:** keep explanations tied to IGCSE 0606 maths (CLAUDE.md rule).

## 4. Decisions made

**By Isaac and Sameer**
- OpenFOAM in a Docker container, official image, native Apple Silicon build.
- 4 windows, not 5.
- A wind-tunnel format: wind across the whole inlet face, not a single nozzle. This replaces the earlier nozzle question.
- k-ω SST turbulence model, and the turbulence colouring **stays** as a view on screen.
- **Dynamic convergence:** stop a run once things change by **less than 0.2% over the last 50 iterations** (§5.4). Cap at 1000 iterations, with a ~500-iteration preview first.
- Boundary layers where they matter, chosen to keep compute reasonable (§5.3).
- The tunnel size depends on each model, big enough to show the whole wake.
- Air density and viscosity are set per object.
- Robust pause / resume / stop and a final saved file per run.
- File writes minimised inside Docker.
- **Speeds fixed per object:** plane **102 m/s (Mach 0.3)**, F1 car 40 m/s, building **50 m/s** at roof height (a Category 3 hurricane; the breeze and other storm categories are shown by V² scaling), human 10 m/s.
- **Knob = wind direction**, buttons = smoke-wand height.
- **Building:** hurricane wind load.
- **iPad remote:** yes, alongside the macropad.
- No two-building setup.
- Use the Mac mini as a second machine.

## 5. How it works

### 5.1 OpenFOAM and the container

- Image: **`opencfd/openfoam-default:2512`**, pinned. Its Docker Hub listing has an **arm64** build (checked 26 Oct), so it runs natively on M4 with no Intel emulation. The repo's current image (`openfoam/openfoam10-paraview510`) is Intel-only and gets replaced.
- **Keeping file writes fast:** Docker on a Mac is slow at writing into Mac folders. So each run works inside a **Docker volume** (the container's own disk), not a Mac folder:
  - fields are written in binary;
  - a save happens only every 50 iterations, and only the last two saves are kept;
  - only the finished result gets copied back to the Mac.
- **No RAM disk:** with 16 GB, a RAM disk plus the solver plus the Docker VM could push the Mac into swap, which would be far slower. The disk volume is nearly as fast because writes are rare. *(Builder's call.)*
- **Native alternative:** the Docker image is already a native Apple Silicon build. A native macOS build (OpenFOAM.app) gets tried only if the first test shows Docker's file handling is too slow for the budget. *(Builder's call.)*
- Docker Desktop settings on each Mac: 4+ CPUs, ~8 GB memory.

### 5.2 Tunnel and mesh (automatic, from the STL)

1. **Tunnel size from the model's bounding box:** about 3 body lengths ahead, 8 behind and 3 to the sides and top. Objects that stand on the ground (car, building, human) sit on the floor. That keeps the walls from squeezing the flow and leaves room for the full wake. *(Builder's call; tuned at the test.)*
2. **Real size and real speed.** Cell count depends on the shape, not the size. Running at real size with real speeds gives the correct Reynolds number for free. The viewer scales everything to fit the screen.
3. **blockMesh** for the tunnel box, then **surfaceFeatureExtract + snappyHexMesh**: cut the model out of the air, refine near the surface and in the wake, and add boundary layers (§5.3). **checkMesh** must pass before any run.
4. **Changing the wind direction:** for every object the model is turned and re-meshed for each direction, so the tunnel and its boundary conditions never change. This adds ~10 minutes per run and keeps one code path. *(Builder's call.)*

### 5.3 Boundary conditions and boundary layers

| Boundary | Plane | F1 car | Building | Human |
|---|---|---|---|---|
| Inlet face | even wind at V, at the set angle | even wind at V, at the set yaw | real wind profile (slower near the ground), V at roof height | even wind at V |
| Outlet | air can flow back in (pressureInletOutletVelocity + totalPressure 0) | same | same | same |
| Floor | none (free air, slip) | **moving floor at V** (rolling road) | rough ground, wall function | slip floor |
| Sides and top | slip | slip | slip | slip |
| Model surface | noSlip, 3 boundary layers, wall functions | same | same | same |

- **Boundary layers:** 3 thin cell layers on each model's surface, sized so wall functions work. That adds ~30–40% cells. More layers mostly help predict exactly where flow separates, which doesn't change what visitors see, and would cost much more. *(Builder's call.)*
- **Floors:**
  - A moving floor has no boundary layer relative to the car, like the rolling road in a real F1 tunnel, so it needs no extra layers.
  - The building's ground uses a wall function with the real wind profile (atmBoundaryLayerInletVelocity), so it doesn't need layers either.
- Turbulence k, ω at the inlet: ~5% intensity, or from the wind profile for the building. nut: nutkWallFunction on walls, calculated elsewhere.
- **Air per object (config):**
  - Sea level, ρ = 1.225 kg/m³ and ν = 1.5×10⁻⁵ m²/s, for all four objects as simulated.
  - The plane's fuel widget uses cruise conditions (ρ ≈ 0.364 kg/m³ at 11 km, 250 m/s).

### 5.4 One run, and when it stops

- **simpleFoam + kOmegaSST**, in parallel on 4 processes.
- **Dynamic convergence:** a run stops when drag and lift **each change by less than 0.2% over the last 50 iterations**. Lift is measured against the drag's size, because for symmetric shapes lift sits near zero and any wobble would look huge. There's a minimum of 200 iterations so a run can't stop falsely early, and a cap of 1000 (500 in the preview). The batch runner checks this from OpenFOAM's `forces` output and tells OpenFOAM to save and stop. *(Builder's call.)*
- **Runs that never settle** (likely the human and building, whose wakes keep wobbling): they hit the cap and export the **average of the last 100 iterations** (fieldAverage). They're flagged in the log for Isaac to look at.
- **Head start:** each run starts from a quick potentialFoam solution, because each direction has its own mesh. *(Builder's call.)*
- Each run logs the iterations used, final residuals, drag and lift.

### 5.5 Pause, resume, stop, save

- **Saves:** every 50 iterations, keeping the last two (purgeWrite 2).
- **Stop:** finishes the current iteration, saves, then exits. It's done through the run's control file, which OpenFOAM re-reads while running.
- **Pause:** stop now and resume later. `docker pause` also works for a short break.
- **Resume:** continues from the last save (startFrom latestTime). No progress is lost beyond the last 50 iterations.
- **Batch queue:** a progress file lists every run as waiting, running, done or flagged. Restarting the batch skips finished runs and resumes the half-done one. A crash or reboot just continues.
- **Final output per run:** a VTK file with the final (or averaged) velocity, pressure and turbulence fields on the air and the model surface, plus a small JSON of metrics and convergence info. Intermediate saves are deleted. The VTK is kept so streamlines can be re-traced without re-running.

### 5.6 Batch runner and import

- One command per machine. The laptop runs `airplane, f1_car` and the Mac mini runs `building, human`.
- **Coarse-first:** directions 1, 5 and 9 run first (12 runs total). After those, a complete exhibit exists. The rest fill the gaps.
- **Crashes:** a run that crashes or blows up is marked failed with its log, and the batch moves on.
- Mac mini results come to the laptop over SSH (Remote Login) or a shared folder.
- **Import:** `solver/vtk_importer.py` turns each VTK into the cache keys:
  - surface pressure on the model;
  - streamlines from a **smoke wand** at 10 heights;
  - k along each streamline;
  - the Navier-Stokes term values (§5.7);
  - metrics.
- **Streamlines are computed at import, not in OpenFOAM,** so the 10 wand heights cost no extra runs.
- **Gap filling:** any unfinished direction uses the nearest finished one and is flagged. The app never crashes on a missing run.
- **Fallback:** the current working version is tagged `v1-fallback` in git before any change, so the old exhibit can be brought back with one command.

### 5.7 What visitors see (viewer)

**Controls**
- **Knob:** wind direction. In between the simulated angles, the 0606 linear interpolation blends surface pressure and metrics.
- **Buttons:** smoke-wand height (10 steps).

**Views (Tab key, or iPad buttons)**
1. **Pressure view:** model coloured by surface pressure, streamlines coloured by speed.
2. **Navier-Stokes view:** the equation on screen, with a live bar for each term at the current setting:
   - (v·∇)v, the air speeding up and turning;
   - −∇p/ρ, the pressure push;
   - ν_eff∇²v, viscous friction including k-ω SST turbulent mixing (ν_eff = ν + ν_t). With wall functions, the molecular ν alone is about 10⁻⁴ of the other terms everywhere. *(Controller ruling, 27 Oct.)*

   The air is coloured by whichever term is biggest. Judges see that friction only matters in the thin layer next to the surface (the boundary layer).
3. **Turbulence view (k-ω SST):** streamlines coloured by turbulence energy k.

- All three views come from saved results, so nothing is re-run. The `Tab` key is in the config but `monitor_window.py` has no handler yet, and the builder adds one.
- **Metrics from OpenFOAM:** drag and lift come from the `forces` function object. Reference speed = the object's wind speed V. Reference area = frontal area at that direction.

### 5.8 Sustainability widgets

- **F1 car and plane:** two widgets overlaid on the view.
  - **Fuel saved:** from drag power, using the existing fuel and CO₂ factors. SDG 13, Climate Action.
  - **Speed gain:** the extra top speed lower drag gives at the same engine power. Drag power = ½ρC_dAV³, so V_new = V × (C_dA_old / C_dA_new)^(1/3).
  - Comparisons are against the object's own worst (highest drag-power) simulated angle, so the saving changes as visitors turn the knob and is zero at the worst angle. The plane's widget is worked out at cruise. *(Controller ruling, 27 Oct. A fixed C_d = 1.0 reference gave an absurd 67,200 L/h for the plane.)*
- **Building:** an SDG 9 (Industry, Innovation and Infrastructure) widget for hurricane wind load. It shows the force on the tower and the bending moment at its base, at hurricane speed versus a 12 km/h breeze (forces scale with V²), and which wind direction is worst. Earthquake and tsunami get a static info card at most.

### 5.9 iPad remote

- The exhibit app runs a small local web server on the laptop. The iPad opens one page with:
  - a wind-direction dial;
  - a smoke-wand slider;
  - view buttons;
  - a short "what you're seeing" line.
- iPad commands go into the same input controller as the keyboard, so switching stays under 1 ms. The display stays a native app.
- **Network:** the laptop and iPad join the same network. A phone hotspot works with no internet needed, so we don't depend on venue Wi-Fi.
- The macropad stays plugged in as a backup. Both work at once.

## 6. Schedule with check-points

Each check-point is a "go / fix it" moment. Isaac looks at the result before the next stage starts. Viewer work (views, widgets, iPad remote) is built Tue–Thu in parallel with the runs, using test data.

| When | Stage | Check-point |
|---|---|---|
| **Mon night / Tue morning** | Test run: 1 object, 1 direction | Mesh clean, it settles, looks right in the viewer. Measure time per run, pick mesh size, and compare Docker vs native. |
| **Tue** | Preview: 4 objects × 3 directions, cap 500 | All 4 models mesh cleanly, wakes visible, timing fits. **Settings frozen after this.** |
| **Tue night → Wed** | Full run, both Macs, coarse-first, cap 1000 | **Wed evening:** all 36 done, so the exhibit is safe. |
| **Thu** | Import, final views and widgets, iPad on the real network | Thu night: freeze. |
| **Fri** | Full rehearsal on 4 monitors with macropad and iPad | Buffer for problems. |

**Time budget (estimate until the test):** 36 runs × ~45 min ≈ 27 machine-hours, which is ~14 hours across both Macs. There's room for reruns.

**If the test shows it's too slow, cut in this order:**
1. 5 directions instead of 9.
2. A coarser mesh.

## 7. Models

Isaac finds the models **by Tuesday morning**. Monday's test uses a cleaned-up starter model.

**What a good model file looks like**
- **Format:** `.stl` or `.obj` is best. `.step`/`.stp` (CAD) also works and gets converted.
- **Outside shape only:** no interiors, no separate seats or windows.
- **Closed surface, no holes:** models made for **3D printing** are almost always closed, so they're the safest choice.
- **Detail:** about 20,000–200,000 triangles and under ~50 MB.
- **Size, units and orientation don't matter:** the pipeline scales the model to real size, and the builder rotates it so the nose faces the wind and up is up.
- **Licence:** free to use, e.g. CC-BY. Note the author for the poster credit.

**Where to look**
- **Printables.com** and **Thingiverse**, which have 3D-print models. Search "A380", "F1 car", "Burj Khalifa", "mannequin".
- **NASA 3D Resources** (science.nasa.gov/3d-resources) for aircraft.
- **GrabCAD** for STEP files (free account).
- **Avoid** game models (`.fbx`, `.blend`, `.glb`). They're usually loose pieces with holes.

**Converting:** drop whatever was downloaded into `assets/models/incoming/<object>/`. The builder converts it, merges it into one closed surface, rotates it, and shows a preview image before it's used.
- By hand with **FreeCAD:** open STEP → Mesh Design → "Create mesh from shape" → export STL.
- By hand with **Blender:** open FBX/GLB → join parts (Ctrl+J) → export STL.

## 8. Explaining it to judges (accuracy notes)

- Incompressible Navier-Stokes solves for **velocity and pressure together**.
- **The equation on screen assumes air isn't squashed (incompressible).** That holds below about Mach 0.3 (~100 m/s). At Mach 1 shock waves form, and different equations (compressible flow, plus an energy equation) are needed. That's why the plane runs at Mach 0.3 (102 m/s), the edge of where the equation holds. The fuel widget applies the result to cruise-altitude air and notes that it leaves out shock-wave drag.
- **Why speed isn't a control:** at these Reynolds numbers the flow pattern barely changes with speed, and forces grow with V². Direction changes the pattern a lot, so it's the more interesting control.
- **k-ω SST** is a turbulence model added on top of the *averaged* Navier-Stokes equations (RANS). It estimates the effect of swirls too small for the mesh. The turbulence view shows its k.
- simpleFoam does **not** use the Lattice Boltzmann method.
- "Iterations" are repeated corrections toward the steady answer, not steps in time.
- A one-page explainer covering this goes next to the exhibit.

## 9. Left to the builder

- Exact refinement levels and layer sizes.
- The angle values for each object, e.g.:
  - plane angle of attack 0–16° (stall shows at the top end);
  - car yaw 0–16°;
  - building 0° up to its symmetry angle;
  - human 0–180°.
- The minimum iteration count and the turbulence inlet values.
- The exact Navier-Stokes term calculation.
- Widget reference values.
- iPad page layout.
- File and folder names.

All of these are recorded in the config.

## 10. What we need

- [ ] Docker Desktop open on **both** Macs, 4+ CPUs and ~8 GB memory
- [ ] Mac mini chip and memory checked (command in §3)
- [ ] Remote Login (SSH) on the Mac mini, or a shared folder
- [ ] Laptop plugged in with sleep off during runs
- [ ] Models by Tuesday morning (§7)
- [ ] An iPad, and a phone hotspot for the exhibition

---

### Please check

1. Does the workflow in §5 match how you'd set up the case?
2. Are the boundary conditions and boundary layers in §5.3 right?
3. Is the schedule in §6 realistic?
4. Anything in §8 you'd explain differently?
