# NavFlow OpenFOAM Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the home-made solver's cached results with real OpenFOAM (simpleFoam + k-ω SST) wind-tunnel runs: 4 objects × 9 wind directions. Show them in the native 4-monitor viewer with three views (pressure, Navier-Stokes terms, turbulence), sustainability widgets and an iPad remote.

**Architecture:** A new package `solver/foam/` turns each STL into an automatic OpenFOAM case. A resumable batch runner drives that case inside a long-lived Docker container working on a Docker volume, and checks our own 0.2%/50-iteration convergence rule. An importer converts each finished run's VTK into a v2 `.npz` cache. The cache holds the model-frame surface pressure, smoke-wand streamlines with k and the dominant N-S term per point, and metrics. The app's state manager, input controller and monitor window are moved from (nozzle height, speed) to (wand height, wind direction), and a small stdlib HTTP server feeds iPad commands into the same controller.

**Tech Stack:** Python 3.12, numpy, scipy, trimesh + manifold3d, pyvista/vtk, PySide6 + pyvistaqt, pytest, Docker Desktop with `opencfd/openfoam-default:2512` (arm64).

**Spec:** `docs/superpowers/specs/2026-10-26-openfoam-pipeline-design.md`

## Global Constraints

- Python 3.12 in `.venv/`. Run tests with `./.venv/bin/python -m pytest`.
- The display stays a native PySide6 + PyVista app. The iPad page is only a remote control (CLAUDE.md rule).
- Preserve and extend comments that map code to IGCSE 0606 maths: linear interpolation, V² and V³ scaling, cube roots, the chain rule in derivatives (CLAUDE.md rule).
- Docker image `opencfd/openfoam-default:2512`, Docker volume `navflow-cases`, 4 MPI processes per run.
- Convergence: stop when drag and lift each change by **less than 0.2% over the last 50 iterations**, lift measured against max(|mean lift|, |mean drag|). Minimum 200 iterations. Cap 1000 (preview 500).
- A run that hits the cap exports the fieldAverage fields (window 100) and is marked `flagged`.
- Saves: every 50 iterations, `purgeWrite 2`, `writeFormat binary`.
- Fixed speeds:
  - airplane 102 m/s (Mach 0.3);
  - f1_car 40 m/s;
  - building 50 m/s at roof height;
  - human 10 m/s.
- Sea-level air for all runs: ρ = 1.225 kg/m³, ν = 1.5×10⁻⁵ m²/s.
- 9 wind directions per object, 10 smoke-wand heights.
- State lookup from RAM must stay under 1 ms, from keyboard, macropad or iPad.
- The model is rotated and re-meshed for every direction. The tunnel is always axis-aligned, with wind along +x.
- Commit after each task on branch `feat/openfoam-pipeline`. Never push.

## Review Focus

1. **A downloaded model with holes or in many overlapping pieces.** Expected: `prepare_model` either unions the pieces into one closed surface or stops with a message naming the problem. It never writes a broken STL that snappyHexMesh silently meshes wrong. (Task 3)
2. **A run that crashes or blows up (FOAM FATAL, floating point exception, NaN forces).** Expected: that run is marked `failed` with its log path kept, and the batch moves on to the next run. (Task 6)
3. **The laptop sleeps, reboots or Docker restarts mid-run.** Expected: rerunning the same command resumes the interrupted run from its last save, or restarts it cleanly if the volume lost it. No finished run is repeated. (Tasks 5, 6)
4. **Lift that sits near zero** (building or human head-on, car at 0° yaw). Expected: the convergence rule still triggers once drag is steady, instead of running to the cap because of a tiny lift wobble. (Task 5)
5. **The iPad sends nonsense or the Wi-Fi drops.** Expected: bad commands get HTTP 400 and change nothing, out-of-range values are clamped, and the macropad keeps working throughout. (Task 11)

---

## File Structure

| Path | Responsibility |
|---|---|
| `config/exhibition_config.json` | Modify: add `openfoam` block and a `cfd` block per object; remove `wind_tunnel`, `solver_defaults` |
| `solver/foam/__init__.py` | Package marker |
| `solver/foam/settings.py` | `FoamSettings`, `ObjectCfd`, `load_settings()` |
| `solver/foam/geometry.py` | Rotation, model preparation, tunnel bounds, frontal area, wand heights, seed rakes, view bounds |
| `solver/prepare_model.py` | CLI: `assets/models/incoming/<obj>/*` → `assets/models/<obj>/model.stl` + preview PNG |
| `solver/foam/case_writer.py` + `solver/foam/templates/*.tmpl` | Writes a complete OpenFOAM case for one run |
| `solver/foam/convergence.py` | Parses `force.dat` and applies the 0.2% rule |
| `solver/foam/queue.py` | Run records, coarse-first order, atomic JSON progress file |
| `solver/foam/runner.py` | Docker container, one-run lifecycle, batch loop, stop flag |
| `solver/run_openfoam.py` | CLI: `run`, `stop`, `status`, `test` |
| `app/cache_format.py` | v2 cache key names, `write_cache`, `read_cache` |
| `solver/foam/ns_terms.py` | Navier-Stokes term magnitudes from U, p |
| `solver/foam/sustainability.py` | Drag power, fuel/CO₂, speed gain, hurricane wind-load table |
| `solver/foam/importer.py` + `solver/import_results.py` | Results → v2 cache with gap filling |
| `app/state_manager.py` | Rewrite: v2 cache, (wand, knob) lookup with 0606 interpolation |
| `app/input_controller.py` | Rewrite: wand, knob, view mode |
| `app/hud_widgets.py` | N-S term bars, sustainability overlays, polyline helper |
| `app/monitor_window.py` | Modify: new HUD, wind arrow, wand rod, three views |
| `app/remote_server.py` + `app/remote/index.html` | iPad remote |
| `app/main.py` | Modify: wiring, remote server, no auto-precompute |
| `tests/fake_cache.py` | Writes a synthetic v2 cache for tests and UI development |
| `tests/test_*.py` | One test file per module above |
| `pytest.ini` | testpaths, `docker` marker |
| `scripts/sync_to_mini.sh`, `scripts/pull_from_mini.sh` | rsync code to and results from the Mac mini |
| Delete | `solver/batch_precompute.py`, `solver/vtk_importer.py`, `solver/openfoam_bridge.py`, `solver/mesh_voxelizer.py` (kept in git history and the `v1-fallback` tag) |

---

### Task 1: Environment, fallback tag, Docker probe

**Files:**
- Create: `pytest.ini`
- Modify: `requirements.txt` (add `pytest>=8.0`, `manifold3d>=2.4`)

**Interfaces:**
- Produces: `FOAM_BASHRC` value and whether the image runs as root, recorded in the Task 6 constant. (Probed 26 Oct: image runs as root, `/usr/lib/openfoam/openfoam2512`, `simpleFoam` on PATH in a login shell, linuxARM64 build.)

- [ ] **Step 1:** `git tag v1-fallback && git checkout -b feat/openfoam-pipeline`.
- [ ] **Step 2:** `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt` (with the two new lines added).
- [ ] **Step 3:** Write `pytest.ini`: `[pytest]`, `testpaths = tests`, `markers = docker: needs Docker Desktop running with the OpenFOAM image`.
- [ ] **Step 4:** Run: `./.venv/bin/python -m pytest tests/test_system.py -q`. Expected: 3 passed (baseline on the v1 cache).
- [ ] **Step 5:** `docker pull opencfd/openfoam-default:2512`. Then: `docker run --rm opencfd/openfoam-default:2512 bash -lc 'id -u; ls /usr/lib/openfoam; ls $FOAM_TUTORIALS/incompressible/simpleFoam/motorBike/system'`.
  - Expected: prints a uid and `openfoam2512`, and lists `fvSchemes fvSolution snappyHexMeshDict …`.
  - Record `FOAM_BASHRC = "/usr/lib/openfoam/openfoam2512/etc/bashrc"`, or whatever the listing shows.
- [ ] **Step 6:** `docker cp` the motorBike `system/fvSchemes` and `system/fvSolution` out (via `docker create` + `docker cp`) into `solver/foam/templates/` as the starting point for Task 4.
- [ ] **Step 7:** Commit: `git add pytest.ini requirements.txt solver/foam/templates && git commit -m "chore: venv, pytest, OpenFOAM image probe"`.

### Task 2: Settings and config

**Files:**
- Create: `solver/foam/__init__.py`, `solver/foam/settings.py`
- Modify: `config/exhibition_config.json`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces:
  - `FoamSettings`, a frozen dataclass with fields:
    - `image: str`, `volume: str`, `n_procs: int`;
    - `max_iterations: int`, `preview_iterations: int`, `min_iterations: int`;
    - `convergence_window: int`, `convergence_tolerance: float`;
    - `write_interval: int`, `average_window: int`;
    - `domain_factors: dict[str, float]` (keys `upstream downstream lateral top`);
    - `base_cells_per_size: int`, `surface_refinement: tuple[int, int]`, `wake_refinement: int`, `n_layers: int`;
    - `wand_count: int`, `seeds_per_wand: int`.
  - `ObjectCfd`, a frozen dataclass with fields:
    - `obj_id: str`, `speed_m_s: float`;
    - `real_size_m: float`, `size_axis: str` (`"x"` or `"z"`);
    - `base_rotation_deg: tuple[float, float, float]`;
    - `rotation_axis: str` (`"y"` or `"z"`), `directions_deg: tuple[float, ...]`, `direction_label: str`;
    - `floor: str` (`"none" | "moving" | "rough" | "slip"`);
    - `density_kg_m3: float`, `kinematic_viscosity_m2_s: float`;
    - `reference_cd: float`;
    - `cruise: dict | None` (`{"density_kg_m3", "speed_m_s"}`);
    - `abl_z0_m: float | None`;
    - `widgets: tuple[str, ...]`;
    - `surface_refinement: tuple[int, int] | None` (per-object override).
  - `load_settings(config_path: Path | str = "config/exhibition_config.json") -> tuple[FoamSettings, dict[str, ObjectCfd], dict]`. The third element is the raw config.

- [ ] **Step 1: Write the failing test**

```python
def test_load_settings_values():
    s, objs, raw = load_settings("config/exhibition_config.json")
    assert s.image == "opencfd/openfoam-default:2512" and s.volume == "navflow-cases"
    assert (s.n_procs, s.max_iterations, s.preview_iterations, s.min_iterations) == (4, 1000, 500, 200)
    assert (s.convergence_window, s.convergence_tolerance, s.write_interval, s.average_window) == (50, 0.002, 50, 100)
    assert s.wand_count == 10
    assert [objs[k].speed_m_s for k in ("airplane", "f1_car", "building", "human")] == [102.0, 40.0, 50.0, 10.0]
    assert all(len(o.directions_deg) == 9 for o in objs.values())
    assert objs["f1_car"].floor == "moving" and objs["building"].floor == "rough" and objs["airplane"].floor == "none"
    assert objs["airplane"].cruise == {"density_kg_m3": 0.364, "speed_m_s": 250.0}
    assert objs["airplane"].widgets == ("fuel", "speed") and objs["building"].widgets == ("wind_load",)

def test_load_settings_rejects_unknown_floor(tmp_path): ...  # floor "lava" -> ValueError naming the object
```

- [ ] **Step 2:** Run `./.venv/bin/python -m pytest tests/test_settings.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Edit the config. Add `"openfoam"` with the Global Constraints values, plus:
  - `domain_factors {upstream 3, downstream 8, lateral 3, top 3}`;
  - `base_cells_per_size 6`, `surface_refinement [4, 5]`, `wake_refinement 2`, `n_layers 3`, `seeds_per_wand 24`.

  Give each object a `cfd` block:

| obj | real_size_m / axis | rotation_axis | directions_deg | label | floor | extra |
|---|---|---|---|---|---|---|
| airplane | 72.7 / x | y | 0,2,…,16 | Angle of attack | none | cruise {0.364, 250}, widgets fuel,speed |
| f1_car | 5.6 / x | z | 0,2,…,16 | Crosswind (yaw) | moving | widgets fuel,speed |
| building | 828 / z | z | 0,7.5,…,60 | Wind direction | rough | abl_z0_m 0.5, widgets wind_load |
| human | 1.75 / z | z | 0,22.5,…,180 | Body angle to wind | slip | widgets none |

  All objects get `reference_cd 1.0`, `base_rotation_deg [0,0,0]` (set per real model in Task 13), and sea-level ρ, ν. Also:
  - building: add `"info_card"` text on earthquakes and tsunamis being structural/water problems, not airflow;
  - controls: add `"remote_port": 8765` and `"knob_step": 0.03125`;
  - delete `wind_tunnel` and `solver_defaults`.
- [ ] **Step 4:** Implement `load_settings`. Validate `floor` and `rotation_axis`, and require 9 directions in ascending order. Raise `ValueError` with the object id on bad input.
- [ ] **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6:** Commit `feat: openfoam settings and per-object cfd config`.

### Task 3: Geometry and model preparation

**Files:**
- Create: `solver/foam/geometry.py`, `solver/prepare_model.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: `ObjectCfd` (Task 2).
- Produces:
  - `rotation_matrix(axis: str, deg: float) -> np.ndarray`: 3×3, right-hand rule.
  - `class ModelNotWatertight(ValueError)`.
  - `prepare_model(src: Path, cfd: ObjectCfd) -> trimesh.Trimesh`. Steps:
    1. load (STL/OBJ/PLY/GLB/GLTF);
    2. apply `base_rotation_deg` (x, then y, then z);
    3. if not one watertight body, split and `trimesh.boolean.union(..., engine="manifold")`;
    4. if still not watertight, raise `ModelNotWatertight` with the hole count;
    5. scale so the extent along `size_axis` equals `real_size_m`;
    6. place the bbox centre at x = y = 0; z bottom = 0 when `floor != "none"`, else z centred at 0.
  - `run_geometry(model0, cfd, deg) -> trimesh.Trimesh`: a copy rotated by `rotation_matrix(cfd.rotation_axis, deg)` about the origin.
  - `tunnel_bounds(model, factors, floor) -> tuple[float, float, float, float, float, float]` (xmin, xmax, ymin, ymax, zmin, zmax). D = max bbox extent.
    - x: `[min − up·D, max + down·D]`;
    - y: `[min − lat·D, max + lat·D]`;
    - z: `[0, max + top·D]` with a floor, else `[min − top·D, max + top·D]`.
  - `frontal_area(model, resolution: int = 400) -> float`: silhouette on the y-z plane, rasterised.
  - `wand_heights(model0, count) -> np.ndarray`: `count` evenly spaced z from `zmin − 0.1H` to `zmax + 0.1H`.
  - `seed_rake(run_model, z, n) -> np.ndarray (n, 3)`: x = xmin − 0.5·Lx, y spanning `[ymin − 0.1W, ymax + 0.1W]`, the given z.
  - `view_bounds(model0) -> tuple6`: x `[xmin − 1L, xmax + 3L]`, y `[ymin − 1W, ymax + 1W]`, z `[max(zmin − 0.5H, floor), zmax + 0.5H]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_rotation_matrix_right_hand():
    assert np.allclose(rotation_matrix("z", 90) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(rotation_matrix("y", 90) @ [1, 0, 0], [0, 0, -1])

def test_angle_of_attack_lifts_nose(box_cfd_airplane):   # nose = xmin end
    m0 = prepare_model(stl_box(4, 1, 1), box_cfd_airplane)
    nose0 = m0.vertices[m0.vertices[:, 0].argmin()]
    m10 = run_geometry(m0, box_cfd_airplane, 10.0)
    assert m10.vertices[m10.vertices[:, 0].argmin()][2] > nose0[2]

def test_prepare_model_scales_and_places_on_floor(): ...   # 2x1x1 box, real_size 10 along x -> extents (10,5,5), zmin == 0, x/y centred
def test_prepare_model_unions_overlapping_parts(): ...     # two overlapping boxes in one STL -> watertight, body_count == 1
def test_prepare_model_rejects_open_surface(): ...         # box with one face deleted -> ModelNotWatertight
def test_tunnel_bounds_floor_and_free_air(): ...           # unit cube: floor -> zmin 0, zmax 4; free air -> zmin -3.5; x span [-3.5, 8.5]
def test_frontal_area_of_box(): ...                        # 4x2x1 box along x -> approx 2.0 within 2%
def test_wand_and_rake_shapes(): ...                       # 10 heights ascending; rake (24,3), all x < model xmin
```

- [ ] **Step 2:** Run `pytest tests/test_geometry.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `geometry.py`.
- [ ] **Step 4:** Implement the CLI: `solver/prepare_model.py --object <id> [--src <file>]`.
  - The default src is the single file in `assets/models/incoming/<id>/`.
  - It writes `assets/models/<id>/model.stl` and an off-screen pyvista preview `assets/models/<id>/preview.png` with the wind arrow along +x.
  - It prints the extents and triangle count.
  - STEP/FBX inputs exit with the FreeCAD/Blender instructions from spec §7.
- [ ] **Step 5:** Run the tests. Expected: PASS. Then run `./.venv/bin/python solver/prepare_model.py --object airplane --src assets/models/airplane/model.stl`. Expected: it either writes the model and preview, or fails with the hole/union message. Report which.
- [ ] **Step 6:** Commit `feat: model preparation and tunnel geometry`.

### Task 4: Case writer

**Files:**
- Create: `solver/foam/case_writer.py`, `solver/foam/templates/{blockMeshDict,snappyHexMeshDict,surfaceFeatureExtractDict,controlDict,decomposeParDict,transportProperties,turbulenceProperties,U,p,k,omega,nut}.tmpl` (plus `fvSchemes`, `fvSolution` from Task 1)
- Test: `tests/test_case_writer.py`

**Interfaces:**
- Consumes: `FoamSettings`, `ObjectCfd`, `run_geometry`, `tunnel_bounds`.
- Produces:
  - `RunSpec(run_id: str, obj_id: str, direction_index: int, direction_deg: float, max_iterations: int)`, a frozen dataclass. `run_id = f"{obj_id}_d{direction_index}"`.
  - `write_case(case_dir: Path, run_model: trimesh.Trimesh, cfd: ObjectCfd, s: FoamSettings, spec: RunSpec) -> None`.
  - Patch names: `inlet` (x-min), `outlet` (x-max), `sides` (y faces), `top`, `floor` (z-min), `model`.

**Decisions the templates encode** (string.Template `$name` substitution):
- **blockMesh:**
  - cell size h = D / `base_cells_per_size`, counts = ceil(extent / h);
  - `locationInMesh` = (xmin + 1.37h, ymin + 1.41h, zmin + 1.73h).
- **snappyHexMesh:**
  - refinement:
    - surface `model` at `surface_refinement` (the object override wins);
    - `featureEdges` level = surface max;
    - box `wake` from model xmin − 0.2D to xmax + 3D, model y/z ± 0.5D, at `wake_refinement`.
  - `addLayers true`, `model` with `nSurfaceLayers` = `n_layers`, `expansionRatio 1.2`, `finalLayerThickness 0.4`, `minThickness 0.1`, `relativeSizes true`.
- **controlDict:**
  - `application simpleFoam`, `startFrom latestTime`, `stopAt endTime`;
  - `endTime` = `spec.max_iterations`, `deltaT 1`;
  - `writeControl timeStep`, `writeInterval` = `write_interval`, `purgeWrite 2`, `writeFormat binary`, `runTimeModifiable true`.
  - Functions:
    - `forces` (type forces, patches (model), `rho rhoInf`, `rhoInf` = ρ, `CofR (0 0 0)`, `writeControl timeStep`, `writeInterval 1`);
    - `fieldAverage1` (U, p, k mean, `window` = `average_window`, `windowType approximate`).
- **Fields:**

| Patch | U | p | k | omega | nut |
|---|---|---|---|---|---|
| inlet, floor none/moving/slip | `fixedValue (V 0 0)` | zeroGradient | fixedValue k_in | fixedValue ω_in | calculated |
| inlet, floor rough | `atmBoundaryLayerInletVelocity` (Uref V, Zref model height, z0, flowDir (1 0 0), zDir (0 0 1), d 0) | zeroGradient | `atmBoundaryLayerInletK` | `atmBoundaryLayerInletOmega` | calculated |
| outlet | pressureInletOutletVelocity | totalPressure p0 0 | inletOutlet | inletOutlet | calculated |
| sides, top | slip | zeroGradient | zeroGradient | zeroGradient | calculated |
| floor `none` / `slip` | slip | zeroGradient | zeroGradient | zeroGradient | calculated |
| floor `moving` | fixedValue (V 0 0) | zeroGradient | kqRWallFunction | omegaWallFunction | nutkWallFunction |
| floor `rough` | noSlip | zeroGradient | kqRWallFunction | omegaWallFunction | atmNutkWallFunction z0 |
| model | noSlip | zeroGradient | kqRWallFunction | omegaWallFunction | nutkWallFunction |

  k_in = 1.5(0.05V)², ω_in = √k_in / (0.09^0.25 · 0.05D).
- `transportProperties`: Newtonian, `nu`.
- `turbulenceProperties`: `simulationType RAS; RAS { model kOmegaSST; turbulence on; printCoeffs on; }`.
- `decomposeParDict`: `numberOfSubdomains` = `n_procs`, `method scotch`.
- The STL goes to `constant/triSurface/model.stl` (solid name `model`). `surfaceFeatureExtractDict` uses `includedAngle 150`.
- `fvSchemes` / `fvSolution` are the motorBike ones, with a `potentialFlow { nNonOrthogonalCorrectors 5; }` entry.

- [ ] **Step 1: Write the failing tests:**
  - `test_write_case_creates_all_files`: every template output exists under `system/`, `constant/`, `0/`.
  - `test_control_dict_values`: regex finds `endTime 1000;`, `writeInterval 50;`, `purgeWrite 2;`, `writeFormat binary;`, `rhoInf 1.225;`, `window 100;`.
  - `test_moving_floor_and_rough_floor_fields`: f1_car `0/U` has `floor` → `fixedValue` with `uniform (40 0 0)`. The building has `atmBoundaryLayerInletVelocity` on the inlet and `atmNutkWallFunction` in `0/nut`.
  - `test_layers_and_procs`: `nSurfaceLayers 3`, `numberOfSubdomains 4`.
  - `test_location_in_mesh_outside_model`: `run_model.contains([point])` is False.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement `write_case` and the templates.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Add the Docker test `@pytest.mark.docker test_starter_airplane_meshes`.
  1. Write the case for the starter airplane at `base_cells_per_size=4`, `surface_refinement=(2, 3)`.
  2. Copy it into a throwaway container.
  3. Run `blockMesh && surfaceFeatureExtract && snappyHexMesh -overwrite && checkMesh`.

  Assert the output contains `Mesh OK.`. Run: `pytest -m docker tests/test_case_writer.py -q`. Expected: PASS (a few minutes).
- [ ] **Step 6:** Commit `feat: automatic OpenFOAM case writer`.

### Task 5: Convergence rule and run queue

**Files:**
- Create: `solver/foam/convergence.py`, `solver/foam/queue.py`
- Test: `tests/test_convergence.py`, `tests/test_queue.py`

**Interfaces:**
- Produces:
  - `parse_force_dat(text: str) -> np.ndarray` with shape (N, 4): iteration, Fx, Fy, Fz (totals). It skips `#` lines and strips `(` `)`, so both the `(x y z)` and plain-column formats parse.
  - `load_forces(texts: list[str]) -> np.ndarray`: concatenates restart segments, keeping the last row per iteration.
  - `is_converged(forces: np.ndarray, window: int, tol: float, min_iters: int) -> bool`.
  - `is_diverged(forces: np.ndarray) -> bool`: any non-finite value.
  - `RunRecord`, a dataclass:
    - `run_id`, `obj_id`, `direction_index: int`, `direction_deg: float`;
    - `status: str` (one of `waiting running stopped done flagged failed`);
    - `iterations: int = 0`, `converged: bool = False`, `log: str = ""`, `machine: str = ""`.
  - `coarse_first_order(n: int) -> list[int]`.
  - `build_queue(objects: dict[str, ObjectCfd], obj_ids: list[str], directions: list[int] | None = None) -> list[RunRecord]`. Its order is coarse-first index-major: every object's index 0, then every object's index 4, and so on.
  - `load_queue(path) -> list[RunRecord]` (missing → []).
  - `save_queue(path, records)`: writes to `path.tmp`, then `os.replace`.
  - `merge_queue(existing, new) -> list[RunRecord]`: keeps the existing status for known `run_id`s.
  - `next_run(records) -> RunRecord | None`: the first `running` or `stopped` record, else the first `waiting`.

The rule (0606 note: this is a "small increment" test, δF/F < 0.002):

```python
def is_converged(forces, window, tol, min_iters):
    if len(forces) < window or forces[-1, 0] < min_iters or is_diverged(forces):
        return False
    last = forces[-window:]
    drag_scale = max(abs(last[:, 1].mean()), 1e-12)
    lift_scale = max(abs(last[:, 3].mean()), drag_scale)
    return (np.ptp(last[:, 1]) <= tol * drag_scale) and (np.ptp(last[:, 3]) <= tol * lift_scale)
```

- [ ] **Step 1: Write the failing tests:**
  - `test_converged_when_steady`: 300 rows with Fx = 100 ± 0.05 noise → True.
  - `test_not_converged_when_drifting`: Fx falling 1% across the last 50 → False.
  - `test_converged_with_near_zero_lift`: Fx steady at 100, Fz = ±0.1 → True.
  - `test_not_before_min_iters`: 150 steady rows → False.
  - `test_nan_is_diverged`.
  - `test_parse_both_formats`: `"10 (1 2 3) (0 0 0) (1 2 3)"` and `"10 1 2 3 0 0 0 1 2 3"` → `[10, 1, 2, 3]`.
  - `test_restart_segments_dedupe`.
  - `test_coarse_first_order`:
    - `coarse_first_order(9) == [0, 4, 8, 2, 6, 1, 3, 5, 7]`;
    - `coarse_first_order(5) == [0, 2, 4, 1, 3]`;
    - `coarse_first_order(1) == [0]`.
  - `test_build_queue_interleaves_objects`.
  - `test_next_run_prefers_interrupted_runs`: waiting, running, done → returns the running one.
  - `test_save_queue_atomic_roundtrip`.
  - `test_merge_keeps_done_status`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement both modules.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat: convergence rule and resumable run queue`.

### Task 6: Runner and CLI

**Files:**
- Create: `solver/foam/runner.py`, `solver/run_openfoam.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: everything from Tasks 2–5.
- Produces:
  - `FOAM_BASHRC: str` (from Task 1).
  - `class Executor(Protocol): def run(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess`, and `SubprocessExecutor`.
  - `class FoamContainer`, constructed as `__init__(self, s: FoamSettings, name: str, ex: Executor)`, with methods:
    - `start()`: `docker run -d --name <name> -v <volume>:/cases -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 <image> sleep infinity`. If the container already exists, it starts it;
    - `copy_in(local_case: Path, run_id)`;
    - `sh(run_id, script: str, detach=False)`: `docker exec [-d] <name> bash -lc "source $FOAM_BASHRC; cd /cases/<run_id>; <script>"`;
    - `read(run_id, rel_path) -> str` (returns "" if missing);
    - `exists(run_id, rel_path) -> bool`;
    - `copy_out(run_id, rel_path, local_dir)`;
    - `remove_case(run_id)`.
  - `RunContext`, a dataclass:
    - `settings`, `objects: dict[str, ObjectCfd]`, `models: dict[str, trimesh.Trimesh]` (model0 per object);
    - `root: Path` (`runs/full`, `runs/preview` or `runs/test`), `cap: int`;
    - `container: FoamContainer`, `poll_s: float = 20.0`.
  - `run_one(rec: RunRecord, ctx: RunContext) -> RunRecord`.
  - `run_batch(obj_ids: list[str], ctx: RunContext, machine: str) -> list[RunRecord]`.
  - `stop_flag(root) -> Path` (= `root / "STOP"`).
  - Result layout: `<root>/<obj_id>/<run_id>/result/` holds `VTK/`, `postProcessing/` and `run.json`. `run.json` contains:
    - `{run_id, obj_id, direction_index, direction_deg, speed_m_s, iterations, converged, fields: "instant"|"mean", status}`.

**Lifecycle of `run_one`:**
1. **Fresh run** (status `waiting`, or `running`/`stopped` but `processor0` is missing from the volume):
   1. `write_case` locally under `<root>/<obj>/<run_id>/case`, then `copy_in`.
   2. `blockMesh > log.blockMesh 2>&1 && surfaceFeatureExtract > … && snappyHexMesh -overwrite > … && checkMesh > log.checkMesh 2>&1`. If `log.checkMesh` lacks `Mesh OK.`, mark `failed`.
   3. `decomposePar -force > … && mpirun -np N potentialFoam -parallel -writep > log.potentialFoam 2>&1`.
2. **Resume** (`processor0` present): skip step 1. `startFrom latestTime` continues from the last save.
3. **Launch:** `mpirun -np N simpleFoam -parallel > log.simpleFoam 2>&1` with `detach=True`. Set status `running` and save the queue.
4. **Poll every `poll_s`:**
   - read every `postProcessing/forces/*/force.dat` → `load_forces`;
   - if the stop flag exists or `is_converged` is true: `sh(run_id, "foamDictionary -entry stopAt -set writeNow system/controlDict")` once;
   - stop polling when `pgrep -x simpleFoam` returns non-zero.
5. **Classify:**
   - `failed`: `log.simpleFoam` contains `FOAM FATAL` or `Floating point exception`, or `is_diverged`;
   - else `stopped`: the stop flag caused the stop;
   - else `done`, fields instant: converged;
   - else `flagged`, fields mean: reached the cap.
6. **Export** (done or flagged): `reconstructPar -latestTime && foamToVTK -latestTime -fields '(U p k omega nut UMean pMean kMean)'`. Then `copy_out` VTK and postProcessing, write `run.json` and `remove_case`. For a failed run, copy out the logs only.

`run_batch`:
- loads, merges and saves `<root>/queue_<machine>.json` (one per machine, so the laptop's and Mac mini's queues never overwrite each other);
- removes a stale stop flag at start;
- installs a SIGINT handler that creates the stop flag;
- loops `next_run`;
- returns after a `stopped` result.

CLI `solver/run_openfoam.py`:
- `run --objects airplane f1_car [--preview] [--machine laptop]`: preview = directions `[0, 4, 8]`, cap 500, root `runs/preview`. Full = cap 1000, root `runs/full`.
- `stop [--preview]`: touches the flag.
- `status [--preview]`: one table row per run.
- `test --object X --direction-index I --iterations N [--mesh-only]`: root `runs/test`.
- If `docker info` fails, every command exits 2 with "Docker Desktop isn't running — open it and try again."

- [ ] **Step 1: Write the failing tests** with a `FakeExecutor` that records commands and returns scripted outputs:
  - `test_fresh_run_meshes_then_launches_detached`;
  - `test_resume_skips_meshing_when_processor0_exists`;
  - `test_restarts_when_case_missing_from_volume`;
  - `test_stop_flag_sets_write_now_and_status_stopped`;
  - `test_converged_run_is_done_instant`;
  - `test_cap_run_is_flagged_mean`;
  - `test_fatal_log_marks_failed_and_batch_continues`;
  - `test_bad_mesh_marks_failed`;
  - `test_docker_not_running_message` (CLI with an executor whose `docker info` fails → exit code 2 and the message).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Docker integration, `@pytest.mark.docker test_short_real_run`: `test --object airplane --direction-index 0 --iterations 60`, with the coarse Task 4 settings passed through a `--coarse` flag. Assert that `result/run.json` exists, status is in (`flagged`, `done`), and `result/VTK/*.vtm` exists. Run: `pytest -m docker tests/test_runner.py -q`.
- [ ] **Step 6:** Commit `feat: resumable Docker batch runner and CLI`.

### Task 7: First real test run (checkpoint data)

- [ ] **Step 1:** In the background, run `./.venv/bin/python solver/run_openfoam.py test --object airplane --direction-index 4 --iterations 1000` with the full default settings.
- [ ] **Step 2:** When it finishes, record in `runs/test/REPORT.md`:
  - cell count (from `log.checkMesh`);
  - mesh time;
  - seconds per iteration (from `log.simpleFoam` ExecutionTime);
  - iterations to converge;
  - Docker memory peak (`docker stats --no-stream`).

  Work out the predicted time for 36 runs.
- [ ] **Step 3:** If the prediction is over 27 machine-hours, lower `surface_refinement` or `base_cells_per_size` in the config and re-run Steps 1–2. Only if the Docker file handling is the bottleneck (high system time in `docker stats`) try OpenFOAM.app.
- [ ] **Step 4:** Commit the config and report: `chore: tune mesh size from test run`.

(Tasks 8–12 proceed while this runs.)

### Task 8: Cache format, Navier-Stokes terms, sustainability, importer

**Files:**
- Create: `app/cache_format.py`, `solver/foam/ns_terms.py`, `solver/foam/sustainability.py`, `solver/foam/importer.py`, `solver/import_results.py`, `tests/fake_cache.py`
- Test: `tests/test_ns_terms.py`, `tests/test_sustainability.py`, `tests/test_importer.py`

**Interfaces:**
- `app/cache_format.py`:
  - `FORMAT_VERSION = 2`; `state_key(d) -> f"d{d}"`; `wand_key(d, w) -> f"d{d}_w{w}"`;
  - `write_cache(path, payload: dict)` (`np.savez_compressed`); `read_cache(path) -> dict` (raises `ValueError` if `format_version != 2`).
  - Payload keys:
    - `format_version`, `mesh_vertices` (V,3 f32), `mesh_faces` (F,3 i32);
    - `directions_deg` (9,), `wand_heights` (10,), `wind_vectors` (9,3), `view_bounds` (6,), `config_json`;
    - `{state}_metrics` (JSON), `{state}_pressure` (V, f32 Pa), `{state}_ns_terms` (JSON `{convective, pressure, viscous}`, m/s²);
    - `{wand}_lines` (M,3 f32, NaN-joined), `{wand}_k` (M f32, NaN at separators), `{wand}_dominant` (M int8: 0 convective, 1 pressure, 2 viscous, −1 at separators).
- `ns_term_fields(grid: pv.DataSet, nu: float, u_name: str, p_name: str) -> pv.DataSet`. It adds point arrays `ns_convective`, `ns_pressure`, `ns_viscous` and `ns_dominant`:
  - convective = |(v·∇)v|, with (v·∇)v_i = Σ_j v_j ∂v_i/∂x_j;
  - pressure = |∇p|, where p is kinematic;
  - viscous = ν|∇²v|, with ∇²v_i = Σ_j ∂/∂x_j(∂v_i/∂x_j) from a second `compute_derivative` pass.
  - The 0606 comments explain these as rates of change built with the chain rule.
- `ns_term_summary(grid, bounds6) -> dict[str, float]`: mean of each magnitude inside the bounds.
- `sustainability`:
  - `drag_power_kw(cd, area, rho, v) = ½ρC_dAV³/1000`;
  - `fuel_and_co2(power_kw, ref_power_kw, fuel_l_per_kwh, co2_kg_per_l) -> (fuel_saved_l_h, co2_saved_kg_h)`, clamped at ≥ 0;
  - `speed_gain_pct(cd, ref_cd) = ((ref_cd/cd)^(1/3) − 1)·100`;
  - `wind_load_table(force_n, moment_nm, v_sim) -> list[dict]`. Rows `{label, v_m_s, force_n, moment_nm}` for Breeze 3.33, Category 1 33, Category 3 50 and Category 5 70 m/s, scaled by (v/v_sim)².
- `import_object(obj_id: str, settings: FoamSettings, cfd: ObjectCfd, model0: trimesh.Trimesh, obj_config: dict, results_root: Path, out_path: Path) -> dict`. The returned report is `{"imported": [idx], "filled": {idx: from_idx}, "flagged": [idx]}`. Per finished direction θ:
  1. **Read the result:** the highest-time `VTK/*.vtm` → `internal` and the `model` patch (found by block name, recursively). The fields are `U,p,k` or `UMean,pMean,kMean`, per `run.json`.
  2. **Pressure on the model:** rotate the model0 vertices by R(θ), take the nearest model-patch cell centre (cKDTree), and set pressure = p·ρ.
  3. **N-S terms:** `ns_term_fields` on the internal mesh clipped to the run-frame view region. The summary covers the model bbox ± 0.25D.
  4. **Streamlines:** for each wand height, `streamlines_from_source(seed_rake(run_model, z, seeds_per_wand), vectors=U, max_length=1.5·(view xmax − seed x))`. Rotate the points by R(θ)ᵀ back into the model frame, sample k and `ns_dominant`, and NaN-join the lines.
  5. **Forces and metrics:** take the last `force.dat` row (or the mean of the last 100 rows if flagged) and `moment.dat`. The metrics JSON has:
     - `direction_deg`, `wind_speed_m_s`, `reynolds_number` = VD/ν;
     - `drag_force_n` = Fx, `side_force_n` = Fy, `lift_force_n` = Fz;
     - `drag_coefficient_cd` = Fx/(½ρV²A), `lift_coefficient_cl` = Fz/(½ρV²A), with A = `frontal_area(run_model)`;
     - `frontal_area_m2`, `drag_power_kw`, `base_moment_nm` (|My, Mx| magnitude);
     - `iterations`, `converged`, `filled_from: null`;
     - `fuel_saved_l_per_hr`, `co2_saved_kg_per_hr`, `speed_gain_pct`, using the cruise ρ and V when `cfd.cruise` is set;
     - `wind_load` (table) for `wind_load` objects.

  Missing directions take the nearest finished direction's arrays, with `filled_from` set. Zero finished directions → `ValueError(f"No finished runs for {obj_id}")` and nothing is written. Wind vector = R(θ)ᵀ·(1, 0, 0).
- `solver/import_results.py --root runs/full [--objects …]`: writes `assets/cache/<obj>_states.npz` and prints each report.
- `tests/fake_cache.py`: `write_fake_cache(path: Path, obj_config: dict, n_dir: int = 9, n_wand: int = 10) -> None`. It writes a valid v2 cache from a box mesh, where pressure in direction d is constant `d` (so interpolation is checkable) and the lines are straight.

- [ ] **Step 1: Write the failing tests:**
  - `test_terms_on_analytic_field`: a structured grid on [1,2]³ with v = (x², xy, 0), p = x + 2y + 3z, ν = 0.1. At interior points, within 5%:
    - convective = |(2x³, 2x²y, 0)|;
    - pressure = √14;
    - viscous = 0.1·|(2, 0, 0)|.
  - `test_dominant_index`: the index of the largest term.
  - Sustainability:
    - `drag_power_kw(1.0, 2.0, 1.225, 40) == pytest.approx(78.4)`;
    - `speed_gain_pct(0.5, 1.0) == pytest.approx(25.99, abs=0.01)`;
    - `wind_load_table(1000, 5000, 50)` Category 5 row → `force_n == pytest.approx(1960)`;
    - fuel is clamped at 0 when power exceeds the reference.
  - `test_import_fake_result` (build a fake `result/` dir with a pyvista MultiBlock `.vtm`, `force.dat`, `moment.dat` and `run.json` for directions 0 and 4): the cache reads back with `format_version == 2`, the pressure length equals the vertex count, lines contain NaN separators, and `filled == {1: 0, 2: 0, 3: 4, …}` with ties going to the lower index.
  - `test_import_with_no_results_raises`.
  - `test_flagged_run_uses_mean_fields`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement all five modules.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** When Task 7's run exists, run `./.venv/bin/python solver/import_results.py --root runs/test --objects airplane`. Expected: writes the cache with 1 imported and 8 filled. Fix the block-name lookup if the real `.vtm` nests differently, and add that layout to `test_import_fake_result`.
- [ ] **Step 6:** Commit `feat: v2 cache importer with N-S terms and sustainability metrics`.

### Task 9: State manager and input controller (v2)

**Files:**
- Modify (rewrite): `app/state_manager.py`, `app/input_controller.py`
- Test: `tests/test_system.py`. Replace `test_state_manager` and `test_input_controller` with the tests below; keep `test_navier_stokes_divergence`.

**Interfaces:**
- `ObjectStateManager(obj_id: str, cache_dir: Path | str = "assets/cache")`:
  - attributes `directions_deg`, `wand_heights`, `view_bounds`, `mesh_vertices`, `mesh_faces`, `obj_config`, `wind_speed`;
  - `direction_for_knob(knob: float) -> float` = d_min + knob·(d_max − d_min), with knob clamped to [0, 1];
  - `get_state(wand_index: int, knob: float) -> dict` with keys:
    - `wand_index`, `wand_z`, `direction_deg`, `wind_speed`, `wind_vector`;
    - `metrics`, `ns_terms`, `surface_pressure` (0606 interpolated between the bracketing directions);
    - `lines`, `k`, `dominant`, `filled` (nearest direction);
    - `mesh_vertices`, `mesh_faces`, `obj_config`.

  It decodes all JSON and arrays at load time so `get_state` is pure numpy.
- `ExhibitionInputController(wand_count: int = 10, knob_step: float = 1/32)`:
  - signals `state_changed = Signal(int, float)` (wand_index, knob) and `view_changed = Signal(int)`;
  - attributes `current_wand_index` (start 4), `current_knob` (start 0.0), `current_view` (start 0);
  - methods `set_wand`, `step_wand`, `set_knob`, `step_knob`, `set_view`, `cycle_view` (0→1→2→0);
  - `handle_key_press(event) -> bool`: digits 1–0 → wand 0–9, Up/W and Down/S → wand ±1, Right/D/+/=/] and Left/A/-/_/[ → knob ±step, Tab → `cycle_view`;
  - `handle_wheel_event(event) -> bool` → knob.

- [ ] **Step 1: Write the failing tests** (using `write_fake_cache` into `tmp_path`):
  - `test_lookup_under_1ms` (median of 200 calls < 1.0 ms);
  - `test_interpolation_between_directions`: at the knob halfway between d1 and d2, mean pressure = 1.5;
  - `test_nearest_lines`;
  - `test_rejects_v1_cache` (`ValueError`);
  - `test_controller_signals_and_clamps`: knob stays in [0, 1], wand in [0, 9];
  - `test_tab_cycles_view` (QKeyEvent with Key_Tab).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement. Keep the 0606 docstring with t = (d − d₁)/(d₂ − d₁).
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat: direction/wand state manager and controller`.

### Task 10: Monitor window and HUD widgets

**Files:**
- Create: `app/hud_widgets.py`
- Modify: `app/monitor_window.py`
- Test: `tests/test_hud_widgets.py`

**Interfaces:**
- Consumes: the `get_state` dict and the controller signals.
- Produces:
  - `polyline_from_nan_joined(points: np.ndarray, scalars: dict[str, np.ndarray] | None = None) -> pv.PolyData`: one line cell per NaN-separated run, with scalars kept for the non-NaN points.
  - `log_bar_value(x: float, lo: float = 1e-3, hi: float = 1e5) -> int`: 0–100 on a log10 scale.
  - `NsTermsPanel(QFrame)` with `.set_terms(terms: dict[str, float])`: the equation ρ(v·∇)v = −∇p + μ∇²v, three labelled bars, and the dominant bar highlighted.
  - `SustainabilityOverlay(QFrame)`, constructed with `(widgets: tuple[str, ...], parent)`, with `.set_metrics(metrics: dict)`:
    - `fuel` card: fuel and CO₂ saved per hour, SDG 13;
    - `speed` card: top-speed gain %;
    - `wind_load` card: table rows plus "worst direction so far" from `metrics["wind_load"]`, SDG 9.

**Monitor window changes:**
- **Controls in the HUD:**
  - "SMOKE WAND: step k/10 (z = … m)";
  - "<direction_label>: θ°";
  - "WIND SPEED: V m/s (fixed)".
- **3D scene:**
  - the nozzle actor is replaced by a wind arrow at the upstream view edge along `wind_vector`, and a thin rod across y at `wand_z`;
  - the wireframe and camera come from `view_bounds`;
  - `SustainabilityOverlay` sits over the top-left of the plotter;
  - the building's `info_card` text goes in the HUD.
- **Views** (one handler on `controller.view_changed`, so all four monitors switch together):
  - 0: streamlines cyan, model pressure;
  - 1: streamlines coloured by `dominant` (3 fixed colours plus a legend);
  - 2: streamlines coloured by `k` (log scale, bar title "Turbulence energy k (m²/s²) · k-ω SST").
- **Tab key:** override `event()` / `focusNextPrevChild` so Qt doesn't swallow Tab for focus changes.
- **Particles:** animate along the polylines.
- **Footer:** shows the remote URL when one is given.

- [ ] **Step 1: Write the failing tests:**
  - `test_polyline_from_nan_joined`: 2 lines → `n_lines == 2`, scalars length = 5;
  - `test_log_bar_value`: 1e-3 → 0, 1e5 → 100, 10 → 50;
  - `test_overlay_text` (offscreen `QT_QPA_PLATFORM=offscreen`): fuel card shows "L/h" and "SDG 13"; wind_load card shows "Category 5".
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement `hud_widgets.py`, then the `monitor_window.py` changes.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Manual check:
  1. Write fake caches for all 4 objects into `assets/cache/`. Use the real airplane cache from Task 8 Step 5 if it exists.
  2. Run `./.venv/bin/python app/main.py --dev-mode`.
  3. Screenshot the window after pressing 3, turning the knob 8 pulses, and Tab twice.

  Expected: wand moves, arrow turns, views switch on all 4 windows. Show Isaac the screenshot.
- [ ] **Step 6:** Commit `feat: three-view monitor window with N-S bars and SDG widgets`.

### Task 11: iPad remote

**Files:**
- Create: `app/remote_server.py`, `app/remote/index.html`
- Test: `tests/test_remote_server.py`

**Interfaces:**
- `parse_command(body: bytes) -> tuple[str, float]`. It accepts JSON `{"kind": "wand"|"knob"|"view", "value": number}`. It raises `ValueError` for bad JSON, an unknown kind or a non-finite value.
- `class RemoteServer(QObject)`:
  - `command = Signal(str, float)`;
  - `__init__(self, controller: ExhibitionInputController, port: int = 8765)`;
  - `start() -> str`: binds 0.0.0.0, uses a `ThreadingHTTPServer` in a daemon thread, and returns `http://<lan-ip>:<port>/` (the LAN IP comes from a UDP-connect probe, falling back to 127.0.0.1);
  - `stop()`.
- **Routes:**
  - `GET /` → index.html;
  - `GET /state` → `{"wand", "knob", "view", "wand_count"}`;
  - `POST /cmd` → 204, or 400 on a `ValueError`.
- `command` is connected to controller methods with `Qt.QueuedConnection`, so they run on the GUI thread. The values are clamped by the controller.
- **index.html** (one file, no external assets, large touch targets):
  - an SVG dial dragged around → knob 0–1, sent at most 20 times a second;
  - a wand slider 1–10;
  - three view buttons;
  - a short "what you're seeing" line per view;
  - `/state` polled every 500 ms while not dragging.

- [ ] **Step 1: Write the failing tests:**
  - `test_parse_command_accepts_valid`;
  - `test_parse_command_rejects_bad_input` (`b"{"`, `{"kind":"x"}`, `{"kind":"knob","value":"nan"}`, `{"kind":"knob","value":1e999}`);
  - `test_post_cmd_moves_controller` (start on port 0, POST knob 0.5, process Qt events, assert `controller.current_knob == 0.5`);
  - `test_bad_post_is_400_and_state_unchanged`;
  - `test_get_state_json`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement both files.
- [ ] **Step 4:** Run them. Expected: PASS.
- [ ] **Step 5:** Commit `feat: iPad remote over local HTTP`.

### Task 12: Wiring, legacy removal, docs

**Files:**
- Modify: `app/main.py`, `CLAUDE.md`, `run_exhibition.sh` (if it references precompute)
- Create: `scripts/sync_to_mini.sh`, `scripts/pull_from_mini.sh`
- Delete: `solver/batch_precompute.py`, `solver/vtk_importer.py`, `solver/openfoam_bridge.py`, `solver/mesh_voxelizer.py`

**Interfaces:**
- `main.py` builds `ExhibitionInputController(wand_count, knob_step)` from the config and starts `RemoteServer` on `controls.remote_port` unless `--no-remote` is passed. It prints the URL, and passes it to each `MonitorWindow(obj_id, controller, cache_dir, remote_url)`.
- A missing or v1 cache prints "Run solver/import_results.py (or git checkout v1-fallback for the old exhibit)" and exits 1.
- `sync_to_mini.sh`: `rsync -av --exclude .venv --exclude runs --exclude .git ./ "${MINI_HOST:-mac-mini.local}:navflow/"`. `pull_from_mini.sh`: `rsync -av "${MINI_HOST:-mac-mini.local}:navflow/runs/full/" runs/full/`. Each machine keeps its own `queue_<machine>.json` (Task 6), so pulling results never overwrites the laptop's queue.

- [ ] **Step 1:** Make the changes. Update CLAUDE.md:
  - commands (prepare_model, run_openfoam, import_results, the remote);
  - the structure tree;
  - keybindings (wand, knob, Tab);
  - the cache format v2 note;
  - the fallback tag.
- [ ] **Step 2:** Run `./.venv/bin/python -m pytest -q` (no docker marker). Expected: all pass. Run `grep -rn "batch_precompute\|vtk_importer\|openfoam_bridge\|mesh_voxelizer\|z_steps\|wind_speeds" app solver tests`. Expected: no hits.
- [ ] **Step 3:** Commit `chore: wire remote, remove legacy pipeline, update docs`.

### Task 13: Models, preview, full run, import (operations with checkpoints)

- [ ] **Step 1 (Tue morning, needs Isaac's files):** for each object, `prepare_model.py --object <id>`. Set `base_rotation_deg` in the config until the preview PNG shows the nose facing −x and up as +z. Show Isaac the 4 previews. Commit the models and config.
- [ ] **Step 2:** `run_openfoam.py run --objects airplane f1_car building human --preview --machine laptop`. That's 12 runs, cap 500 (or split with the mini via `sync_to_mini.sh` + `ssh`).
- [ ] **Step 3 (Isaac checkpoint):** `import_results.py --root runs/preview`, then launch the app. Isaac checks wakes, colours and timing. **Settings frozen**: commit the config with `chore: freeze settings after preview`.
- [ ] **Step 4:** `sync_to_mini.sh`. Laptop: `run --objects airplane f1_car --machine laptop`. Mini (over ssh, in `caffeinate -i`): `run --objects building human --machine mini`.
- [ ] **Step 5 (Wed evening):** `pull_from_mini.sh`, `import_results.py --root runs/full`, `pytest -q`, launch the app. Report flagged and failed runs to Isaac and re-run any failed ones with `run` (they're resumable).
- [ ] **Step 6 (Thu):** final import. Test the iPad on the phone hotspot. Commit the caches: `feat: OpenFOAM results for exhibition`.
