"""Turn finished OpenFOAM runs into the version 2 results cache (assets/cache/<obj>_states.npz).

Input layout (written by solver/foam/runner.py), one folder per wind direction:

    <results_root>/<obj_id>/<run_id>/result/
        run.json                                   status, fields ("instant" | "mean"), iterations ...
        postProcessing/forces/<time>/force.dat     total force (N) per iteration
        postProcessing/forces/<time>/moment.dat    total moment (N m) about the origin (the base centre)
        VTK/<run_id>_<time>/internal.vtu           the flow field in the whole tunnel
        VTK/<run_id>_<time>/boundary/model.vtp     the model surface (pressure on its faces)

Only the highest <time> folder is read. The .vtm index files are ignored because pyvista mislabels
their blocks.

Maths link (IGCSE 0606):
  * The tunnel is fixed and the MODEL is rotated by R(theta) for each wind direction. To show the
    results on the un-rotated model, points are rotated back with the inverse rotation. A rotation
    matrix R is orthogonal, so R^-1 = R^T: for row vectors that is a plain  p_run @ R.
  * The wind vector in the model frame is R^T (1, 0, 0): the first column-of-R^T trick, i.e. the
    x-axis rotated backwards.
  * A missing direction takes the nearest finished one: minimise |i - j| over finished j (ties go to
    the lower index) - a nearest-neighbour rule on an arithmetic progression of angles.
  * Cd = F / (1/2 rho V^2 A) turns a force in newtons into a dimensionless number; P = F V.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree

from app.cache_format import FORMAT_VERSION, state_key, wand_key, write_cache
from solver.foam.convergence import load_forces
from solver.foam.geometry import frontal_area, rotation_matrix, run_geometry, seed_rake, view_bounds, wand_heights
from solver.foam.ns_terms import ns_term_fields, ns_term_summary
from solver.foam.settings import FoamSettings, ObjectCfd
from solver.foam.sustainability import drag_power_kw, fuel_and_co2, speed_gain_pct, wind_load_table

FINISHED = ("done", "flagged")
FIELD_NAMES = {"instant": ("U", "p", "k"), "mean": ("UMean", "pMean", "kMean")}
SUMMARY_MARGIN = 0.25  # the N-S term summary covers the model bbox +- 0.25 D (D = longest bbox side)
CLIP_PAD = 0.25  # extra D around the clip box so edge cells still have neighbours for their derivatives


def _finished_runs(obj_id: str, results_root: Path) -> dict[int, tuple[Path, dict]]:
    """{direction index: (result folder, run.json)} for every finished run of `obj_id`."""
    runs: dict[int, tuple[Path, dict]] = {}
    for run_json in sorted((Path(results_root) / obj_id).glob("*/result/run.json")):
        try:
            meta = json.loads(run_json.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("status") in FINISHED and meta.get("obj_id", obj_id) == obj_id:
            runs[int(meta["direction_index"])] = (run_json.parent, meta)
    return runs


def _latest_vtk_dir(result: Path, run_id: str) -> Path:
    """The VTK/<run_id>_<time> folder with the largest time."""
    best: tuple[float, Path] | None = None
    prefix = run_id + "_"
    for d in (result / "VTK").iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            try:
                t = float(d.name[len(prefix):])
            except ValueError:
                continue
            if best is None or t > best[0]:
                best = (t, d)
    if best is None:
        raise ValueError(f"{run_id}: no VTK/{prefix}<time> folder in {result}")
    return best[1]


def _read_history(result: Path, filename: str) -> np.ndarray:
    """Iteration, x, y, z rows of forces/<time>/<filename>, restart folders joined in time order."""
    dirs = []
    for d in (result / "postProcessing" / "forces").iterdir():
        try:
            dirs.append((float(d.name), d))
        except ValueError:
            continue
    texts = [(d / filename).read_text() for _, d in sorted(dirs) if (d / filename).exists()]
    return load_forces(texts)


def _settled(history: np.ndarray, flagged: bool, window: int) -> np.ndarray:
    """(x, y, z) of the last row, or of the mean of the last `window` rows for a run that hit its cap."""
    if not len(history):
        raise ValueError("no force/moment rows")
    rows = history[-window:] if flagged else history[-1:]
    return rows[:, 1:4].mean(axis=0)


def _pressure_on_model(patch: pv.PolyData, name: str, model0: trimesh.Trimesh, rot: np.ndarray, rho: float) -> np.ndarray:
    """Pascals at every vertex of the un-rotated model: nearest patch cell centre of the rotated model."""
    if name in patch.cell_data:
        sites, values = patch.cell_centers().points, np.asarray(patch.cell_data[name])
    else:
        sites, values = np.asarray(patch.points), np.asarray(patch.point_data[name])
    run_vertices = np.asarray(model0.vertices) @ rot.T  # R v, written for row vectors
    _, nearest = cKDTree(sites).query(run_vertices)
    return (values[nearest] * rho).astype(np.float32)  # p is kinematic (p / rho): multiply by rho


def _clip(grid: pv.UnstructuredGrid, box: np.ndarray, keep: tuple[str, ...]) -> pv.UnstructuredGrid:
    """Whole cells whose centre is inside `box` (xmin, xmax, ymin, ymax, zmin, zmax), keeping only `keep` arrays.

    Cutting before taking derivatives keeps the cost proportional to the view region, not the
    whole (up to millions of cells) tunnel. Everything is vectorised numpy.
    """
    centres = grid.cell_centers().points
    inside = np.all((centres >= box[0::2]) & (centres <= box[1::2]), axis=1)
    sub = grid.extract_cells(np.flatnonzero(inside))
    for data in (sub.point_data, sub.cell_data):
        for name in [n for n in data.keys() if n not in keep]:
            del data[name]
    return sub


def _streamlines(
    grid: pv.DataSet, seeds: np.ndarray, u_name: str, k_name: str, max_length: float, rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN-joined lines (model frame), k along them and the dominant N-S term along them."""
    out = grid.streamlines_from_source(
        pv.PolyData(seeds), vectors=u_name, integration_direction="forward",
        max_length=max_length, compute_vorticity=False,
    )
    pts_all = np.asarray(out.points)
    lines_out, k_out, dom_out = [], [], []
    if out.n_points:
        terms = np.column_stack([out.point_data[f"ns_{n}"] for n in ("convective", "pressure", "viscous")])
        dominant = terms.argmax(axis=1)  # the three magnitudes are interpolated; take the largest on the line
        k_all = np.asarray(out.point_data[k_name], dtype=float)
        conn, i = np.asarray(out.lines), 0
        while i < len(conn):
            n = int(conn[i])
            ids = conn[i + 1:i + 1 + n]
            i += n + 1
            if n < 2:
                continue
            lines_out += [pts_all[ids] @ rot, np.full((1, 3), np.nan)]  # R^-1 = R^T: back to the model frame
            k_out += [k_all[ids], [np.nan]]
            dom_out += [dominant[ids], [-1]]
    if not lines_out:
        return np.empty((0, 3), np.float32), np.empty(0, np.float32), np.empty(0, np.int8)
    return (
        np.vstack(lines_out).astype(np.float32),
        np.concatenate(k_out).astype(np.float32),
        np.concatenate(dom_out).astype(np.int8),
    )


def _metrics(
    cfd: ObjectCfd, obj_config: dict, settings: FoamSettings, meta: dict, result: Path, run_model: trimesh.Trimesh
) -> dict:
    flagged = meta["status"] == "flagged"
    force = _settled(_read_history(result, "force.dat"), flagged, settings.average_window)
    moment = _settled(_read_history(result, "moment.dat"), flagged, settings.average_window)

    rho = cfd.density_kg_m3
    v = float(meta.get("speed_m_s", cfd.speed_m_s))
    area = frontal_area(run_model)
    q_a = 0.5 * rho * v**2 * area  # dynamic pressure x frontal area, the force scale
    cd = float(force[0] / q_a)
    cl = float(force[2] / q_a)

    # Fuel and CO2 are quoted at cruise conditions when the object has them (aircraft at altitude).
    rho_c = cfd.cruise["density_kg_m3"] if cfd.cruise else rho
    v_c = cfd.cruise["speed_m_s"] if cfd.cruise else v
    power = drag_power_kw(cd, area, rho_c, v_c)
    ref_power = drag_power_kw(cfd.reference_cd, area, rho_c, v_c)
    fuel, co2 = fuel_and_co2(
        power, ref_power, float(obj_config["fuel_burn_factor_l_per_kwh"]), float(obj_config["co2_kg_per_l_fuel"])
    )
    base_moment = float(np.hypot(moment[0], moment[1]))  # |(Mx, My)|: overturning moment about the base centre
    metrics = {
        "direction_deg": float(meta["direction_deg"]),
        "wind_speed_m_s": v,
        "reynolds_number": v * cfd.real_size_m / cfd.kinematic_viscosity_m2_s,  # Re = V D / nu
        "drag_force_n": float(force[0]),
        "side_force_n": float(force[1]),
        "lift_force_n": float(force[2]),
        "drag_coefficient_cd": cd,
        "lift_coefficient_cl": cl,
        "frontal_area_m2": float(area),
        "drag_power_kw": float(drag_power_kw(cd, area, rho, v)),
        "base_moment_nm": base_moment,
        "iterations": int(meta["iterations"]),
        "converged": bool(meta["converged"]),
        "filled_from": None,
        "fuel_saved_l_per_hr": float(fuel),
        "co2_saved_kg_per_hr": float(co2),
        "speed_gain_pct": float(speed_gain_pct(cd, cfd.reference_cd)),
    }
    if "wind_load" in cfd.widgets:
        metrics["wind_load"] = wind_load_table(float(force[0]), base_moment, v)
    return metrics


def _import_direction(
    idx: int, result: Path, meta: dict, settings: FoamSettings, cfd: ObjectCfd, model0: trimesh.Trimesh,
    obj_config: dict, heights: np.ndarray,
) -> dict:
    """Everything the cache stores for one finished wind direction."""
    deg = float(cfd.directions_deg[idx])
    rot = rotation_matrix(cfd.rotation_axis, deg)  # model frame -> run frame
    run_model = run_geometry(model0, cfd, deg)
    u_name, p_name, k_name = FIELD_NAMES[meta["fields"]]
    vtk_dir = _latest_vtk_dir(result, meta["run_id"])

    # 2. surface pressure on the model
    patch = pv.read(vtk_dir / "boundary" / "model.vtp")
    pressure = _pressure_on_model(patch, p_name, model0, rot, cfd.density_kg_m3)

    # 3. Navier-Stokes terms on the flow field around the model
    grid = pv.read(vtk_dir / "internal.vtu")
    view = np.array(view_bounds(run_model, cfd.floor))
    lo, hi = run_model.bounds
    d = float(run_model.extents.max())
    summary_box = np.column_stack([lo - SUMMARY_MARGIN * d, hi + SUMMARY_MARGIN * d]).ravel()
    box = np.empty(6)  # smallest box holding the view region and the summary region, plus a pad
    box[0::2] = np.minimum(view[0::2], summary_box[0::2]) - CLIP_PAD * d
    box[1::2] = np.maximum(view[1::2], summary_box[1::2]) + CLIP_PAD * d
    region = _clip(grid, box, (u_name, p_name, k_name))
    del grid
    region = ns_term_fields(region, cfd.kinematic_viscosity_m2_s, u_name, p_name)
    ns_terms = ns_term_summary(region, tuple(summary_box))

    # 4. streamlines from each smoke wand
    wands = []
    for z in heights:
        seeds = seed_rake(run_model, float(z), settings.seeds_per_wand)
        max_length = 1.5 * (view[1] - seeds[0, 0])
        wands.append(_streamlines(region, seeds, u_name, k_name, float(max_length), rot))

    # 5. forces and metrics
    metrics = _metrics(cfd, obj_config, settings, meta, result, run_model)
    return {"metrics": metrics, "pressure": pressure, "ns_terms": ns_terms, "wands": wands}


def import_object(
    obj_id: str, settings: FoamSettings, cfd: ObjectCfd, model0: trimesh.Trimesh, obj_config: dict,
    results_root: Path, out_path: Path,
) -> dict:
    """Import every finished direction of `obj_id` and write the v2 cache to `out_path`.

    Returns {"imported": [idx], "filled": {idx: from_idx}, "flagged": [idx]}. Raises ValueError (and
    writes nothing) if no direction has finished.
    """
    runs = _finished_runs(obj_id, Path(results_root))
    if not runs:
        raise ValueError(f"No finished runs for {obj_id}")
    n_dir = len(cfd.directions_deg)
    bad = [i for i in runs if not 0 <= i < n_dir]
    if bad:
        raise ValueError(f"{obj_id}: direction index {bad} outside 0..{n_dir - 1}")

    heights = wand_heights(model0, settings.wand_count)
    states = {
        idx: _import_direction(idx, result, meta, settings, cfd, model0, obj_config, heights)
        for idx, (result, meta) in sorted(runs.items())
    }

    filled: dict[int, int] = {}
    for idx in range(n_dir):
        if idx not in states:
            filled[idx] = min(states, key=lambda j: (abs(j - idx), j))  # nearest finished; ties -> lower

    payload: dict = {
        "format_version": np.array(FORMAT_VERSION),
        "mesh_vertices": np.asarray(model0.vertices, dtype=np.float32),
        "mesh_faces": np.asarray(model0.faces, dtype=np.int32),
        "directions_deg": np.asarray(cfd.directions_deg, dtype=float),
        "wand_heights": np.asarray(heights, dtype=float),
        "wind_vectors": np.array([rotation_matrix(cfd.rotation_axis, a).T @ [1.0, 0.0, 0.0] for a in cfd.directions_deg]),
        "view_bounds": np.array(view_bounds(model0, cfd.floor), dtype=float),
        "config_json": json.dumps(obj_config),
    }
    for idx in range(n_dir):
        src = states[filled.get(idx, idx)]
        metrics = dict(src["metrics"])
        if idx in filled:
            metrics["filled_from"] = filled[idx]
        s = state_key(idx)
        payload[f"{s}_metrics"] = json.dumps(metrics)
        payload[f"{s}_pressure"] = src["pressure"]
        payload[f"{s}_ns_terms"] = json.dumps(src["ns_terms"])
        for w, (lines, k, dominant) in enumerate(src["wands"]):
            key = wand_key(idx, w)
            payload[f"{key}_lines"], payload[f"{key}_k"], payload[f"{key}_dominant"] = lines, k, dominant
    write_cache(out_path, payload)

    return {
        "imported": sorted(states),
        "filled": filled,
        "flagged": sorted(i for i, (_, meta) in runs.items() if meta["status"] == "flagged"),
    }
