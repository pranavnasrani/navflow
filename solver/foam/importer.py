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
import warnings
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
from solver.foam.sustainability import (
    drag_power_kw, fuel_and_co2, fuel_burn_l_per_hr, level_flight_drag_n, speed_gain_pct, wind_load_table,
)

FINISHED = ("done", "flagged")
FIELD_NAMES = {"instant": ("U", "p", "k"), "mean": ("UMean", "pMean", "kMean")}
NUT_NAME = "nut"  # eddy viscosity; the run exports no nutMean, so mean-field runs use the instantaneous nut
SUMMARY_MARGIN = 0.25  # the N-S term summary covers the model bbox +- 0.25 D (D = longest bbox side)
DIRECTION_TOLERANCE_DEG = 1e-6  # run.json and the config must name the same angle
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


PRESSURE_NEIGHBOURS = 8  # solver faces averaged for each display vertex
SAME_SIDE_COS = 0.2  # a face counts only if its normal is within ~78 degrees of the vertex normal


def _pressure_on_model(patch: pv.PolyData, name: str, model0: trimesh.Trimesh, rot: np.ndarray, rho: float) -> np.ndarray:
    """Pascals at every vertex of the un-rotated model, from the solver's surface faces.

    The display model and the solver's surface are two different meshes of the same shape, so each
    vertex takes a weighted average of the PRESSURE_NEIGHBOURS nearest solver faces:
        p = sum(w_i p_i) / sum(w_i),   w_i = 1 / d_i^2   (inverse-square distance weighting)
    Only faces on the same side of the surface count (their unit normals n_i and the vertex normal n
    have n . n_i > SAME_SIDE_COS, a dot product / angle-between-vectors test), so the two faces of a thin
    wing never mix. A single nearest face gave blotchy, stair-stepped contours.
    """
    surf = patch.triangulate()
    if name in surf.cell_data:
        normals = surf.compute_normals(cell_normals=True, point_normals=False, split_vertices=False,
                                       consistent_normals=False, auto_orient_normals=False).cell_data["Normals"]
        sites, values = surf.cell_centers().points, np.asarray(surf.cell_data[name], dtype=float)
    else:
        normals = surf.compute_normals(cell_normals=False, point_normals=True, split_vertices=False,
                                       consistent_normals=False, auto_orient_normals=False).point_data["Normals"]
        sites, values = np.asarray(surf.points), np.asarray(surf.point_data[name], dtype=float)
    run_vertices = np.asarray(model0.vertices) @ rot.T  # R v, written for row vectors
    run_normals = np.asarray(model0.vertex_normals) @ rot.T
    k = min(PRESSURE_NEIGHBOURS, len(sites))
    dist, idx = cKDTree(sites).query(run_vertices, k=k)
    dist, idx = dist.reshape(len(run_vertices), k), idx.reshape(len(run_vertices), k)
    cos = np.einsum("vj,vkj->vk", run_normals, np.asarray(normals)[idx])
    if np.median(cos[:, 0]) < 0:  # OpenFOAM's boundary normals point out of the air, i.e. into the model
        cos = -cos
    w = (cos > SAME_SIDE_COS) / np.maximum(dist, 1e-12) ** 2
    lonely = w.sum(axis=1) == 0  # no same-side face nearby: fall back to the nearest face
    w[lonely, 0] = 1.0
    p = (w * values[idx]).sum(axis=1) / w.sum(axis=1)
    return (p * rho).astype(np.float32)  # p is kinematic (p / rho): multiply by rho


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """NaN-joined lines (model frame), k, the dominant N-S term and the air speed |U| along them.

    Maths link (IGCSE 0606): the speed is the magnitude of the 3D velocity vector,
    |U| = sqrt(u^2 + v^2 + w^2) (Pythagoras in three dimensions).
    """
    out = grid.streamlines_from_source(
        pv.PolyData(seeds), vectors=u_name, integration_direction="forward",
        max_length=max_length, compute_vorticity=False,
    )
    pts_all = np.asarray(out.points)
    lines_out, k_out, dom_out, speed_out = [], [], [], []
    if out.n_points:
        terms = np.column_stack([out.point_data[f"ns_{n}"] for n in ("convective", "pressure", "viscous")])
        dominant = terms.argmax(axis=1)  # the three magnitudes are interpolated; take the largest on the line
        k_all = np.asarray(out.point_data[k_name], dtype=float)
        speed_all = np.linalg.norm(np.asarray(out.point_data[u_name], dtype=float), axis=1)  # |U| at each point
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
            speed_out += [speed_all[ids], [np.nan]]
    if not lines_out:
        return np.empty((0, 3), np.float32), np.empty(0, np.float32), np.empty(0, np.int8), np.empty(0, np.float32)
    return (
        np.vstack(lines_out).astype(np.float32),
        np.concatenate(k_out).astype(np.float32),
        np.concatenate(dom_out).astype(np.int8),
        np.concatenate(speed_out).astype(np.float32),
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

    # Fuel is quoted at cruise conditions when the object has them (aircraft at altitude), otherwise at the
    # simulated air and speed. P = 1/2 rho (Cd A) v^3, so the same Cd A costs far less power in thin air.
    rho_c = cfd.cruise["density_kg_m3"] if cfd.cruise else rho
    v_c = cfd.cruise["speed_m_s"] if cfd.cruise else v
    eval_power = drag_power_kw(cd, area, rho_c, v_c)
    # Lift-to-drag ratio L/D = F_lift / F_drag (a ratio, so it has no units).
    lift_to_drag = float(force[2] / force[0]) if force[0] > 0 else 0.0
    # An aircraft with a mass in its cruise block flies level: D = W / (L/D), power P = D v (F v, in kW).
    # An angle with no upward lift cannot fly level, so it keeps the Cd A power above.
    mass = (cfd.cruise or {}).get("mass_kg")
    level_drag = level_flight_drag_n(float(mass), lift_to_drag) if mass else None
    if level_drag is not None:
        eval_power = level_drag * v_c / 1000.0
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
        "lift_to_drag": lift_to_drag,
        "iterations": int(meta["iterations"]),
        "converged": bool(meta["converged"]),
        "filled_from": None,
        "eval_speed_m_s": float(v_c),  # the speed the fuel numbers are worked out at
        "eval_altitude_km": float(cfd.cruise.get("altitude_km", 0.0)) if cfd.cruise else 0.0,  # 0 = sea level
        "eval_drag_power_kw": float(eval_power),
        "fuel_burn_l_per_hr": float(fuel_burn_l_per_hr(eval_power, float(obj_config["fuel_burn_factor_l_per_kwh"]))),
        # The three "saved" numbers and the reference angle need every finished direction: _apply_reference.
        "fuel_saved_l_per_hr": 0.0,
        "co2_saved_kg_per_hr": 0.0,
        "speed_gain_pct": 0.0,
        "reference_direction_deg": float(meta["direction_deg"]),
        # False: savings against the worst angle. True: an aircraft's extra fuel against its best L/D angle.
        "reference_is_best": bool(mass),
    }
    if "wind_load" in cfd.widgets:
        # The wind pushes the tower sideways: use the horizontal force |(Fx, Fy)|, like the moment's |(Mx, My)|.
        metrics["wind_load"] = wind_load_table(float(np.hypot(force[0], force[1])), base_moment, v)
    return metrics


def _apply_reference(metrics: dict, ref: dict, obj_config: dict) -> None:
    """Fill in the savings of one state against the reference state `ref`.

    The reference is the object's worst angle, or an aircraft's best L/D angle (reference_is_best).
    Maths link (IGCSE 0606): the drag power at the evaluation condition is P = 1/2 rho (Cd A) v^3,
    or P = (W / (L/D)) v for an aircraft in level flight.
      * fuel saved   = (P_ref - P) x litres per kWh, never below 0 (a difference of two powers);
        for the best angle the sign flips: extra fuel = (P - P_best) x litres per kWh, also never below 0;
      * CO2          = fuel x kg CO2 per litre (direct proportion);
      * speed gain   = ((Cd A)_ref / (Cd A))^(1/3) - 1, because at equal power v is proportional to (Cd A)^(-1/3).
        Only meaningful against the worst angle, so it is 0 against the best.
    Everything stays a plain float so the state manager can blend it linearly between angles.
    """
    best = bool(ref.get("reference_is_best"))
    power, ref_power = metrics["eval_drag_power_kw"], ref["eval_drag_power_kw"]
    fuel, co2 = fuel_and_co2(
        ref_power if best else power, power if best else ref_power,
        float(obj_config["fuel_burn_factor_l_per_kwh"]), float(obj_config["co2_kg_per_l_fuel"]),
    )
    cda = metrics["drag_coefficient_cd"] * metrics["frontal_area_m2"]
    ref_cda = ref["drag_coefficient_cd"] * ref["frontal_area_m2"]
    metrics["fuel_saved_l_per_hr"] = float(fuel)
    metrics["co2_saved_kg_per_hr"] = float(co2)
    metrics["speed_gain_pct"] = 0.0 if best else float(speed_gain_pct(cda, ref_cda))
    metrics["reference_direction_deg"] = float(ref["direction_deg"])


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
    region = _clip(grid, box, (u_name, p_name, k_name, NUT_NAME))
    del grid
    region = ns_term_fields(region, cfd.kinematic_viscosity_m2_s, u_name, p_name, NUT_NAME)
    if k_name not in region.point_data:  # a cell-only export: average k onto the points like U and p
        region.point_data[k_name] = region.cell_data_to_point_data().point_data[k_name]
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
    # The results are rotated back by the config's angle for this index, so a run made for a different
    # angle (the config was edited since) would be drawn at the wrong rotation: treat it as not finished.
    for idx, (result, meta) in sorted(runs.items()):
        deg_run, deg_cfg = float(meta["direction_deg"]), float(cfd.directions_deg[idx])
        if abs(deg_run - deg_cfg) > DIRECTION_TOLERANCE_DEG:
            warnings.warn(f"{meta['run_id']}: run.json says {deg_run:g} deg but the config has {deg_cfg:g} deg "
                          f"for direction {idx}; skipping this result (re-run it or restore the config)", stacklevel=2)
            del runs[idx]
    if not runs:
        raise ValueError(f"No finished runs for {obj_id} that match the angles in the config")

    heights = wand_heights(model0, settings.wand_count, cfd.floor)
    states = {
        idx: _import_direction(idx, result, meta, settings, cfd, model0, obj_config, heights)
        for idx, (result, meta) in sorted(runs.items())
    }

    # The sustainability reference is the object's own worst FINISHED direction (largest drag power), or for an
    # aircraft its best one (largest L/D, so the least level-flight drag). It is only known once every finished
    # direction has been read, so the savings are filled in as a second pass.
    if next(iter(states.values()))["metrics"]["reference_is_best"]:
        ref_idx = max(states, key=lambda j: states[j]["metrics"]["lift_to_drag"])
    else:
        ref_idx = max(states, key=lambda j: states[j]["metrics"]["eval_drag_power_kw"])
    ref_metrics = dict(states[ref_idx]["metrics"])
    for st in states.values():
        _apply_reference(st["metrics"], ref_metrics, obj_config)

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
            metrics["direction_deg"] = float(cfd.directions_deg[idx])  # the slot's own angle; the numbers are the source's
        s = state_key(idx)
        payload[f"{s}_metrics"] = json.dumps(metrics)
        payload[f"{s}_pressure"] = src["pressure"]
        payload[f"{s}_ns_terms"] = json.dumps(src["ns_terms"])
        for w, (lines, k, dominant, speed) in enumerate(src["wands"]):
            key = wand_key(idx, w)
            payload[f"{key}_lines"], payload[f"{key}_k"], payload[f"{key}_dominant"] = lines, k, dominant
            payload[f"{key}_speed"] = speed
    write_cache(out_path, payload)

    return {
        "imported": sorted(states),
        "filled": filled,
        "flagged": sorted(i for i, (_, meta) in runs.items() if meta["status"] == "flagged"),
    }
