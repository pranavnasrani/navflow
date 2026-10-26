"""A small valid v2 cache for tests of the state manager and the viewer (no OpenFOAM needed).

Pressure in direction d is the constant `d` (pascals), so linear interpolation is easy to check:
halfway between directions 2 and 3 every vertex must read 2.5. Streamlines are straight lines
along +x at the wand height.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from app.cache_format import FORMAT_VERSION, state_key, wand_key, write_cache

_N_LINES = 5  # straight lines per wand
_POINTS_PER_LINE = 6


def _rotation_y(deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def write_fake_cache(path: Path, obj_config: dict, n_dir: int = 9, n_wand: int = 10) -> None:
    box = trimesh.creation.box(extents=(4.0, 2.0, 2.0))
    directions = np.arange(n_dir) * 2.0
    heights = np.linspace(-1.0, 1.0, n_wand)
    has_wind_load = "wind_load" in obj_config.get("cfd", {}).get("widgets", ())

    payload: dict = {
        "format_version": np.array(FORMAT_VERSION),
        "mesh_vertices": np.asarray(box.vertices, dtype=np.float32),
        "mesh_faces": np.asarray(box.faces, dtype=np.int32),
        "directions_deg": directions,
        "wand_heights": heights,
        "wind_vectors": np.array([_rotation_y(a).T @ [1.0, 0.0, 0.0] for a in directions]),
        "view_bounds": np.array([-6.0, 14.0, -4.0, 4.0, -2.0, 2.0]),
        "config_json": json.dumps(obj_config),
    }

    # One line: points along +x, then a NaN separator row (and the matching k / dominant entries).
    xs = np.linspace(-5.0, 5.0, _POINTS_PER_LINE)
    ys = np.linspace(-1.5, 1.5, _N_LINES)
    for d in range(n_dir):
        s = state_key(d)
        metrics = {
            "direction_deg": float(directions[d]), "wind_speed_m_s": 40.0, "reynolds_number": 1.0e7,
            "drag_force_n": 100.0 + d, "side_force_n": 1.0, "lift_force_n": -5.0,
            "drag_coefficient_cd": 0.3 + 0.01 * d, "lift_coefficient_cl": -0.1,
            "frontal_area_m2": 2.0, "drag_power_kw": 4.0 + d, "base_moment_nm": 50.0,
            "iterations": 300, "converged": True, "filled_from": None,
            "fuel_saved_l_per_hr": 1.0, "co2_saved_kg_per_hr": 2.5, "speed_gain_pct": 10.0,
        }
        if has_wind_load:
            metrics["wind_load"] = [{"label": "Breeze", "v_m_s": 3.33, "force_n": 1.0, "moment_nm": 2.0}]
        payload[f"{s}_metrics"] = json.dumps(metrics)
        payload[f"{s}_pressure"] = np.full(len(box.vertices), float(d), dtype=np.float32)
        payload[f"{s}_ns_terms"] = json.dumps({"convective": 1.0, "pressure": 2.0, "viscous": 0.5})
        for w in range(n_wand):
            pts, k, dom = [], [], []
            for y in ys:
                pts.append(np.column_stack([xs, np.full_like(xs, y), np.full_like(xs, heights[w])]))
                pts.append(np.full((1, 3), np.nan))
                k.append(np.linspace(0.5, 1.5, _POINTS_PER_LINE))
                k.append([np.nan])
                dom.append(np.arange(_POINTS_PER_LINE) % 3)
                dom.append([-1])
            key = wand_key(d, w)
            payload[f"{key}_lines"] = np.vstack(pts).astype(np.float32)
            payload[f"{key}_k"] = np.concatenate(k).astype(np.float32)
            payload[f"{key}_dominant"] = np.concatenate(dom).astype(np.int8)
    write_cache(path, payload)
