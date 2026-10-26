"""
In-Memory State Manager and Linear Interpolation Engine (cache format v2).

Loads one object's OpenFOAM results (assets/cache/<obj>_states.npz) fully into RAM, decoding every
JSON entry once at load time, so that `get_state` is a handful of numpy operations (well under 1 ms).

The visitor controls two things:
  * the smoke-wand height, one of 10 discrete steps (the macropad buttons);
  * the wind direction, a continuous knob in [0, 1] mapped onto the object's direction sweep.

Maths link (IGCSE 0606): the knob maps to a direction by a straight line,
    d = d_min + knob * (d_max - d_min),
and a direction between two solved directions d1 < d < d2 is blended by linear interpolation,
    t = (d - d1) / (d2 - d1)  in [0, 1],     y = (1 - t) * y1 + t * y2,
applied to the surface pressure, the telemetry numbers and the Navier-Stokes terms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.cache_format import read_cache, state_key, wand_key


def _is_number(value) -> bool:
    """True for int/float values that are not bool (bool is a subclass of int in Python)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _blend_numbers(a: dict, b: dict, t: float, nearest: dict) -> dict:
    """y = (1 - t) * y1 + t * y2 for numeric entries; everything else comes from `nearest`."""
    out = {}
    for key, v_near in nearest.items():
        v1, v2 = a.get(key), b.get(key)
        if _is_number(v1) and _is_number(v2):
            out[key] = (1.0 - t) * float(v1) + t * float(v2)
        else:
            out[key] = v_near
    return out


def worst_wind_direction(directions_deg, metrics_list: list[dict]) -> float | None:
    """The solved direction (theta in degrees) whose wind load is largest, or None if there is no wind-load table.

    "Wind load" is the force in the LAST row of the table (the strongest reference wind, a hurricane): every row is
    the same simulated force times (v / v_sim)^2, so the largest row-force is also the largest simulated force.
    Directions that were filled from a neighbour (`filled_from` is not None) only copy that neighbour's numbers, so
    they are not candidates. Ties go to the lower direction (max returns the first).
    """
    best: tuple[float, float] | None = None
    for deg, m in zip(directions_deg, metrics_list):
        rows = m.get("wind_load")
        if not rows or m.get("filled_from") is not None:
            continue
        force = float(rows[-1]["force_n"])
        if best is None or force > best[0]:
            best = (force, float(deg))
    return None if best is None else best[1]


class ObjectStateManager:
    def __init__(self, obj_id: str, cache_dir: Path | str = "assets/cache"):
        self.obj_id = obj_id
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / f"{obj_id}_states.npz"

        if not self.cache_file.exists():
            raise FileNotFoundError(
                f"Cache file {self.cache_file} not found. "
                f"Please run 'python3 solver/import_results.py' first."
            )

        # Raises ValueError for a v1 (or any non-v2) cache.
        data = read_cache(self.cache_file)

        self.directions_deg = np.asarray(data["directions_deg"], dtype=float)
        self.wand_heights = np.asarray(data["wand_heights"], dtype=float)
        self.view_bounds = np.asarray(data["view_bounds"], dtype=float)
        self.mesh_vertices = data["mesh_vertices"]
        self.mesh_faces = data["mesh_faces"]
        self.obj_config = json.loads(data["config_json"])
        self.wind_speed = float(self.obj_config["cfd"]["speed_m_s"])

        n_dir, n_wand = len(self.directions_deg), len(self.wand_heights)
        wind_vectors = np.asarray(data["wind_vectors"], dtype=float)

        self.worst_direction_deg: float | None = None  # set below, once the metrics are decoded

        # Decode everything now so get_state never touches JSON or the npz file.
        self._wind_vectors = wind_vectors
        self._metrics = []
        self._ns_terms = []
        self._pressure = []
        self._wands = []  # _wands[d][w] = (lines, k, dominant)
        for d in range(n_dir):
            s = state_key(d)
            self._metrics.append(json.loads(data[f"{s}_metrics"]))
            self._ns_terms.append(json.loads(data[f"{s}_ns_terms"]))
            self._pressure.append(np.asarray(data[f"{s}_pressure"]))
            self._wands.append([
                (
                    data[f"{wand_key(d, w)}_lines"],
                    data[f"{wand_key(d, w)}_k"],
                    data[f"{wand_key(d, w)}_dominant"],
                )
                for w in range(n_wand)
            ])

        # Computed once here (not per frame, and not from the angles a visitor happened to visit).
        self.worst_direction_deg = worst_wind_direction(self.directions_deg, self._metrics)

    def direction_for_knob(self, knob: float) -> float:
        """Straight-line map d = d_min + knob * (d_max - d_min), with the knob clamped to [0, 1]."""
        k = min(max(float(knob), 0.0), 1.0)
        d_min, d_max = self.directions_deg[0], self.directions_deg[-1]
        return float(d_min + k * (d_max - d_min))

    def _bracket(self, direction: float) -> tuple[int, int, float]:
        """Indices (i1, i2) of the solved directions around `direction`, and t = (d - d1)/(d2 - d1)."""
        dirs = self.directions_deg
        last = len(dirs) - 1
        i1 = int(np.searchsorted(dirs, direction, side="right")) - 1
        i1 = min(max(i1, 0), last)
        if i1 == last:  # at (or past) the upper end: the last direction itself, t = 0
            return last, last, 0.0
        i2 = i1 + 1
        t = (direction - dirs[i1]) / (dirs[i2] - dirs[i1])
        return i1, i2, float(min(max(t, 0.0), 1.0))

    def get_state(self, wand_index: int, knob: float) -> dict:
        """
        State for smoke-wand `wand_index` (0..9) and wind-direction `knob` (0..1).

        Surface pressure, telemetry metrics and N-S terms use IGCSE 0606 linear interpolation
        between the two bracketing solved directions:
            t = (d - d1) / (d2 - d1),   y = (1 - t) * y1 + t * y2.
        Streamlines, k, dominant-term colouring and `filled` come from the nearest direction
        (t > 0.5 picks d2), since lines cannot be blended point by point.
        """
        w = min(max(int(wand_index), 0), len(self.wand_heights) - 1)
        d = self.direction_for_knob(knob)
        i1, i2, t = self._bracket(d)
        near = i2 if t > 0.5 else i1

        # 1. Surface pressure: y = (1 - t) * p1 + t * p2 (one numpy expression over all vertices).
        pressure = (1.0 - t) * self._pressure[i1] + t * self._pressure[i2]

        # 2. Telemetry and Navier-Stokes terms, blended number by number.
        metrics = _blend_numbers(self._metrics[i1], self._metrics[i2], t, self._metrics[near])
        ns_terms = _blend_numbers(self._ns_terms[i1], self._ns_terms[i2], t, self._ns_terms[near])

        # 3. Wind direction in the model frame: blend the two unit vectors, then renormalise.
        vec = (1.0 - t) * self._wind_vectors[i1] + t * self._wind_vectors[i2]
        norm = float(np.linalg.norm(vec))
        wind_vector = vec / norm if norm > 0.0 else self._wind_vectors[near].copy()

        lines, k, dominant = self._wands[near][w]
        return {
            "wand_index": w,
            "wand_z": float(self.wand_heights[w]),
            "direction_deg": d,
            "wind_speed": self.wind_speed,
            "wind_vector": wind_vector,
            "metrics": metrics,
            "worst_direction_deg": self.worst_direction_deg,  # None unless the object has a wind-load table
            "ns_terms": ns_terms,
            "surface_pressure": pressure,
            "lines": lines,
            "k": k,
            "dominant": dominant,
            "filled": self._metrics[near].get("filled_from") is not None,
            "mesh_vertices": self.mesh_vertices,
            "mesh_faces": self.mesh_faces,
            "obj_config": self.obj_config,
        }
