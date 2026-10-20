"""
In-Memory State Manager and Linear Interpolation Engine.
Loads pre-computed CFD states from .npz bundles into RAM for instant (<1ms) switching.
Implements IGCSE 0606 Linear Interpolation between discrete wind speeds.
"""

from pathlib import Path
import json
import numpy as np


class ObjectStateManager:
    def __init__(self, obj_id: str, cache_dir: Path | str = "assets/cache"):
        self.obj_id = obj_id
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / f"{obj_id}_states.npz"

        if not self.cache_file.exists():
            raise FileNotFoundError(
                f"Cache file {self.cache_file} not found. "
                f"Please run 'python3 solver/batch_precompute.py' first."
            )

        # Load NPZ into memory
        self.data = np.load(self.cache_file, allow_pickle=True)
        self.z_steps = self.data["z_steps"]
        self.wind_speeds = self.data["wind_speeds"]
        self.mesh_vertices = self.data["mesh_vertices"]
        self.mesh_faces = self.data["mesh_faces"]
        self.obj_config = json.loads(str(self.data["config_json"]))

        # Cache decoded metrics and arrays for zero-latency retrieval
        self.state_cache = {}
        for z_i in range(len(self.z_steps)):
            for v_i in range(len(self.wind_speeds)):
                k = f"z{z_i}_v{v_i}"
                metrics = json.loads(str(self.data[f"{k}_metrics"]))
                pressure = self.data[f"{k}_pressure"]
                streamlines = self.data[f"{k}_streamlines"]
                self.state_cache[k] = {
                    "metrics": metrics,
                    "pressure": pressure,
                    "streamlines": streamlines
                }

    def get_state(self, z_step_index: int, wind_speed: float) -> dict:
        """
        Retrieves the CFD state for a given discrete Z-step index (0..9)
        and continuous wind speed (V), using IGCSE 0606 Linear Interpolation:
           y = (1 - t)*y1 + t*y2,  where t = (V - V1) / (V2 - V1)
        """
        z_idx = np.clip(z_step_index, 0, len(self.z_steps) - 1)
        z_val = float(self.z_steps[z_idx])

        # Clamp wind speed within range
        ws = float(np.clip(wind_speed, self.wind_speeds[0], self.wind_speeds[-1]))

        # Find adjacent velocity indices
        if ws <= self.wind_speeds[0]:
            v1_idx, v2_idx = 0, 0
            t = 0.0
        elif ws >= self.wind_speeds[-1]:
            v1_idx = len(self.wind_speeds) - 1
            v2_idx = v1_idx
            t = 0.0
        else:
            v1_idx = int(np.searchsorted(self.wind_speeds, ws) - 1)
            v2_idx = v1_idx + 1
            v1 = self.wind_speeds[v1_idx]
            v2 = self.wind_speeds[v2_idx]
            t = (ws - v1) / (v2 - v1)  # Linear fraction in [0, 1]

        state1 = self.state_cache[f"z{z_idx}_v{v1_idx}"]
        state2 = self.state_cache[f"z{z_idx}_v{v2_idx}"]

        # 1. Linearly interpolate surface pressure
        p1 = state1["pressure"]
        p2 = state2["pressure"]
        interp_pressure = (1.0 - t) * p1 + t * p2

        # 2. Linearly interpolate telemetry metrics
        m1 = state1["metrics"]
        m2 = state2["metrics"]
        interp_metrics = {}
        for key in m1:
            val1 = m1[key]
            val2 = m2[key]
            if isinstance(val1, (int, float)):
                interp_metrics[key] = (1.0 - t) * float(val1) + t * float(val2)
            else:
                interp_metrics[key] = val1

        # Use nearest velocity streamlines for 3D trajectory
        active_streamlines = state2["streamlines"] if t > 0.5 else state1["streamlines"]

        return {
            "z_index": int(z_idx),
            "z_val": float(z_val),
            "wind_speed": float(ws),
            "metrics": interp_metrics,
            "surface_pressure": interp_pressure,
            "streamlines": active_streamlines,
            "mesh_vertices": self.mesh_vertices,
            "mesh_faces": self.mesh_faces,
            "obj_config": self.obj_config
        }
