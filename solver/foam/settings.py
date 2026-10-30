"""Typed settings for the OpenFOAM pipeline, read from config/exhibition_config.json.

Two frozen dataclasses carry everything later stages need:
  * FoamSettings - run-wide solver, convergence and meshing numbers (the "openfoam" block).
  * ObjectCfd    - per-object physics: fixed speed, real-world size, wind-direction sweep.

Maths link (IGCSE 0606): each object sweeps 9 wind directions, evenly spaced, so the
direction list is an arithmetic progression a, a+d, a+2d, ... The convergence test compares
the change in drag/lift over a window of iterations against a small fraction of their mean.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_FLOORS = ("none", "moving", "rough", "slip")
VALID_ROTATION_AXES = ("y", "z")
VALID_SIZE_AXES = ("x", "z")
N_DIRECTIONS = 9


@dataclass(frozen=True)
class FoamSettings:
    image: str
    volume: str
    n_procs: int
    max_iterations: int
    preview_iterations: int
    min_iterations: int
    convergence_window: int
    convergence_tolerance: float
    write_interval: int
    average_window: int
    domain_factors: dict[str, float]  # keys: upstream downstream lateral top
    base_cells_per_size: int
    surface_refinement: tuple[int, int]
    wake_refinement: int
    n_layers: int
    wand_count: int
    seeds_per_wand: int


@dataclass(frozen=True)
class ObjectCfd:
    obj_id: str
    speed_m_s: float
    real_size_m: float
    size_axis: str  # "x" or "z"
    base_rotation_deg: tuple[float, float, float]
    rotation_axis: str  # "y" or "z"
    directions_deg: tuple[float, ...]
    direction_label: str
    floor: str  # "none" | "moving" | "rough" | "slip"
    density_kg_m3: float
    kinematic_viscosity_m2_s: float
    cruise: dict | None  # {"density_kg_m3", "speed_m_s", "altitude_km", "mass_kg"} (mass: level-flight fuel)
    abl_z0_m: float | None
    widgets: tuple[str, ...]
    surface_refinement: tuple[int, int] | None  # per-object override


def _build_object(obj_id: str, c: dict) -> ObjectCfd:
    floor = c["floor"]
    if floor not in VALID_FLOORS:
        raise ValueError(f"{obj_id}: floor {floor!r} must be one of {VALID_FLOORS}")
    axis = c["rotation_axis"]
    if axis not in VALID_ROTATION_AXES:
        raise ValueError(f"{obj_id}: rotation_axis {axis!r} must be one of {VALID_ROTATION_AXES}")
    size_axis = c["size_axis"]
    if size_axis not in VALID_SIZE_AXES:
        raise ValueError(f"{obj_id}: size_axis {size_axis!r} must be one of {VALID_SIZE_AXES}")
    dirs = tuple(float(d) for d in c["directions_deg"])
    if len(dirs) != N_DIRECTIONS:
        raise ValueError(f"{obj_id}: need {N_DIRECTIONS} directions, got {len(dirs)}")
    if any(b <= a for a, b in zip(dirs, dirs[1:])):
        raise ValueError(f"{obj_id}: directions_deg must be strictly ascending, got {list(dirs)}")
    refine = c.get("surface_refinement")
    return ObjectCfd(
        obj_id=obj_id,
        speed_m_s=float(c["speed_m_s"]),
        real_size_m=float(c["real_size_m"]),
        size_axis=size_axis,
        base_rotation_deg=tuple(float(a) for a in c["base_rotation_deg"]),
        rotation_axis=axis,
        directions_deg=dirs,
        direction_label=c["direction_label"],
        floor=floor,
        density_kg_m3=float(c["density_kg_m3"]),
        kinematic_viscosity_m2_s=float(c["kinematic_viscosity_m2_s"]),
        cruise=c.get("cruise"),
        abl_z0_m=None if c.get("abl_z0_m") is None else float(c["abl_z0_m"]),
        widgets=tuple(c.get("widgets", ())),
        surface_refinement=None if refine is None else (int(refine[0]), int(refine[1])),
    )


def load_settings(
    config_path: Path | str = "config/exhibition_config.json",
) -> tuple[FoamSettings, dict[str, ObjectCfd], dict]:
    """Read the config; return (run settings, per-object CFD settings by id, raw config dict)."""
    raw = json.loads(Path(config_path).read_text())
    f = raw["openfoam"]
    settings = FoamSettings(
        image=f["image"],
        volume=f["volume"],
        n_procs=int(f["n_procs"]),
        max_iterations=int(f["max_iterations"]),
        preview_iterations=int(f["preview_iterations"]),
        min_iterations=int(f["min_iterations"]),
        convergence_window=int(f["convergence_window"]),
        convergence_tolerance=float(f["convergence_tolerance"]),
        write_interval=int(f["write_interval"]),
        average_window=int(f["average_window"]),
        domain_factors={k: float(v) for k, v in f["domain_factors"].items()},
        base_cells_per_size=int(f["base_cells_per_size"]),
        surface_refinement=(int(f["surface_refinement"][0]), int(f["surface_refinement"][1])),
        wake_refinement=int(f["wake_refinement"]),
        n_layers=int(f["n_layers"]),
        wand_count=int(f["wand_count"]),
        seeds_per_wand=int(f["seeds_per_wand"]),
    )
    objects = {o["id"]: _build_object(o["id"], o["cfd"]) for o in raw["objects"]}
    return settings, objects, raw
