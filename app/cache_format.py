"""Version 2 of the per-object results cache (assets/cache/<obj>_states.npz).

One cache file holds one object. Its 9 wind directions are the "states" d0..d8; for each state
there are 10 smoke-wand heights w0..w9. Everything the display needs is read into RAM once, so a
state lookup is a dictionary access (well under 1 ms).

Keys (all stored with np.savez_compressed, no pickles):
    format_version                      int, always 2
    mesh_vertices (V,3) f32, mesh_faces (F,3) i32        the model, in its own (un-rotated) frame
    directions_deg (9,), wand_heights (10,)              the sweep axes
    wind_vectors (9,3)                                   wind direction in the model frame
    view_bounds (6,)                                     xmin xmax ymin ymax zmin zmax of the display box
    config_json                                          the object's config entry, as JSON text
    d<i>_metrics                                         JSON dict of numbers (see solver/foam/importer.py)
    d<i>_pressure (V,) f32                               surface pressure in pascals, one value per vertex
    d<i>_ns_terms                                        JSON {convective, pressure, viscous}, m/s^2
    d<i>_w<j>_lines (M,3) f32                            streamlines in the model frame, joined with NaN rows
    d<i>_w<j>_k (M,) f32                                 turbulent kinetic energy along the lines (NaN at separators)
    d<i>_w<j>_dominant (M,) int8                         0 convective, 1 pressure, 2 viscous, -1 at separators

Maths link (IGCSE 0606): the display blends neighbouring directions by linear interpolation,
y = (1 - t) y1 + t y2 with t = (a - a1)/(a2 - a1), so every blended quantity is stored as a plain
number or array (JSON text is only used for the dictionaries that are not blended).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2


def state_key(d: int) -> str:
    """Prefix of everything stored for wind direction index `d`."""
    return f"d{d}"


def wand_key(d: int, w: int) -> str:
    """Prefix of the streamlines of wand `w` in wind direction `d`."""
    return f"d{d}_w{w}"


def write_cache(path: Path | str, payload: dict) -> None:
    """Write `payload` as a compressed .npz, atomically (a half-written cache is never visible)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")  # np.savez keeps names that already end in .npz
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def read_cache(path: Path | str) -> dict:
    """Load a v2 cache fully into RAM. Text entries come back as `str`, format_version as `int`.

    Raises ValueError for any file that is not a v2 cache (including the old v1 caches).
    """
    out: dict = {}
    with np.load(Path(path), allow_pickle=False) as z:
        for name in z.files:
            arr = z[name]
            if arr.ndim == 0 and arr.dtype.kind == "U":
                out[name] = str(arr)
            elif name == "format_version":
                out[name] = int(arr)
            else:
                out[name] = arr
    if out.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: cache format {out.get('format_version')!r} is not version {FORMAT_VERSION}; "
            "re-run solver/import_results.py"
        )
    return out
