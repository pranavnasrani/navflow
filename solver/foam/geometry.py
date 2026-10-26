"""Model preparation and wind-tunnel geometry for the OpenFOAM pipeline.

Conventions (shared by the case writer, runner and importer):
  * The wind blows along +x. The model's nose faces -x, towards the inlet.
  * The tunnel is always axis-aligned. To test another wind direction we ROTATE THE MODEL
    (never the tunnel) with `run_geometry`, then re-mesh.
  * Units are metres once `prepare_model` has scaled the model to its real size.

Maths link (IGCSE 0606):
  * Rotation about an axis is a 2x2 rotation block built from sin/cos (circular measure,
    trigonometry): x' = x cos(theta) - y sin(theta), y' = x sin(theta) + y cos(theta).
  * Scaling is multiplication by the ratio real size / model size (linear scaling).
  * Wand heights are an arithmetic progression a, a+d, a+2d, ... (np.linspace).
  * The frontal area is an area under a shape's outline, here counted as pixels x pixel area.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from solver.foam.settings import ObjectCfd

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class ModelNotWatertight(ValueError):
    """Raised when a model cannot be turned into one closed surface."""


def rotation_matrix(axis: str, deg: float) -> np.ndarray:
    """3x3 rotation about "x", "y" or "z" by `deg` degrees, right-hand rule.

    Right-hand rule: with the thumb along the axis, positive angles curl the fingers.
    Rz(90) takes +x to +y; Ry(90) takes +x to -z (so a positive y rotation lifts the -x nose).
    """
    if axis not in _AXIS_INDEX:
        raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _open_edge_count(mesh: trimesh.Trimesh) -> int:
    """Edges used by only one triangle. A closed surface has none."""
    return int(len(trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)))


def _as_single_mesh(loaded) -> trimesh.Trimesh:
    """STL/OBJ/PLY give a Trimesh; GLB/GLTF give a Scene, so merge its geometry (transforms applied)."""
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ModelNotWatertight("model file contains no triangle surface")
    return loaded


def _split_bodies(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Split into edge-connected bodies (scipy engine, so networkx is not needed)."""
    groups = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), engine="scipy"
    )
    return [trimesh.Trimesh(mesh.vertices, mesh.faces[g], process=True) for g in groups]


def _make_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return one closed surface: split the file into bodies and union them with manifold."""
    if mesh.is_watertight and mesh.body_count == 1:
        return mesh
    bodies = _split_bodies(mesh)
    try:
        merged = trimesh.boolean.union(list(bodies), engine="manifold")
    except Exception as exc:  # manifold refuses non-closed or degenerate input
        raise ModelNotWatertight(
            f"could not union {len(bodies)} body(ies) ({_open_edge_count(mesh)} open edges, holes): {exc}"
        ) from exc
    if not merged.is_watertight:
        raise ModelNotWatertight(f"model is still not watertight after union: {_open_edge_count(merged)} open edges (holes)")
    return merged


def prepare_model(src: Path, cfd: ObjectCfd) -> trimesh.Trimesh:
    """Load `src` and return one closed, real-size mesh placed in tunnel coordinates.

    Steps: load, base rotation (x then y then z), union into one watertight body,
    scale so the extent along `cfd.size_axis` equals `cfd.real_size_m`, then centre the
    bbox at x = y = 0 with z bottom = 0 (floor) or z centred at 0 (free air).
    """
    mesh = _as_single_mesh(trimesh.load(str(src), force="mesh"))

    rx, ry, rz = (float(a) for a in cfd.base_rotation_deg)
    r = rotation_matrix("z", rz) @ rotation_matrix("y", ry) @ rotation_matrix("x", rx)
    transform = np.eye(4)
    transform[:3, :3] = r
    mesh = mesh.copy()
    mesh.apply_transform(transform)

    mesh = _make_watertight(mesh)
    if mesh.volume < 0:  # inside-out normals: flip so the volume is positive
        mesh.invert()

    # Linear scaling: factor = real length / model length along the size axis.
    mesh.apply_scale(cfd.real_size_m / mesh.extents[_AXIS_INDEX[cfd.size_axis]])

    lo, hi = mesh.bounds
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2] if cfd.floor != "none" else -(lo[2] + hi[2]) / 2])
    mesh.apply_translation(shift)
    return mesh


def run_geometry(model0: trimesh.Trimesh, cfd: ObjectCfd, deg: float) -> trimesh.Trimesh:
    """Copy of `model0` rotated by `deg` about the origin, around `cfd.rotation_axis`."""
    out = model0.copy()
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix(cfd.rotation_axis, deg)
    out.apply_transform(transform)
    return out


def tunnel_bounds(model: trimesh.Trimesh, factors: dict, floor: str) -> tuple[float, float, float, float, float, float]:
    """Wind-tunnel box (xmin, xmax, ymin, ymax, zmin, zmax) sized in units of D = longest bbox side.

    x: [min - upstream*D, max + downstream*D];  y: [min - lateral*D, max + lateral*D];
    z: [0, max + top*D] with a floor, else [min - top*D, max + top*D].
    """
    lo, hi = model.bounds
    d = float(model.extents.max())
    zmin = 0.0 if floor != "none" else lo[2] - factors["top"] * d
    return (
        float(lo[0] - factors["upstream"] * d), float(hi[0] + factors["downstream"] * d),
        float(lo[1] - factors["lateral"] * d), float(hi[1] + factors["lateral"] * d),
        float(zmin), float(hi[2] + factors["top"] * d),
    )


def frontal_area(model: trimesh.Trimesh, resolution: int = 400) -> float:
    """Silhouette area seen by the wind (projection onto the y-z plane), in m^2.

    Each triangle is projected (x dropped) and filled on a `resolution`-pixel raster; the area is
    (filled pixels) x (pixel area). Overlapping triangles (front and back faces) just re-fill the
    same pixels, so the result is the true outline area, not a sum of triangle areas.
    """
    yz = model.vertices[:, 1:3]
    lo, hi = yz.min(axis=0), yz.max(axis=0)
    pixel = float((hi - lo).max()) / resolution
    if pixel == 0.0:
        return 0.0
    px = (yz - lo) / pixel + 1.0  # one pixel of padding
    size = (int(np.ceil((hi[0] - lo[0]) / pixel)) + 3, int(np.ceil((hi[1] - lo[1]) / pixel)) + 3)
    img = Image.new("1", size, 0)
    draw = ImageDraw.Draw(img)
    tri = px[model.faces]  # (n, 3, 2)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]  # 2D cross product = twice the signed area
    for t in tri[np.abs(cross) > 1e-9]:  # edge-on triangles project to a line: skip
        draw.polygon([tuple(p) for p in t], fill=1)
    return float(np.count_nonzero(np.asarray(img))) * pixel * pixel


def wand_heights(model0: trimesh.Trimesh, count: int) -> np.ndarray:
    """`count` evenly spaced z heights from zmin - 0.1H to zmax + 0.1H (H = model height)."""
    lo, hi = model0.bounds[:, 2]
    h = hi - lo
    return np.linspace(lo - 0.1 * h, hi + 0.1 * h, count)


def seed_rake(run_model: trimesh.Trimesh, z: float, n: int) -> np.ndarray:
    """(n, 3) streamline seed points on a line across the flow, upstream of the model.

    x = xmin - 0.5*Lx; y spans [ymin - 0.1W, ymax + 0.1W]; all at height `z`.
    """
    lo, hi = run_model.bounds
    lx, w = hi[0] - lo[0], hi[1] - lo[1]
    ys = np.linspace(lo[1] - 0.1 * w, hi[1] + 0.1 * w, n)
    return np.column_stack([np.full(n, lo[0] - 0.5 * lx), ys, np.full(n, z)])


def view_bounds(model0: trimesh.Trimesh, floor: str) -> tuple[float, float, float, float, float, float]:
    """Display box: x [xmin - L, xmax + 3L], y [ymin - W, ymax + W], z [max(zmin - H/2, floor), zmax + H/2].

    `floor` is the config string; the lower z clamp is 0 when it is not "none", else there is none.
    """
    lo, hi = model0.bounds
    lx, w, h = hi - lo
    zmin = lo[2] - 0.5 * h
    if floor != "none":
        zmin = max(zmin, 0.0)
    return (
        float(lo[0] - lx), float(hi[0] + 3 * lx),
        float(lo[1] - w), float(hi[1] + w),
        float(zmin), float(hi[2] + 0.5 * h),
    )
