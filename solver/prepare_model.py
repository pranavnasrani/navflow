"""Prepare a downloaded 3D model for the OpenFOAM wind tunnel.

    ./.venv/bin/python solver/prepare_model.py --object airplane [--src file] [--out model.stl]

Reads a model (STL/OBJ/PLY/GLB/GLTF), merges it into one closed surface, rotates it by the
object's base rotation (nose towards -x), scales it to real size, places it in tunnel
coordinates, then writes assets/models/<id>/model.stl and a preview.png showing the model with
the wind arrow along +x so a human can check "nose faces the wind, up is up" before it is used.

Maths link (IGCSE 0606): the real-size scale factor is real length / model length (a ratio),
and the rotation uses sin/cos of the base angles (see solver/foam/geometry.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.foam.geometry import ModelNotWatertight, frontal_area, prepare_model
from solver.foam.settings import load_settings

READABLE = {".stl", ".obj", ".ply", ".glb", ".gltf"}
CAD_INSTRUCTIONS = """\
{name}: STEP and FBX files cannot be read directly. Convert first:
  - STEP (.step/.stp) with FreeCAD: open STEP -> Mesh Design -> "Create mesh from shape" -> export STL.
  - FBX with Blender: open FBX -> join all parts (Ctrl+J) -> export STL.
Then re-run with --src pointing at the .stl (or drop it in assets/models/incoming/<object>/)."""


def find_incoming(obj_id: str) -> Path:
    """The single model file in assets/models/incoming/<id>/."""
    folder = PROJECT_ROOT / "assets" / "models" / "incoming" / obj_id
    files = sorted(p for p in folder.glob("*") if p.is_file() and not p.name.startswith(".")) if folder.is_dir() else []
    if len(files) != 1:
        sys.exit(f"Expected exactly one model file in {folder}, found {len(files)}. Use --src to name the file.")
    return files[0]


def write_preview(mesh, path: Path) -> bool:
    """Off-screen PNG of the model with a wind arrow along +x. Returns False if rendering fails."""
    try:
        import pyvista as pv

        pv.OFF_SCREEN = True
        plotter = pv.Plotter(off_screen=True, window_size=(1000, 700))
        plotter.set_background("#14181f")
        faces = [[3, *f] for f in mesh.faces]
        poly = pv.PolyData(mesh.vertices, faces)
        plotter.add_mesh(poly, color="#c8d3e0", smooth_shading=True)
        lo, hi = mesh.bounds
        length = float(mesh.extents.max())
        start = (lo[0] - 0.9 * length, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
        plotter.add_mesh(pv.Arrow(start=start, direction=(1, 0, 0), scale=0.7 * length), color="#4aa3ff")
        plotter.add_text("wind  ->  +x", position="upper_left", font_size=10, color="white")
        plotter.add_axes()
        plotter.camera_position = "xz"  # side view: wind left to right, up is up
        plotter.reset_camera()
        plotter.screenshot(str(path))
        plotter.close()
        return True
    except Exception as exc:  # no GPU/display in some environments: the model is still written
        print(f"warning: preview not written ({type(exc).__name__}: {exc})", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--object", required=True, dest="obj_id", help="airplane | f1_car | building | human")
    ap.add_argument("--src", type=Path, help="model file (default: the single file in assets/models/incoming/<id>/)")
    ap.add_argument("--out", type=Path, help="output STL (default: assets/models/<id>/model.stl); preview.png goes next to it")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "exhibition_config.json")
    args = ap.parse_args(argv)

    _, objects, _ = load_settings(args.config)
    if args.obj_id not in objects:
        sys.exit(f"Unknown object {args.obj_id!r}; choose from {sorted(objects)}")
    cfd = objects[args.obj_id]

    src = args.src or find_incoming(args.obj_id)
    if src.suffix.lower() not in READABLE:
        if src.suffix.lower() in {".step", ".stp", ".fbx"}:
            sys.exit(CAD_INSTRUCTIONS.format(name=src.name))
        sys.exit(f"Unsupported model format {src.suffix!r}. Readable: {', '.join(sorted(READABLE))}.")

    try:
        mesh = prepare_model(src, cfd)
    except ModelNotWatertight as exc:
        sys.exit(f"{src.name}: cannot make one closed surface: {exc}\n"
                 "Use a model made for 3D printing, or repair it (Blender: Mesh > Clean Up > Fill Holes).")

    out = args.out or PROJECT_ROOT / "assets" / "models" / args.obj_id / "model.stl"
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    preview = out.with_name("preview.png")
    wrote_preview = write_preview(mesh, preview)

    ex = mesh.extents
    print(f"{args.obj_id}: {src.name} -> {out}")
    print(f"  extents (m): x={ex[0]:.3f} y={ex[1]:.3f} z={ex[2]:.3f}   triangles: {len(mesh.faces)}")
    print(f"  frontal area: {frontal_area(mesh):.3f} m^2")
    if wrote_preview:
        print(f"  preview: {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
