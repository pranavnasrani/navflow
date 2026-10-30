"""Prepare a downloaded 3D model for the OpenFOAM wind tunnel.

    ./.venv/bin/python solver/prepare_model.py --object airplane [--src file] [--out model.stl]
        [--wrap 500] [--drop gears]

Reads a model (STL/OBJ/PLY/GLB/GLTF/BLEND/FBX; all but STL/OBJ/PLY are first exported by Blender,
see solver/blender_export.py), merges it into one closed surface (--wrap shrink-wraps display
models made of open parts, see geometry.wrap_surface), rotates it by the
object's base rotation (nose towards -x), scales it to real size, places it in tunnel
coordinates, then writes assets/models/<id>/model.stl and a preview.png showing the model with
the wind arrow along +x so a human can check "nose faces the wind, up is up" before it is used.

Maths link (IGCSE 0606): the real-size scale factor is real length / model length (a ratio),
and the rotation uses sin/cos of the base angles (see solver/foam/geometry.py).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.foam.geometry import ModelNotWatertight, frontal_area, prepare_model
from solver.foam.settings import load_settings

READABLE = {".stl", ".obj", ".ply", ".glb", ".gltf"}
# glTF is y-up and trimesh keeps that; Blender turns every format z-up, the axes base_rotation_deg
# is set against. So .glb/.gltf go through Blender too when it is installed.
VIA_BLENDER = {".blend", ".fbx", ".glb", ".gltf"}
BLENDER_PATHS = ("/Applications/Blender.app/Contents/MacOS/Blender", "blender")
CAD_INSTRUCTIONS = """\
{name}: STEP files cannot be read directly. Convert first with FreeCAD:
  open STEP -> Mesh Design -> "Create mesh from shape" -> export STL.
Then re-run with --src pointing at the .stl (or drop it in assets/models/incoming/<object>/)."""


def find_blender() -> str | None:
    for path in BLENDER_PATHS:
        if Path(path).is_file() or shutil.which(path):
            return path
    return None


def export_with_blender(src: Path, out: Path, drop: str = "") -> Path:
    """Run solver/blender_export.py headless to turn a .blend/.fbx (or any Blender-readable file) into an STL."""
    blender = find_blender()
    if blender is None:
        sys.exit(f"{src.name}: needs Blender to convert (brew install --cask blender), or export an STL yourself.")
    script = Path(__file__).resolve().parent / "blender_export.py"
    cmd = [blender, "-b", "--factory-startup", "-P", str(script), "--", str(src), str(out)] + ([drop] if drop else [])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out.is_file():
        sys.exit(f"Blender could not export {src.name}:\n{result.stdout[-2000:]}{result.stderr[-2000:]}")
    return out


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
    ap.add_argument("--wrap", type=float, metavar="RES",
                    help="shrink-wrap a model made of open parts; cube size = longest side / RES (try 400-500)")
    ap.add_argument("--drop", default="", help="Blender object-name prefixes to leave out, comma-separated (e.g. gears)")
    args = ap.parse_args(argv)

    _, objects, _ = load_settings(args.config)
    if args.obj_id not in objects:
        sys.exit(f"Unknown object {args.obj_id!r}; choose from {sorted(objects)}")
    cfd = objects[args.obj_id]

    src = args.src or find_incoming(args.obj_id)
    tmp = tempfile.TemporaryDirectory()
    ext = src.suffix.lower()
    if ext in {".glb", ".gltf"} and find_blender() is None and not args.drop:
        print(f"warning: Blender not found, reading {src.name} with trimesh (y-up: check preview.png)", file=sys.stderr)
    elif ext in VIA_BLENDER or (args.drop and ext in READABLE):
        print(f"{src.name}: exporting with Blender...")
        src = export_with_blender(src, Path(tmp.name) / f"{src.stem}.stl", args.drop)
    elif ext not in READABLE:
        if ext in {".step", ".stp"}:
            sys.exit(CAD_INSTRUCTIONS.format(name=src.name))
        sys.exit(f"Unsupported model format {src.suffix!r}. Readable: {', '.join(sorted(READABLE | VIA_BLENDER))}.")

    try:
        mesh = prepare_model(src, cfd, wrap_resolution=args.wrap)
    except ModelNotWatertight as exc:
        hint = "try a higher --wrap" if args.wrap else "try --wrap 500 (shrink-wrap the open parts)"
        sys.exit(f"{src.name}: cannot make one closed surface: {exc}\n{hint}.")
    finally:
        tmp.cleanup()

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
