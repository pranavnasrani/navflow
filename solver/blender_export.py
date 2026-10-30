"""Export a Blender-readable model (.blend/.fbx/.glb/.gltf/.obj) as one joined STL, headless.

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -P solver/blender_export.py -- SRC OUT.stl [EXCLUDE]

solver/prepare_model.py runs this for you for .blend and .fbx files. EXCLUDE is a comma-separated
list of object-name prefixes to leave out (e.g. "gears" for a plane's landing gear). The STL is
still made of open parts; prepare_model's --wrap step closes it (solver/foam/geometry.wrap_surface).
"""
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = Path(argv[0]), Path(argv[1])
exclude = tuple(p for p in (argv[2].split(",") if len(argv) > 2 else []) if p)

ext = src.suffix.lower()
if ext == ".blend":
    bpy.ops.wm.open_mainfile(filepath=str(src))
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(src))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(src))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(src))
    else:
        sys.exit(f"unsupported input {ext}")

# Keep only meshes, with all parent/scale transforms baked in, then join them into one object.
# Parents are cleared first (keeping each part's world position): FBX files hang the parts under
# an empty that carries the Y-up -> Z-up rotation, and deleting it would drop that rotation.
for ob in bpy.data.objects:
    world = ob.matrix_world.copy()
    ob.parent = None
    ob.matrix_world = world
for ob in list(bpy.data.objects):
    if ob.type != "MESH" or ob.name.startswith(exclude):
        bpy.data.objects.remove(ob, do_unlink=True)
meshes = [ob for ob in bpy.data.objects if ob.type == "MESH"]
if not meshes:
    sys.exit("no mesh objects in file")
bpy.ops.object.select_all(action="DESELECT")
for ob in meshes:
    if ob.name not in bpy.context.view_layer.objects:
        bpy.context.scene.collection.objects.link(ob)
    ob.hide_set(False)
    ob.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.object.make_single_user(object=True, obdata=True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
if len(meshes) > 1:
    bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active

bpy.ops.object.select_all(action="DESELECT")
ob.select_set(True)
out.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.stl_export(filepath=str(out), export_selected_objects=True, apply_modifiers=True)
print(f"EXPORT_OK {out} dims={tuple(round(d, 4) for d in ob.dimensions)} faces={len(ob.data.polygons)} dropped={exclude}")
