"""
Generates clean, watertight starter 3D models (.stl) for the 4 exhibition objects:
1. Commercial Airplane (aerospace streamline profile)
2. Formula 1 Car (racing ground-effect aerodynamics)
3. Skyscraper Tower (aerodynamic tapering for vortex shedding mitigation)
4. Human Figure (benchmark bluff body for high drag flow separation)

These models are fully watertight and ready for voxelization and CFD solving.
Users can replace them with their own custom .stl or .obj files at any time.
"""

from pathlib import Path
import numpy as np
import trimesh


def create_airplane_mesh() -> trimesh.Trimesh:
    """Builds a streamlined commercial passenger jet (fuselage, wings, tail, engines)."""
    parts = []

    # Fuselage (main body cylinder + nose cone + tail taper)
    fuselage = trimesh.creation.cylinder(radius=0.06, height=0.7, sections=24)
    # Rotate cylinder along X-axis
    fuselage.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    fuselage.apply_translation([0.35, 0, 0])
    parts.append(fuselage)

    # Nose cone
    nose = trimesh.creation.cone(radius=0.06, height=0.15, sections=24)
    nose.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
    nose.apply_translation([0.0 - 0.075, 0, 0])
    parts.append(nose)

    # Main Wings (swept back airfoil profile approximated by tapered box)
    wings = trimesh.creation.box(extents=[0.22, 0.72, 0.018])
    wings.apply_translation([0.35, 0, 0.01])
    parts.append(wings)

    # Winglets (vertical tips)
    for side in [-1, 1]:
        winglet = trimesh.creation.box(extents=[0.08, 0.012, 0.08])
        winglet.apply_translation([0.35, side * 0.36, 0.04])
        parts.append(winglet)

    # Horizontal Stabilizers (tail wings)
    tail_wings = trimesh.creation.box(extents=[0.12, 0.28, 0.012])
    tail_wings.apply_translation([0.65, 0, 0.04])
    parts.append(tail_wings)

    # Vertical Stabilizer (tail fin)
    tail_fin = trimesh.creation.box(extents=[0.14, 0.015, 0.16])
    tail_fin.apply_translation([0.64, 0, 0.11])
    parts.append(tail_fin)

    # Jet Engines (under-wing nacelles)
    for side in [-1, 1]:
        engine = trimesh.creation.cylinder(radius=0.035, height=0.16, sections=18)
        engine.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        engine.apply_translation([0.33, side * 0.18, -0.04])
        parts.append(engine)

    combined = trimesh.util.concatenate(parts)
    return combined


def create_f1_car_mesh() -> trimesh.Trimesh:
    """Builds a Formula 1 racing car (chassis, cockpit halo, front/rear wings, sidepods, wheels)."""
    parts = []

    # Main Chassis tub (low, long, tapered)
    chassis = trimesh.creation.box(extents=[0.65, 0.14, 0.08])
    chassis.apply_translation([0.35, 0, 0.06])
    parts.append(chassis)

    # Nose cone
    nose = trimesh.creation.cone(radius=0.07, height=0.18, sections=16)
    nose.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
    nose.apply_translation([0.02, 0, 0.05])
    parts.append(nose)

    # Front Wing (wide, close to ground)
    front_wing = trimesh.creation.box(extents=[0.12, 0.42, 0.02])
    front_wing.apply_translation([0.02, 0, 0.025])
    parts.append(front_wing)

    # Front Wing Endplates
    for side in [-1, 1]:
        endplate = trimesh.creation.box(extents=[0.14, 0.012, 0.06])
        endplate.apply_translation([0.02, side * 0.21, 0.045])
        parts.append(endplate)

    # Sidepods
    for side in [-1, 1]:
        sidepod = trimesh.creation.box(extents=[0.28, 0.09, 0.07])
        sidepod.apply_translation([0.36, side * 0.12, 0.055])
        parts.append(sidepod)

    # Cockpit Halo / Airbox intake
    airbox = trimesh.creation.box(extents=[0.16, 0.07, 0.10])
    airbox.apply_translation([0.38, 0, 0.13])
    parts.append(airbox)

    # Rear Wing & Endplates
    rear_wing = trimesh.creation.box(extents=[0.14, 0.36, 0.02])
    rear_wing.apply_translation([0.66, 0, 0.16])
    parts.append(rear_wing)
    for side in [-1, 1]:
        rw_endplate = trimesh.creation.box(extents=[0.16, 0.012, 0.12])
        rw_endplate.apply_translation([0.66, side * 0.18, 0.12])
        parts.append(rw_endplate)

    # 4 Wheels (cylinders)
    wheel_coords = [
        (0.12, -0.20), (0.12, 0.20),   # Front wheels
        (0.58, -0.22), (0.58, 0.22)    # Rear wheels
    ]
    for wx, wy in wheel_coords:
        wheel = trimesh.creation.cylinder(radius=0.065, height=0.08, sections=20)
        # Rotate cylinder along Y-axis
        wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        wheel.apply_translation([wx, wy, 0.065])
        parts.append(wheel)

    combined = trimesh.util.concatenate(parts)
    return combined


def create_building_mesh() -> trimesh.Trimesh:
    """Builds an aerodynamic tapered skyscraper tower (reduces vortex shedding at high altitude)."""
    parts = []

    # Base section (strong square base with rounded chamfer)
    base = trimesh.creation.box(extents=[0.24, 0.24, 0.25])
    base.apply_translation([0.35, 0, 0.125])
    parts.append(base)

    # Mid section (tapered)
    mid = trimesh.creation.box(extents=[0.19, 0.19, 0.25])
    mid.apply_translation([0.35, 0, 0.375])
    parts.append(mid)

    # Upper section (slender)
    upper = trimesh.creation.box(extents=[0.14, 0.14, 0.25])
    upper.apply_translation([0.35, 0, 0.625])
    parts.append(upper)

    # Crown spire (aerodynamic cone tip)
    spire = trimesh.creation.cone(radius=0.06, height=0.18, sections=16)
    spire.apply_translation([0.35, 0, 0.75 + 0.09])
    parts.append(spire)

    combined = trimesh.util.concatenate(parts)
    return combined


def create_human_mesh() -> trimesh.Trimesh:
    """Builds a human figure (head, torso, arms, legs) representing a high-drag bluff body."""
    parts = []

    # Head (sphere)
    head = trimesh.creation.icosphere(subdivisions=2, radius=0.055)
    head.apply_translation([0.35, 0, 0.68])
    parts.append(head)

    # Torso (box / cylinder)
    torso = trimesh.creation.box(extents=[0.11, 0.22, 0.28])
    torso.apply_translation([0.35, 0, 0.48])
    parts.append(torso)

    # Arms (cylinders)
    for side in [-1, 1]:
        arm = trimesh.creation.cylinder(radius=0.032, height=0.28, sections=16)
        arm.apply_translation([0.35, side * 0.15, 0.45])
        parts.append(arm)

    # Legs (cylinders)
    for side in [-1, 1]:
        leg = trimesh.creation.cylinder(radius=0.045, height=0.34, sections=16)
        leg.apply_translation([0.35, side * 0.065, 0.17])
        parts.append(leg)

    combined = trimesh.util.concatenate(parts)
    return combined


def fit_mesh_to_wind_tunnel(mesh: trimesh.Trimesh,
                            target_center=(1.0, 0.5, 0.48),
                            max_span=(0.7, 0.6, 0.6)) -> trimesh.Trimesh:
    """
    Normalizes any input mesh to fit safely in the center of the wind tunnel test section.
    Leaves ample clearance around all sides for inlet jet flow and boundary conditions.
    """
    bounds = mesh.bounds
    current_size = bounds[1] - bounds[0]
    scale_factors = [max_span[i] / max(current_size[i], 1e-6) for i in range(3)]
    uniform_scale = min(scale_factors)

    mesh.apply_scale(uniform_scale)

    # Center mesh at target position
    current_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    translation = np.array(target_center) - current_center
    mesh.apply_translation(translation)

    return mesh


def generate_all_models(output_dir: Path | str = "assets/models"):
    """Generates all 4 starter models and saves them to disk."""
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        "airplane": create_airplane_mesh,
        "f1_car": create_f1_car_mesh,
        "building": create_building_mesh,
        "human": create_human_mesh
    }

    print("Generating starter 3D models for the exhibition...")
    for obj_id, gen_fn in generators.items():
        obj_folder = base_dir / obj_id
        obj_folder.mkdir(parents=True, exist_ok=True)
        stl_path = obj_folder / "model.stl"

        raw_mesh = gen_fn()
        normalized_mesh = fit_mesh_to_wind_tunnel(raw_mesh)
        normalized_mesh.export(str(stl_path))
        print(f" -> Created {stl_path} ({len(normalized_mesh.vertices)} vertices, {len(normalized_mesh.faces)} faces)")


if __name__ == "__main__":
    generate_all_models()
