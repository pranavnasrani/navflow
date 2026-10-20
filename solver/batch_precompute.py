"""
Batch Precomputation Engine for the 4-Monitor Navier-Stokes Exhibition.

For each object:
- Runs 10 discrete circular pipe Z-heights x 5 discrete wind speeds (= 50 CFD states)
- Computes surface pressure mapping for the 3D mesh (red stagnation vs blue suction)
- Pre-integrates 3D streamline trajectories emitted from the circular pipe hole
- Calculates live Navier-Stokes and Sustainability metrics
- Compresses everything into assets/cache/<object_id>_states.npz for <1ms RAM loading.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator

from solver.mesh_voxelizer import MeshVoxelizer
from solver.navier_stokes_3d import NavierStokes3DSolver


def generate_pipe_seed_points(pipe_z: float,
                              pipe_y: float = 0.5,
                              pipe_radius: float = 0.08,
                              num_points: int = 40) -> np.ndarray:
    """
    Generates 3D seed points across the circular pipe inlet for streamline tracing.
    Using circle geometry: r = R * sqrt(rand), theta = 2*pi*rand
    """
    np.random.seed(42)  # Deterministic seed for stable visual particles
    radii = pipe_radius * np.sqrt(np.random.uniform(0, 1, num_points))
    thetas = np.random.uniform(0, 2 * np.pi, num_points)

    y_seeds = pipe_y + radii * np.cos(thetas)
    z_seeds = pipe_z + radii * np.sin(thetas)
    x_seeds = np.full(num_points, 0.02)  # Just inside inlet face

    return np.column_stack([x_seeds, y_seeds, z_seeds])


def trace_streamlines_3d(u: np.ndarray,
                         v: np.ndarray,
                         w: np.ndarray,
                         x_coords: np.ndarray,
                         y_coords: np.ndarray,
                         z_coords: np.ndarray,
                         seeds: np.ndarray,
                         max_steps: int = 120,
                         dt_trace: float = 0.015) -> list[np.ndarray]:
    """
    Traces 3D streamlines from seed points through the velocity vector field
    using 4th-order Runge-Kutta / Euler integration (0606 Kinematics: pos = pos + v*dt).
    """
    # Create 3D velocity interpolators
    interp_u = RegularGridInterpolator((x_coords, y_coords, z_coords), u, bounds_error=False, fill_value=0.0)
    interp_v = RegularGridInterpolator((x_coords, y_coords, z_coords), v, bounds_error=False, fill_value=0.0)
    interp_w = RegularGridInterpolator((x_coords, y_coords, z_coords), w, bounds_error=False, fill_value=0.0)

    streamlines = []
    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = y_coords[0], y_coords[-1]
    z_min, z_max = z_coords[0], z_coords[-1]

    for seed in seeds:
        pos = seed.copy()
        traj = [pos.copy()]
        for _ in range(max_steps):
            if not (x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max and z_min <= pos[2] <= z_max):
                break
            u_val = float(np.squeeze(interp_u(pos)))
            v_val = float(np.squeeze(interp_v(pos)))
            w_val = float(np.squeeze(interp_w(pos)))
            vel = np.array([u_val, v_val, w_val])
            speed = np.linalg.norm(vel)
            if speed < 1e-3:
                break
            # Step position forward
            pos += vel * dt_trace
            traj.append(pos.copy())
        if len(traj) > 4:
            streamlines.append(np.array(traj))

    return streamlines


def map_pressure_to_mesh_vertices(mesh: trimesh.Trimesh,
                                   p: np.ndarray,
                                   x_coords: np.ndarray,
                                   y_coords: np.ndarray,
                                   z_coords: np.ndarray) -> np.ndarray:
    """
    Interpolates the 3D Cartesian pressure field p(x,y,z) onto each vertex of the 3D surface mesh.
    Allows PyVista to render realistic surface pressure contours (red = stagnation, blue = suction).
    """
    interp_p = RegularGridInterpolator((x_coords, y_coords, z_coords), p, bounds_error=False, fill_value=0.0)
    vertex_pressures = interp_p(mesh.vertices)
    return vertex_pressures.astype(np.float32)


def precompute_object(obj_config: dict,
                      wind_tunnel_cfg: dict,
                      solver_cfg: dict,
                      quick_mode: bool = False,
                      output_dir: Path | str = "assets/cache"):
    """
    Runs the full grid of (pipe_z, wind_speed) simulations for a single object.
    """
    obj_id = obj_config["id"]
    print(f"\n========================================================")
    print(f"Precomputing CFD States for: {obj_config['display_name']} ({obj_id})")
    print(f"========================================================")

    model_path = Path(f"assets/models/{obj_id}/model.stl")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model mesh at {model_path}. Run assets/generate_starter_models.py first.")

    nx = solver_cfg.get("grid_nx", 80)
    ny = solver_cfg.get("grid_ny", 40)
    nz = solver_cfg.get("grid_nz", 40)
    domain_size = tuple(wind_tunnel_cfg.get("dimensions", [2.0, 1.0, 1.0]))
    pipe_radius = wind_tunnel_cfg.get("pipe_radius", 0.08)

    voxelizer = MeshVoxelizer(grid_nx=nx, grid_ny=ny, grid_nz=nz, domain_bounds=domain_size)
    voxel_data = voxelizer.load_and_voxelize(model_path)
    mesh = voxel_data["mesh"]

    z_steps = wind_tunnel_cfg["z_steps"]
    wind_speeds = wind_tunnel_cfg["wind_speeds"]

    if quick_mode:
        z_steps = [z_steps[2], z_steps[5], z_steps[8]]  # 3 heights
        wind_speeds = [wind_speeds[1], wind_speeds[3]]    # 2 speeds
        num_solver_steps = 10
        poisson_iter = 25
    else:
        num_solver_steps = solver_cfg.get("precompute_steps", 25)
        poisson_iter = solver_cfg.get("max_poisson_iterations", 50)

    solver = NavierStokes3DSolver(
        nx=nx, ny=ny, nz=nz,
        domain_size=domain_size,
        pipe_radius=pipe_radius,
        density=wind_tunnel_cfg.get("air_density_kg_m3", 1.225),
        kinematic_viscosity=wind_tunnel_cfg.get("kinematic_viscosity_m2_s", 1.5e-5),
        dt=solver_cfg.get("time_step_dt", 0.005)
    )
    solver.set_solid_mask(voxel_data["solid_mask"])

    all_states = {}
    total_states = len(z_steps) * len(wind_speeds)
    state_counter = 0

    for z_idx, pz in enumerate(z_steps):
        for v_idx, ws in enumerate(wind_speeds):
            state_counter += 1
            print(f"[{state_counter}/{total_states}] Solving Z-step {z_idx+1}/{len(z_steps)} (Z={pz:.2f}m) @ Speed {ws:.1f} m/s...")

            solver.solve_state(pipe_z=pz, wind_speed=ws, num_steps=num_solver_steps, poisson_iter=poisson_iter)

            # Telemetry metrics
            metrics = solver.compute_forces_and_telemetry(
                surface_indices=voxel_data["surface_indices"],
                surface_normals=voxel_data["surface_normals"],
                frontal_area=voxel_data["frontal_area"],
                wind_speed=ws,
                object_info=obj_config
            )

            # Vertex surface pressure
            surf_p = map_pressure_to_mesh_vertices(mesh, solver.p, solver.x, solver.y, solver.z)

            # Streamlines from pipe hole
            seeds = generate_pipe_seed_points(pz, pipe_radius=pipe_radius, num_points=32)
            streamlines = trace_streamlines_3d(
                solver.u, solver.v, solver.w,
                solver.x, solver.y, solver.z,
                seeds=seeds
            )

            state_key = f"z{z_idx}_v{v_idx}"
            all_states[state_key] = {
                "z_val": float(pz),
                "v_val": float(ws),
                "metrics": metrics,
                "surface_pressure": surf_p,
                "streamlines": streamlines
            }

    # Save to disk
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"{obj_id}_states.npz"

    # Package into arrays/dictionaries
    # Also save mesh vertices and faces so the visualizer can rebuild the 3D surface instantly
    save_dict = {
        "mesh_vertices": mesh.vertices.astype(np.float32),
        "mesh_faces": mesh.faces.astype(np.int32),
        "z_steps": np.array(z_steps, dtype=np.float32),
        "wind_speeds": np.array(wind_speeds, dtype=np.float32),
        "config_json": json.dumps(obj_config)
    }

    # Flatten states for npz
    for k, v in all_states.items():
        save_dict[f"{k}_metrics"] = json.dumps(v["metrics"])
        save_dict[f"{k}_pressure"] = v["surface_pressure"]
        # Save streamlines as a ragged list of coordinate points
        # Joined with NaN delimiters for high-performance PyVista PolyData lines
        if len(v["streamlines"]) > 0:
            joined_lines = []
            for line in v["streamlines"]:
                joined_lines.append(line)
                joined_lines.append(np.array([[np.nan, np.nan, np.nan]]))
            save_dict[f"{k}_streamlines"] = np.vstack(joined_lines).astype(np.float32)
        else:
            save_dict[f"{k}_streamlines"] = np.empty((0, 3), dtype=np.float32)

    np.savez_compressed(cache_file, **save_dict)
    file_size_mb = cache_file.stat().st_size / (1024 * 1024)
    print(f"Successfully saved {cache_file} ({file_size_mb:.2f} MB, {total_states} CFD states)")


def run_batch_precompute(quick: bool = False):
    """Loads configuration and runs precomputation for all 4 objects."""
    cfg_path = Path("config/exhibition_config.json")
    with open(cfg_path, "r") as f:
        config = json.load(f)

    for obj in config["objects"]:
        precompute_object(
            obj_config=obj,
            wind_tunnel_cfg=config["wind_tunnel"],
            solver_cfg=config.get("solver_defaults", {}),
            quick_mode=quick
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute CFD states for the exhibition")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing (3 Z-steps x 2 speeds)")
    args = parser.parse_args()

    run_batch_precompute(quick=args.quick)
