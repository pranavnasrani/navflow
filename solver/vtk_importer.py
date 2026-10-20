"""
VTK / VTU Importer for ANSYS Fluent and OpenFOAM simulation exports.
Enables importing high-resolution CFD data from ANSYS Fluent (exported as .vtk / .vtu / .case)
or OpenFOAM into the exhibition's compact .npz RAM cache.
"""

from pathlib import Path
import json
import numpy as np
import pyvista as pv
import trimesh

from solver.batch_precompute import generate_pipe_seed_points


class VTKSimulationImporter:
    def __init__(self,
                 wind_tunnel_bounds: tuple = (2.0, 1.0, 1.0),
                 density: float = 1.225):
        self.bounds = wind_tunnel_bounds
        self.rho = density

    def import_vtk_state(self,
                         vtk_file: Path | str,
                         model_mesh: trimesh.Trimesh,
                         pipe_z: float,
                         wind_speed: float,
                         object_info: dict) -> dict:
        """
        Reads a VTK/VTU mesh file, extracts velocity vector field 'U' (or 'Velocity')
        and scalar pressure 'p' (or 'Pressure'), traces streamlines, and computes telemetry.
        """
        vtk_path = Path(vtk_file)
        if not vtk_path.exists():
            raise FileNotFoundError(f"VTK file not found: {vtk_path}")

        print(f"Reading VTK data from {vtk_path.name}...")
        grid = pv.read(str(vtk_path))

        # Detect velocity array name
        vel_name = None
        for cand in ["U", "Velocity", "velocity", "V"]:
            if cand in grid.point_data:
                vel_name = cand
                break
            if cand in grid.cell_data:
                vel_name = cand
                grid = grid.cell_data_to_point_data()
                break

        # Detect pressure array name
        p_name = None
        for cand in ["p", "Pressure", "pressure", "P"]:
            if cand in grid.point_data:
                p_name = cand
                break
            if cand in grid.cell_data:
                p_name = cand
                grid = grid.cell_data_to_point_data()
                break

        if vel_name is None:
            raise KeyError(f"Could not find velocity array in VTK file. Available point arrays: {grid.point_data.keys()}")
        if p_name is None:
            raise KeyError(f"Could not find pressure array in VTK file. Available point arrays: {grid.point_data.keys()}")

        # Interpolate pressure directly onto the model mesh vertices
        surf_pv = pv.PolyData(model_mesh.vertices)
        sampled = surf_pv.sample(grid)
        surface_pressure = np.nan_to_num(sampled.point_data[p_name], nan=0.0).astype(np.float32)

        # Generate seed points at circular pipe hole
        seeds = generate_pipe_seed_points(pipe_z=pipe_z, pipe_radius=0.08, num_points=32)
        seed_poly = pv.PolyData(seeds)

        # Compute streamlines using VTK's high-order Runge-Kutta integrator
        streamlines_poly = grid.streamlines_from_source(
            source=seed_poly,
            vectors=vel_name,
            max_time=2.0,
            integration_direction='forward'
        )

        # Extract coordinates of streamline lines
        lines_coords = streamlines_poly.points if streamlines_poly.n_points > 0 else np.empty((0, 3))

        # Compute aerodynamic forces from surface pressure
        # Drag = sum(p * normal_x * dA)
        tri_centers = model_mesh.triangles_center
        tri_normals = model_mesh.face_normals
        tri_areas = model_mesh.area_faces

        tri_pv = pv.PolyData(tri_centers)
        sampled_tri = tri_pv.sample(grid)
        tri_p = np.nan_to_num(sampled_tri.point_data[p_name], nan=0.0)

        drag_force = float(np.sum(np.maximum(tri_p, 0.0) * np.abs(tri_normals[:, 0]) * tri_areas))
        lift_force = float(np.sum(tri_p * tri_normals[:, 2] * tri_areas))
        drag_force = max(drag_force, 0.1)

        # Frontal area
        frontal_area = float(np.sum(np.abs(tri_normals[:, 0]) * tri_areas) * 0.5)
        frontal_area = max(frontal_area, 0.01)

        # Drag coefficient
        dynamic_p = 0.5 * self.rho * (wind_speed ** 2)
        cd = float(np.clip(drag_force / (max(dynamic_p * frontal_area, 1e-4)), 0.05, 2.5))

        drag_power_kw = (drag_force * wind_speed) / 1000.0
        fuel_factor = object_info.get("fuel_burn_factor_l_per_kwh", 0.30)
        co2_factor = object_info.get("co2_kg_per_l_fuel", 2.45)
        baseline_kw = object_info.get("baseline_drag_power_kw", 500.0) * ((wind_speed / 30.0) ** 3)

        fuel_burn = drag_power_kw * fuel_factor
        baseline_fuel = baseline_kw * fuel_factor
        fuel_saved = max(baseline_fuel - fuel_burn, 0.0)
        co2_saved = fuel_saved * co2_factor
        efficiency_gain = max(0.0, (1.0 - (drag_power_kw / max(baseline_kw, 1e-3))) * 100.0)

        metrics = {
            "wind_speed_m_s": float(wind_speed),
            "reynolds_number": float((wind_speed * 0.8) / 1.5e-5),
            "drag_force_n": float(drag_force),
            "lift_force_n": float(lift_force),
            "drag_coefficient_cd": float(cd),
            "frontal_area_m2": float(frontal_area),
            "drag_power_kw": float(drag_power_kw),
            "fuel_burn_l_per_hr": float(fuel_burn),
            "fuel_saved_l_per_hr": float(fuel_saved),
            "co2_saved_kg_per_hr": float(co2_saved),
            "efficiency_gain_pct": float(efficiency_gain),
            "max_pressure_pa": float(np.max(tri_p))
        }

        return {
            "surface_pressure": surface_pressure,
            "streamlines_points": lines_coords.astype(np.float32),
            "metrics": metrics
        }
