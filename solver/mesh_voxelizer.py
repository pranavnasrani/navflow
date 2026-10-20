"""
Voxelizes 3D triangular surface meshes (.stl / .obj) into a 3D Cartesian grid.
Identifies solid cells (boundary condition u=v=w=0) and computes surface normal vectors
for integrating pressure forces (Aerodynamic Drag and Lift/Downforce).
"""

from pathlib import Path
import numpy as np
import trimesh


class MeshVoxelizer:
    def __init__(self,
                 grid_nx: int = 80,
                 grid_ny: int = 40,
                 grid_nz: int = 40,
                 domain_bounds: tuple = (2.0, 1.0, 1.0)):
        """
        grid_nx, grid_ny, grid_nz: number of grid cells along X, Y, Z
        domain_bounds: physical dimensions of wind tunnel (X_length, Y_width, Z_height)
        """
        self.nx = grid_nx
        self.ny = grid_ny
        self.nz = grid_nz
        self.Lx, self.Ly, self.Lz = domain_bounds

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz

        # Cell center coordinates in physical meters
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy
        self.z = (np.arange(self.nz) + 0.5) * self.dz

        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

    def load_and_voxelize(self, model_path: Path | str) -> dict:
        """
        Loads a 3D mesh (.stl / .obj), centers it in the test section,
        and generates:
        - solid_mask: 3D boolean array (True = solid body, False = air)
        - surface_mask: 3D boolean array of fluid cells adjacent to the solid
        - surface_normals: (N_surf, 3) normal vectors for force integration
        - surface_indices: tuple of (ix, iy, iz) indices of surface cells
        - frontal_area: computed projected frontal area (m^2)
        - mesh: the loaded, scaled trimesh object
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        mesh = trimesh.load(str(model_path), force="mesh")

        # Ensure correct placement in wind tunnel: center around (1.0, 0.5, 0.48)
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        max_allowed = np.array([self.Lx * 0.42, self.Ly * 0.65, self.Lz * 0.65])
        scale = min(max_allowed / np.maximum(size, 1e-6))
        mesh.apply_scale(scale)

        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        target = np.array([self.Lx * 0.50, self.Ly * 0.50, self.Lz * 0.48])
        mesh.apply_translation(target - center)

        # Check which cell centers are inside the mesh
        points = np.stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()], axis=-1)
        contains = mesh.contains(points)
        solid_mask = contains.reshape((self.nx, self.ny, self.nz))

        # Surface cells: fluid cells that have at least one solid neighbor
        # Using 6-connected neighbor stencil
        pad_solid = np.pad(solid_mask, 1, mode="constant", constant_values=False)
        solid_neighbors = (
            pad_solid[2:, 1:-1, 1:-1] | pad_solid[:-2, 1:-1, 1:-1] |
            pad_solid[1:-1, 2:, 1:-1] | pad_solid[1:-1, :-2, 1:-1] |
            pad_solid[1:-1, 1:-1, 2:] | pad_solid[1:-1, 1:-1, :-2]
        )
        surface_mask = (~solid_mask) & solid_neighbors

        # Estimate surface normals for each surface cell pointing into the fluid
        # Normal n = -grad(solid_indicator)
        grad_x = (pad_solid[2:, 1:-1, 1:-1].astype(float) - pad_solid[:-2, 1:-1, 1:-1].astype(float))
        grad_y = (pad_solid[1:-1, 2:, 1:-1].astype(float) - pad_solid[1:-1, :-2, 1:-1].astype(float))
        grad_z = (pad_solid[1:-1, 1:-1, 2:].astype(float) - pad_solid[1:-1, 1:-1, :-2].astype(float))

        nx_surf = -grad_x[surface_mask]
        ny_surf = -grad_y[surface_mask]
        nz_surf = -grad_z[surface_mask]
        norm = np.maximum(np.sqrt(nx_surf**2 + ny_surf**2 + nz_surf**2), 1e-9)
        normals = np.stack([nx_surf / norm, ny_surf / norm, nz_surf / norm], axis=-1)

        # Frontal projected area (integration of silhouette along X)
        # In 0606 calculus: projected area = sum of cell area dy * dz for any column containing solid
        has_solid_in_x = np.any(solid_mask, axis=0)
        frontal_area = float(np.sum(has_solid_in_x) * self.dy * self.dz)
        frontal_area = max(frontal_area, 0.01)

        return {
            "solid_mask": solid_mask,
            "surface_mask": surface_mask,
            "surface_normals": normals,
            "surface_indices": np.where(surface_mask),
            "frontal_area": frontal_area,
            "mesh": mesh,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "grid_shape": (self.nx, self.ny, self.nz)
        }
