"""The three terms of the Navier-Stokes momentum equation, evaluated on an OpenFOAM flow field.

Per unit mass (all in m/s^2, with kinematic pressure p/rho and kinematic viscosity nu):

    (v . grad) v   =   - grad p   +   nu laplacian(v)
    convective         pressure       viscous

Maths link (IGCSE 0606):
  * A partial derivative dv_i/dx_j is a rate of change: a small increment delta v_i divided by a
    small step delta x_j. The mesh gives exactly such increments between neighbouring points.
  * The convective term is the CHAIN RULE. A fluid particle at position x(t) feels the change
        dv_i/dt = sum_j (dv_i/dx_j)(dx_j/dt) = sum_j v_j dv_i/dx_j,
    because dx_j/dt is the velocity component v_j. That sum is a connected rate of change.
  * The viscous term uses SECOND derivatives (friction): laplacian(v_i) = sum_j d/dx_j (dv_i/dx_j),
    i.e. differentiate twice, once per direction, and add. Where the velocity curves strongly,
    neighbouring layers of fluid drag on each other.
  * The pressure term is the gradient |grad p|: the steepness of the pressure "hill".
  * Each magnitude is a 3D vector length sqrt(a^2 + b^2 + c^2).
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

CONVECTIVE, PRESSURE, VISCOUS = 0, 1, 2  # values of `ns_dominant`
_TERM_NAMES = ("convective", "pressure", "viscous")


def _point_array(grid: pv.DataSet, name: str) -> np.ndarray:
    """Point values of `name`, interpolated from cell data if the file only has cell data."""
    if name in grid.point_data:
        return np.asarray(grid.point_data[name])
    return np.asarray(grid.cell_data_to_point_data().point_data[name])


def _derivative(grid: pv.DataSet, scalars: str, gradient: str | bool = False, divergence: str | bool = False) -> np.ndarray:
    """One VTK derivative pass on point data, returned as a float array.

    `faster=True` is deliberate: VTK's exact per-point mode inverts every neighbouring cell's
    Jacobian, which is about 10x slower and returns garbage (values near 1e11) at the degenerate
    polyhedra snappyHexMesh leaves next to the model. The fast mode averages cell derivatives
    instead; it is exact for linear fields and close elsewhere.
    """
    name = gradient if gradient else divergence
    out = grid.compute_derivative(scalars=scalars, gradient=gradient, divergence=divergence, faster=True)
    return np.asarray(out[name], dtype=float)


def ns_term_fields(grid: pv.DataSet, nu: float, u_name: str, p_name: str) -> pv.DataSet:
    """Return a copy of `grid` with point arrays ns_convective, ns_pressure, ns_viscous, ns_dominant.

    `ns_dominant` is the index (0 convective, 1 pressure, 2 viscous) of the largest term, int8.
    `p_name` must be kinematic pressure (p / rho), as OpenFOAM's incompressible solvers write it.
    """
    out = grid.copy()
    out.point_data[u_name] = _point_array(grid, u_name)
    out.point_data[p_name] = _point_array(grid, p_name)
    v = np.asarray(out.point_data[u_name], dtype=float)

    # Velocity gradient: row i, column j holds dv_i/dx_j. VTK lays the 9 values out row by row.
    jac = _derivative(out, u_name, gradient="vtk_grad_u")
    jac = jac.reshape(-1, 3, 3)

    # Chain rule: (v . grad) v_i = sum_j v_j dv_i/dx_j
    convective = np.linalg.norm(np.einsum("nj,nij->ni", v, jac), axis=1)

    # |grad p|: one more first derivative
    grad_p = _derivative(out, p_name, gradient="vtk_grad_p")
    pressure = np.linalg.norm(grad_p, axis=1)

    # laplacian(v_i) = sum_j d/dx_j (dv_i/dx_j): the divergence of the vector (dv_i/dx_0, dv_i/dx_1, dv_i/dx_2),
    # a second compute_derivative pass.
    lap = np.empty((len(v), 3))
    for i in range(3):
        out.point_data["vtk_grad_row"] = np.ascontiguousarray(jac[:, i, :])
        lap[:, i] = _derivative(out, "vtk_grad_row", divergence="vtk_div_row")
    viscous = nu * np.linalg.norm(lap, axis=1)

    for name in ("vtk_grad_u", "vtk_grad_p", "vtk_grad_row", "vtk_div_row"):
        if name in out.point_data:
            del out.point_data[name]
    out.point_data["ns_convective"] = convective
    out.point_data["ns_pressure"] = pressure
    out.point_data["ns_viscous"] = viscous
    out.point_data["ns_dominant"] = np.argmax(np.column_stack([convective, pressure, viscous]), axis=1).astype(np.int8)
    return out


def ns_term_summary(grid: pv.DataSet, bounds6: tuple[float, ...]) -> dict[str, float]:
    """Mean of each term's magnitude over the points inside the box (xmin, xmax, ymin, ymax, zmin, zmax)."""
    lo = np.array(bounds6[0::2], dtype=float)
    hi = np.array(bounds6[1::2], dtype=float)
    inside = np.all((grid.points >= lo) & (grid.points <= hi), axis=1)
    if not inside.any():
        return {name: 0.0 for name in _TERM_NAMES}
    return {name: float(np.mean(grid.point_data[f"ns_{name}"][inside])) for name in _TERM_NAMES}
